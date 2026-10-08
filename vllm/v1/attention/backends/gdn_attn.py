# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass, replace
from typing import Literal

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    checkpoint: MambaPrefillCheckpointMetadata | None = None
    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_sequence_masks_cpu: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    uniform_spec_sequence_length: int | None = None  # None for ragged batches

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None
    # Reusable AITER host/device chunk schedule.  Built once per batch so every
    # GDN layer avoids an offsets device-to-host synchronization.
    aiter_prefill_metadata: object | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


_RADIANCE_GDN_META = __import__("os").environ.get("RADIANCE_GDN_META", "1") == "1"


def _radiance_arange(builder, n: int, device) -> torch.Tensor:
    """arange(n) as a slice of one cached buffer -- read-only downstream.

    Never populate the cache during a graph capture: an allocation made there comes from the
    graph's private pool and is only valid inside a replay, so a buffer meant to outlive the
    step would be reading reused memory. Capture falls back to the stock allocation, and the
    first ordinary build fills the cache.
    """
    c = getattr(builder, "_radiance_arange_buf", None)
    if c is None or c.numel() < n or c.device != device:
        if torch.cuda.is_current_stream_capturing():
            return torch.arange(n, dtype=torch.int32, device=device)
        c = torch.arange(max(n, 4096), dtype=torch.int32, device=device)
        builder._radiance_arange_buf = c
    return c[:n]


def _radiance_empty_idx(builder, device) -> torch.Tensor:
    c = getattr(builder, "_radiance_empty_buf", None)
    if c is None or c.device != device:
        if torch.cuda.is_current_stream_capturing():
            return torch.empty(0, dtype=torch.int32, device=device)
        c = torch.empty(0, dtype=torch.int32, device=device)
        builder._radiance_empty_buf = c
    return c


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.checkpoint_builder = MambaPrefillCheckpointBuilder(
            vllm_config, kv_cache_spec
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal[
            "triton", "flashinfer", "cutedsl", "aiter", "aiter_flydsl"
        ]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)
        self._check_chunk_metadata_override(type(self), self.gdn_prefill_backend)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        # update_block_table() keeps the source group's batch-level FULL graph
        # buffers, so only MRV2, which also reuses metadata at capture, may use
        # it.
        self.supports_update_block_table = (
            vllm_config.use_v2_model_runner
            and device.type == "cuda"
            # Not isinstance: KDA's RecoverSSM/checkpoint metadata is per group.
            and type(self) is GDNAttentionMetadataBuilder
        )
        if self.supports_update_block_table:
            # Opts into MRV2's CUDA-only aligned-index precompute.
            self.mamba_aligned_state_indices: torch.Tensor | None = None

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    @staticmethod
    def _check_chunk_metadata_override(
        builder_cls: type["GDNAttentionMetadataBuilder"], backend: str
    ) -> None:
        """Reject a subclass whose chunk metadata the AITER path would skip.

        AITER brings its own varlen prefill metadata and never calls
        ``_build_chunk_metadata``, so a builder that overrides it would lose
        that override without a word. No in-tree subclass can get here --
        ``_resolve_gdn_prefill_backend`` only selects this backend for models
        with 128-dim GDN key and value heads, which the KDA builders are not --
        but a future one should be told rather than quietly ignored.
        """
        if backend != "aiter_flydsl":
            return
        if (
            builder_cls._build_chunk_metadata
            is GDNAttentionMetadataBuilder._build_chunk_metadata
        ):
            return
        raise RuntimeError(
            f"{builder_cls.__name__} builds its own FLA chunk metadata, which "
            "the 'aiter_flydsl' GDN prefill backend does not use. Select a "
            "different gdn_prefill_backend for this model."
        )

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            assert prefill_query_start_loc is not None
            assert prefill_query_start_loc_cpu is not None
            total_tokens = int(prefill_query_start_loc_cpu[-1].item())
            return prepare_metadata_cutedsl(
                prefill_query_start_loc,
                total_tokens,
                FLA_CHUNK_SIZE,
            )

        # Only prefill batches use FLA chunk ops.
        # Pre-compute on CPU and async-copy to GPU to avoid
        # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        assert prefill_query_start_loc_cpu is not None
        return (
            async_tensor_h2d(
                prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
            async_tensor_h2d(
                prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
        )

    # radiance (patch_gdn_shared_build.py): the runner passes a per-step dict here.
    _radiance_shared_build = True

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        _radiance_shared: dict | None = None,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        # radiance (patch_gdn_shared_build.py): a previous GDN group already built this
        # step's metadata. Only the block-table-derived indices are per-group; everything
        # else is copied buffer-to-buffer from the first group's already-padded buffers
        # into THIS builder's buffers (each group's cudagraph captured its own addresses).
        if _radiance_shared and _radiance_shared.get("kind") == "spec_fast":
            sh = _radiance_shared
            import dataclasses as _dc
            bt = mamba_get_block_table_tensor(
                m.block_table_tensor, m.seq_lens, self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )
            nsd = sh["nsd"]
            bs = m.num_reqs
            if sh["mask_all"]:
                sidx = bt[: sh["mask_rows"], : self.num_spec + 1]
            else:
                sidx = bt[sh["mask_cpu"], : self.num_spec + 1]
            self.spec_state_indices_tensor[:nsd].copy_(sidx, non_blocking=True)
            sst = self.spec_state_indices_tensor[:bs]
            sst[nsd:].fill_(NULL_BLOCK_ID)
            self.spec_sequence_masks[:bs].copy_(sh["masks_src"], non_blocking=True)
            self.spec_token_indx[: sh["si_n"]].copy_(sh["si_src"], non_blocking=True)
            self.non_spec_token_indx[: sh["nsi_n"]].copy_(
                sh["nsi_src"], non_blocking=True)
            self.spec_query_start_loc[: bs + 1].copy_(sh["qsl_src"], non_blocking=True)
            self.num_accepted_tokens[:bs].copy_(sh["acc_src"], non_blocking=True)
            return _dc.replace(
                sh["md"],
                spec_state_indices_tensor=sst,
                spec_sequence_masks=self.spec_sequence_masks[:bs],
                spec_token_indx=self.spec_token_indx[: sh["si_n"]],
                non_spec_token_indx=self.non_spec_token_indx[: sh["nsi_n"]],
                spec_query_start_loc=self.spec_query_start_loc[: bs + 1],
                num_accepted_tokens=self.num_accepted_tokens[:bs],
            )
        _rad_fastpath = False
        _rad_mask_all = False

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        aligned_state_indices = getattr(self, "mamba_aligned_state_indices", None)
        if aligned_state_indices is not None:
            block_table_tensor = aligned_state_indices[: m.num_reqs]
        else:
            block_table_tensor = mamba_get_block_table_tensor(
                m.block_table_tensor,
                m.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )

        uniform_spec_sequence_length = None
        spec_sequence_masks_cpu: torch.Tensor | None = None
        # --- RADIANCE (patch_gdn_metadata.py): one numpy pass over the same buffer
        # instead of mask / index / sum / item and then the mask a second time.
        _r_np = _RADIANCE_GDN_META and num_decode_draft_tokens_cpu is not None
        _ndt_np = num_decode_draft_tokens_cpu.numpy() if _r_np else None
        _mask_np = None if _ndt_np is None else _ndt_np >= 0
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            if _r_np:
                spec_sequence_masks_cpu = torch.from_numpy(_mask_np)
                num_spec_decodes = int(_mask_np.sum())
            else:
                spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
                num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            # A batch whose rows all drafted nothing still has to run the
            # speculative path: that is the only path that applies each row's
            # accepted-token offset to the recurrent state.
            if num_spec_decodes == 0:
                num_spec_decodes = 0
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )

        if spec_sequence_masks is None:
            # V2 already excludes prefills from full decode graphs via
            # has_prefill. Classify first chunks as prefills to mask recycled
            # state; resumed one-token chunks can still use the decode kernels.
            assert m.seq_lens_cpu_upper_bound is not None
            query_lens_cpu = query_start_loc_cpu.diff()
            no_prior_state = (query_lens_cpu > 0) & (
                m.seq_lens_cpu_upper_bound <= query_lens_cpu
            )
            # Capture batches also have seq_len == query_len, but are not
            # prefills.
            if m.is_prefilling is not None:
                no_prior_state &= m.is_prefilling
            else:
                no_prior_state = torch.zeros_like(no_prior_state)
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    m.replace(is_prefilling=no_prior_state),
                    decode_threshold=1,
                    treat_short_extends_as_decodes=False,
                )
            )
            # Exclude trailing padding from both prefill counts.
            if num_prefills:
                num_prefills -= int((query_lens_cpu[num_decodes:] == 0).sum())
                num_prefill_tokens = (
                    int(query_start_loc_cpu[num_decodes + num_prefills])
                    - num_decode_tokens
                )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            spec_query_lens_cpu = query_lens_cpu[spec_sequence_masks_cpu]
            if spec_query_lens_cpu.numel() > 0:
                first_spec_sequence_length = int(spec_query_lens_cpu[0])
                if first_spec_sequence_length > 0 and bool(
                    torch.all(spec_query_lens_cpu == first_spec_sequence_length)
                ):
                    uniform_spec_sequence_length = first_spec_sequence_length

            # Use CPU tensors to avoid CPU-GPU sync
            if _r_np:
                # --- RADIANCE: same integers, one numpy pass. `size` is the numpy
                # spelling of `size(0)` for these 1-D per-request vectors.
                _qlen_np = query_lens_cpu.numpy()
                _nonspec_np = _qlen_np[~_mask_np]
                num_decodes = int((_nonspec_np == 1).sum())
                num_zero_len = int((_nonspec_np == 0).sum())
                num_prefills = _nonspec_np.size - num_decodes - num_zero_len
                num_decode_tokens = num_decodes
                num_prefill_tokens = int(_nonspec_np.sum()) - num_decode_tokens
                num_spec_decode_tokens = (
                    int(_qlen_np.sum()) - num_prefill_tokens - num_decode_tokens
                )
            else:
                non_spec_query_lens_cpu = query_lens_cpu[non_spec_sequence_masks_cpu]
                num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
                # Exclude zero-length padded sequences from prefill count.
                num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
                num_prefills = (
                    non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
                )
                num_decode_tokens = num_decodes
                num_prefill_tokens = (
                    non_spec_query_lens_cpu.sum().item() - num_decode_tokens
                )
                num_spec_decode_tokens = (
                    query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
                )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                if _RADIANCE_GDN_META:
                    # --- RADIANCE: both of these depend only on their length and are
                    # read once, by the copy_ into the persistent buffers below.
                    spec_token_indx = _radiance_arange(
                        self, spec_token_size, query_start_loc.device
                    )
                    non_spec_token_indx = _radiance_empty_idx(
                        self, query_start_loc.device
                    )
                else:
                    spec_token_indx = torch.arange(
                        spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    non_spec_token_indx = torch.empty(
                        0, dtype=torch.int32, device=query_start_loc.device
                    )
                # Filter by spec_sequence_masks to exclude padded sequences
                # --- RADIANCE: when every sequence is a spec decode the mask selects
                # every row, so a slice carries the same values without the gather.
                # Only taken when the cudagraph branch below will consume it, since
                # that consumer is a copy_ and a strided source is fine there.
                if (
                    _RADIANCE_GDN_META
                    and _r_np
                    and self.use_full_cuda_graph
                    and num_spec_decodes <= self.decode_cudagraph_max_bs
                    and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
                    and bool(_mask_np.all())
                ):
                    spec_state_indices_tensor = block_table_tensor[
                        : _mask_np.size, : self.num_spec + 1
                    ]
                    _rad_mask_all = True
                else:
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            if num_prefills == 0 and num_decodes == 0:
                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
            else:
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        aiter_prefill_metadata: object | None = None
        if num_prefills > 0:
            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            if self.gdn_prefill_backend == "aiter_flydsl":
                # AITER carries its own reusable varlen metadata and has no use
                # for FLA's chunk indices, so it replaces them rather than
                # extending what _build_chunk_metadata returns.
                assert prefill_query_start_loc_cpu is not None
                aiter_prefill_metadata = (
                    rocm_aiter_ops.build_gdn_flydsl_prefill_metadata(
                        torch.diff(prefill_query_start_loc_cpu).tolist(),
                        cu_seqlens=prefill_query_start_loc,
                    )
                )
            else:
                chunk_indices, chunk_offsets = self._build_chunk_metadata(
                    prefill_query_start_loc,
                    prefill_query_start_loc_cpu,
                    query_start_loc.device,
                )

            if self.gdn_prefill_backend == "aiter":
                from aiter.ops.triton.gated_delta_net import (
                    build_gated_delta_rule_prefill_metadata,
                )
                from vllm.third_party.flash_linear_attention.ops.utils import (
                    FLA_CHUNK_SIZE,
                )

                prefill_seq_lens_cpu = (
                    prefill_query_start_loc_cpu[1:]
                    - prefill_query_start_loc_cpu[:-1]
                ).tolist()
                aiter_prefill_metadata = (
                    build_gated_delta_rule_prefill_metadata(
                        prefill_seq_lens_cpu,
                        cu_seqlens=prefill_query_start_loc,
                        chunk_size=FLA_CHUNK_SIZE,
                    )
                )

        if num_prefills > 0:
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        checkpoint: MambaPrefillCheckpointMetadata | None = None
        if num_prefills > 0:
            request_rows = list(range(m.num_reqs))
            if spec_sequence_masks_cpu is not None:
                request_rows = (~spec_sequence_masks_cpu).nonzero().flatten().tolist()
            checkpoint = self.checkpoint_builder.build(m, request_rows)

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if self._stage_spec_decode(
            num_prefills, num_decodes, num_spec_decodes, num_spec_decode_tokens
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)
            _rad_fastpath = True

        if self._stage_decode(num_prefills, num_decodes, num_spec_decodes):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            checkpoint=checkpoint,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            aiter_prefill_metadata=aiter_prefill_metadata,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            uniform_spec_sequence_length=uniform_spec_sequence_length,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        # radiance (patch_gdn_shared_build.py): make this build reusable by the step's
        # remaining GDN groups. Only the steady spec-decode cudagraph shape qualifies.
        if (
            _radiance_shared is not None
            and not _radiance_shared
            and _rad_fastpath
            and num_prefills == 0
            and num_decodes == 0
        ):
            _radiance_shared.update(
                kind="spec_fast", md=attn_metadata, nsd=num_spec_decodes,
                mask_all=_rad_mask_all,
                mask_rows=(spec_sequence_masks_cpu.numel()
                           if spec_sequence_masks_cpu is not None else 0),
                mask_cpu=spec_sequence_masks_cpu,
                masks_src=spec_sequence_masks,
                si_src=spec_token_indx, si_n=spec_token_indx.size(0),
                nsi_src=non_spec_token_indx, nsi_n=non_spec_token_indx.size(0),
                qsl_src=spec_query_start_loc, acc_src=num_accepted_tokens,
            )
        return attn_metadata

    def _stage_spec_decode(
        self,
        num_prefills: int,
        num_decodes: int,
        num_spec_decodes: int,
        num_spec_decode_tokens: int,
    ) -> bool:
        """Whether spec-decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _stage_decode(
        self, num_prefills: int, num_decodes: int, num_spec_decodes: int
    ) -> bool:
        """Whether decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        )

    def update_block_table(
        self,
        metadata: GDNAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNAttentionMetadata:
        """Re-gather this group's state indices. The other fields are
        batch-level and stay shared with ``metadata``."""
        m = metadata
        checkpoint = (
            m.checkpoint.regather_state_indices(blk_table)
            if m.checkpoint is not None
            else None
        )
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            assert self.mamba_aligned_state_indices is not None
            blk_table = self.mamba_aligned_state_indices
        masks = m.spec_sequence_masks_cpu
        spec_indices = non_spec_indices = prefill_indices = None
        if masks is None:
            non_spec_indices = blk_table[:, 0]
            if m.num_prefills > 0:
                prefill_indices = non_spec_indices[m.num_decodes :]
        elif m.num_prefills == 0:
            # Same as build(): padded sequences trail the spec decodes.
            spec_indices = blk_table[: m.num_spec_decodes, : self.num_spec + 1]
        else:
            spec_indices = blk_table[masks, : self.num_spec + 1]
            non_spec_indices = prefill_indices = blk_table[~masks, 0]

        if self._stage_spec_decode(
            m.num_prefills, m.num_decodes, m.num_spec_decodes, m.num_spec_decode_tokens
        ):
            assert m.spec_state_indices_tensor is not None
            batch_size = m.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[: m.num_spec_decodes].copy_(
                spec_indices, non_blocking=True
            )
            spec_indices = self.spec_state_indices_tensor[:batch_size]
            spec_indices[m.num_spec_decodes :].fill_(NULL_BLOCK_ID)

        if self._stage_decode(m.num_prefills, m.num_decodes, m.num_spec_decodes):
            assert m.non_spec_state_indices_tensor is not None
            batch_size = m.non_spec_state_indices_tensor.shape[0]
            self.non_spec_state_indices_tensor[: m.num_decodes].copy_(
                non_spec_indices, non_blocking=True
            )
            non_spec_indices = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_indices[m.num_decodes :].fill_(NULL_BLOCK_ID)

        return replace(
            m,
            spec_state_indices_tensor=spec_indices,
            non_spec_state_indices_tensor=non_spec_indices,
            prefill_state_indices=prefill_indices,
            checkpoint=checkpoint,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = torch.diff(m.query_start_loc_cpu).sub_(1)
        assert num_decode_draft_tokens_cpu.shape == num_accepted_tokens.shape

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
