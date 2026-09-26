# Radiance on baremetal (uv)

This directory turns the fork into the **vllm-radiance** distribution without Docker, so vLLM can
be built and installed on a bare-metal ROCm host with `uv` managing the Python environment.

The fork source already carries the radiance vLLM changes (the `patch_*.py` overlays applied
in-tree). This directory supplies the rest of what the vllm-radiance image used to supply at build
time:

| Asset | Location | Installed by |
|---|---|---|
| Runtime modules (`radiance_kernels`, `radiance_gdn`, ...) | repo root, `radiance_*.py` | wheel (`py_modules` in `setup.py`) |
| `pth`-based amdsmi startup hook | `radiance_amdsmi.pth`, `radiance_amdsmi.py` | `bootstrap.sh` |
| Tuned FP8 / MoE / MXFP4 configs | `radiance/configs/` | `bootstrap.sh` |
| `libr4d` gfx1201 kernel library (`r4d.so`) | pinned source + `patches/r4d_radiance_extras.patch` | `bootstrap.sh` |
| MXFP4/W4A8 HIP extension (`radiance_mxfp4_fp8.so`) | `radiance_mxfp4_fp8.hip` | `bootstrap.sh` |
| Out-of-tree patches (torch / transformers / aiter) | `radiance/patches/` | `bootstrap.sh` |
| Qualified runtime env defaults | `radiance/radiance-env.sh` | sourced by the operator |

## Quick start

The default stack mode mirrors the proven **ROCm 10 + AMD whl-next** flow for gfx1201
(`torch 2.13.0+rocm10.0.0`), which matches this fork's `torch == 2.13.0` pin:

```bash
# Host: ROCm 10 + amdrocm-core-dev10.0, hipcc on PATH, uv on PATH.
./radiance/bootstrap.sh                 # amd-wheel stack (default)

# ...or if you already built the env the manual way:
RADIANCE_STACK_MODE=skip ./radiance/bootstrap.sh

source radiance/radiance-env.sh
vllm serve <model> --tensor-parallel-size 8 ...
```

`bootstrap.sh` in `amd-wheel` mode performs the same Phase 1–4 steps as the manual guide
(sysbuild tools, `amd_smi`, `requirements/rocm.txt`, AMD whl-next torch/torchvision/torchaudio,
`amd-quark` removal, the LLVM library dedup symlinks, `--no-build-isolation` build), then layers on
the radiance extras: libr4d, the MXFP4/W4A8 HIP extension, tuned configs, the amdsmi `.pth`, and the
out-of-tree patches.

Useful knobs: `RADIANCE_VENV`, `RADIANCE_STACK_MODE=amd-wheel|auto|skip`,
`RADIANCE_ROCM_ROOT` (defaults to `/opt/rocm/core-10.0` when present), `RADIANCE_TORCH_BACKEND`,
`RADIANCE_INSTALL_MODE=editable|wheel`, `RADIANCE_INSTALL_AITER=1` (+ `RADIANCE_AITER_VERSION`,
`RADIANCE_AITER_COMMIT`, `RADIANCE_AITER_SPEC`), `RADIANCE_R4D_DIR`,
`RADIANCE_SKIP_{DEPS,R4D,HIPEXT,PATCHES,CONFIGS,LLVM_LINK,SMOKE}=1`.

> `amd-wheel` mode uninstalls `amd-quark`, matching the manual guide. The radiance Quark paths
> (`RADIANCE_MXFP4*`, `RADIANCE_QUARK_BF16_MTP`) are default-off and guarded, so non-Quark
> checkpoints are unaffected.
>
> **AITER is not installed by default.** It is not a vLLM dependency (the fork guards its import),
> and the R4D path does not need it. Set `RADIANCE_INSTALL_AITER=1` to build AITER for gfx1201 — it
> provides the `ROCM_AITER_UNIFIED_ATTN` fallback backend, the preshuffle FP8 blockscale GEMM
> (`RADIANCE_PRESHUFFLE`), the GDN AITER knobs, and the targets of `patch_unified_attention_lds`
> plus `patch_radiance_dispatch`'s `SPLITK` hunk. No `amd-aiter` wheel is published (the AMD
> whl-next index and PyPI both lack it), so this is a source build. The qualified image pinned
> **0.1.20** (`fc2e5d57`) for torch 2.12 / ROCm 7.14; because this host is torch 2.13 / ROCm 10, the
> bootstrap defaults to the current tag **0.1.23** (`50da036a`). Override with
> `RADIANCE_AITER_VERSION` / `RADIANCE_AITER_COMMIT` / `RADIANCE_AITER_SPEC`. Changing the AITER
> version changes the AITER-backed tuning paths, so treat it as a re-qualification. When AITER is
> absent, the aiter out-of-tree patch and the MXFP4 tile copy skip with a warning.

## Port status versus vllm-radiance v0.28.0

The radiance overlays were authored as exact-string patches against **vLLM v0.28.0**
(`2cf0a691`). This fork tracks a **newer upstream base** (~1,750 commits later), so each overlay was
classified rather than blindly applied:

### Applied in-tree (the fork *is* the patched source)

`patch_gfx1201`, `patch_radiance_dispatch` (vLLM hunk), `patch_skinny_gemm`, `patch_gdn_wmma`,
`patch_gdn_aiter_prefill`, `patch_preshuffle`, `install_radiance_hooks`, `patch_unpad`,
`patch_mtp_mm_mask`, `patch_mtp_loopbreak`, `patch_qwen3_toolparse`, `patch_conv1d_blockn`,
`patch_r4d`, `patch_dflash_base`, `patch_dflash_fused_kv_fp8`, `patch_dflash_w4`,
`patch_dflash_selector_topk`, `patch_gdn_metadata`, `patch_gdn_shared_build`, `patch_topk_composite`,
`patch_quark_mxfp4`, `patch_quark_bf16_mtp`, `patch_ar_maxbytes`, `patch_ar_geometry`,
`patch_kv_group_size`, `patch_gdn_merge_inproj`, `patch_dynwidth`, `patch_verify_head`,
`patch_fp8_kv_sidecar`, `patch_qwen_open_object_schema`.

Where the newer base had drifted, the overlays were re-anchored value-preservingly:
`patch_r4d` (GDN metadata handle is now dict-resolved), `patch_gdn_metadata` /
`patch_gdn_shared_build` (new upstream prefill/spec bookkeeping), `patch_quark_mxfp4` (aiter module
move + renamed preshuffle symbol), `patch_verify_head` (indent), `patch_dynwidth` (delegated
grammar validation), `patch_gdn_merge_inproj` (both runners gained a warmup context manager).

### Retired — already owned by the newer upstream base

These radiance overlays were backports of upstream fixes that the newer base already contains, so
they were dropped rather than carried as redundant patches (verified by inspection):

- `patch_xgrammar_spec_termination` — `accept_tokens`/`validate_tokens`/`reset` now terminate-aware.
- `patch_xgrammar_spec_reasoning` — the newer base validates post-reasoning spec drafts in
  `structured_output/__init__.py::validate_tokens`.
- `patch_parser_shared_engine` — the shared-engine shortcut no longer exists in `parser_manager.py`.
- `patch_rocm_cudagraph_current_stream` — all three capture sites use `current_stream()`.
- `patch_topk_triton_rows` — the newer base routes to Triton unconditionally; the `<8` sort gate is gone.
- `patch_dflash_logits_cache_stride` — upstream #53017 is present.
- `patch_kv_offload_lifecycle` — upstream #52596 lifecycle present (all NOOP on the newer base).
- `patch_kv_offload_registration` — superseded by upstream's own `pinned_addresses` chunk rollback.
- `patch_kv_offload_rank_sharded` — experimental extension of the retired lifecycle overlay.

### Deferred — needs re-integration against the newer base

- `patch_kv_offload_restore` — the newer base rewrote `_annotate_eagle_groups*` (general hybrid
  annotation with `use_trailing_layer_fallback` / `_groups_partition_layers_exactly`). The radiance
  overlay's Qwen DFlash/DFlash2 + MTP draft-layer detection is not expressed by that general path;
  re-adding it changes which cache groups are volatile, so it must be re-qualified on hardware
  rather than ported mechanically. Until then, native CPU KV hits for hybrid + speculative
  deployments behave as upstream.

The out-of-tree overlays (`patch_unified_attention_lds` → aiter, `patch_from_json_filter` →
transformers, `patch_dynamo_metrics` → torch, `patch_radiance_dispatch` aiter hunk) still run at
bootstrap time, but their anchors were written for the image's pinned torch/transformers/aiter. They
report drift instead of corrupting a file; treat a reported failure as "this optimization is off"
and re-anchor against the installed versions.

## Compatibility note

The published image qualified a compiler-stack unit of ROCm 7.14 / AMD torch 2.12 / AMD Triton 3.7.1
/ AITER 0.1.20 against vLLM v0.28.0. This fork's base instead pins `torch == 2.13.0`, and the host's
proven **ROCm 10 + AMD whl-next** flow (`torch 2.13.0+rocm10.0.0`, precompiled Triton) supplies
exactly that — so build against that stack, not the image's 7.14/2.12 pins. `amd-wheel` mode encodes
it; `RADIANCE_STACK_MODE=skip` is for hosts that already have a matched stack.

Radiance's tuned GEMM/attention kernels target its qualified deployments (FP8 / MXFP4 quantized
Qwen/DeepSeek-style models). The overlays are generic and do not break other quantization paths
(e.g. GPTQ), but their performance tuning may not engage for a checkpoint the image did not qualify.

Baremetal installs were not exercised on the hardware used to port these overlays; the in-tree
source changes are syntax-checked and idempotent, but the native builds, the out-of-tree patches,
and the absence of the retired overlays need a serving qualification on the target GPUs before this
is treated as equivalent to the published image.