# Radiance on baremetal (uv)

> **Full from-zero runbook:** see [`BAREMETAL_SETUP.md`](BAREMETAL_SETUP.md) - prerequisites, clone,
> bootstrap, serve, and the issues hit along the way.

This directory turns the fork into the **vllm-radiance** distribution without Docker, so vLLM can
be built and installed on a bare-metal ROCm host with `uv` managing the Python environment.

## Current base

- vLLM **0.34.0** (`62d4bc3c7e`), the release vllm-radiance `1.2.0-rc1` is built on.
- Radiance overlays, runtime modules, tuned configs and patch archives synced from
  **vllm-radiance `main`** (`dcecd8e`, `1.2.0-rc1.vllm0.34.0`).
- Fork-only work retained: the gfx12 WMMA fused-MoE W4A16 kernel, the RDNA4 scalar-kernel
  improvements, PCIe custom all-reduce, the RDNA hybrid stale-qzeros fix, the Gemma4 vision
  `mm_prefix` gate, the AutoGPTQ negative-dynamic-override fix, and the fork's attention tune.

The fork source already carries the radiance vLLM changes (the `patch_*.py` overlays applied
in-tree). This directory supplies the rest of what the vllm-radiance image supplies at build time:

| Asset | Location | Installed by |
|---|---|---|
| Runtime modules (`radiance_kernels`, `radiance_gdn`, ...) | repo root, `radiance_*.py` | wheel (`py_modules` in `setup.py`) |
| `pth`-based amdsmi startup hook | `radiance_amdsmi.pth`, `radiance_amdsmi.py` | `bootstrap.sh` |
| General-plugin entry point (`radiance = radiance_kernels:install_all`) | `patches/install_radiance_hooks.py` | `bootstrap.sh` |
| Tuned FP8 / MoE / MXFP4 configs | `radiance/configs/` | `bootstrap.sh` |
| `libr4d` gfx1201 kernel library (`r4d.so`) | pinned source + `patches/r4d_radiance_extras.patch` | `bootstrap.sh` |
| MXFP4/W4A8 HIP extension (`radiance_mxfp4_fp8.so`) | `radiance_mxfp4_fp8.hip` | `bootstrap.sh` |
| Out-of-tree patches (torch / transformers / aiter) | `radiance/patches/` | `bootstrap.sh` |
| Qualified runtime env defaults | `radiance/radiance-env.sh` | sourced by the operator |

## Quick start

The default stack mode mirrors the proven **ROCm 10.1 + AMD whl-next** flow for gfx1201
(`torch 2.14.0+rocm10.1.0`), which matches vLLM 0.34's `torch == 2.14.0` pin:

```bash
# Host: ROCm 10.1 + amdrocm-core-dev10.1, hipcc on PATH, uv on PATH.
RADIANCE_INSTALL_AITER=1 ./radiance/bootstrap.sh     # amd-wheel stack (default)

# ...or if you already built the env the manual way:
RADIANCE_STACK_MODE=skip ./radiance/bootstrap.sh

source radiance/radiance-env.sh
vllm serve <model> --tensor-parallel-size 8 ...
```

`bootstrap.sh` in `amd-wheel` mode performs the same Phase 1-4 steps as the manual guide
(sysbuild tools, `amd_smi`, `requirements/rocm.txt`, AMD whl-next torch/torchvision/torchaudio,
`amd-quark` removal, the LLVM library dedup symlinks, `--no-build-isolation` build), then layers on
the radiance extras: libr4d, the MXFP4/W4A8 HIP extension, tuned configs, the amdsmi `.pth`, the
general-plugin entry point, and the out-of-tree patches.

Useful knobs: `RADIANCE_VENV`, `RADIANCE_STACK_MODE=amd-wheel|auto|skip`,
`RADIANCE_ROCM_ROOT` (defaults to the newest `/opt/rocm/core-10.*`, i.e. `/opt/rocm/core-10.1` on
current hosts), `RADIANCE_ROCM_HOME` (root with `lib/cmake/hip-lang`, auto-detected),
`RADIANCE_ROCM_WHEEL_TAG` (default `rocm10.1.0`), `RADIANCE_TORCH_BACKEND`,
`RADIANCE_INSTALL_MODE=editable|wheel`, `RADIANCE_INSTALL_AITER=1` (+ `RADIANCE_AITER_VERSION`,
`RADIANCE_AITER_COMMIT`, `RADIANCE_AITER_SPEC`), `RADIANCE_R4D_DIR`,
`RADIANCE_SKIP_{DEPS,FORK,R4D,HIPEXT,PATCHES,CONFIGS,LLVM_LINK,SMOKE}=1`.

> `amd-wheel` mode uninstalls `amd-quark`, matching the manual guide. The radiance Quark paths
> (`RADIANCE_MXFP4*`, `RADIANCE_QUARK_BF16_MTP`) are default-off and guarded, so non-Quark
> checkpoints are unaffected.
>
> **AITER is not installed by default.** It is not a vLLM dependency (the fork guards its import),
> and the R4D path does not need it. Set `RADIANCE_INSTALL_AITER=1` to build AITER for gfx1201 - it
> provides the `ROCM_AITER_UNIFIED_ATTN` fallback backend, the preshuffle FP8 blockscale GEMM
> (`RADIANCE_PRESHUFFLE`), the GDN AITER knobs, and the targets of `patch_unified_attention_lds`
> plus `patch_radiance_dispatch`'s `SPLITK` hunk. No `amd-aiter` wheel is published (the AMD
> whl-next index and PyPI both lack it), so this is a source build. vllm-radiance 1.2.0-rc1 pins
> **AITER 0.1.26** (`84077b3c`) and **flydsl 0.4.0**, so those are the bootstrap defaults. Override
> with `RADIANCE_AITER_VERSION` / `RADIANCE_AITER_COMMIT` / `RADIANCE_AITER_SPEC`. Changing the
> AITER version changes the AITER-backed tuning paths, so treat it as a re-qualification. When
> AITER is absent, the aiter out-of-tree patches and the MXFP4 tile copy skip with a warning.
>
> The AITER unified-attention compatibility items from the 0.1.23 era still apply on 0.1.26:
> `patch_unified_attention_lds` clamps the staged K/V tile into the R9700's 64 KiB LDS at the
> `>= 0.1.21` `unified_attention_utils` layout, vLLM imports
> `aiter.ops.triton.attention.unified_attention` with a fallback to the old path, and aiter's
> `flydsl` dependency is installed explicitly. `R4D` remains the primary TP2 attention path.

## Patch split

The radiance patch archives are synced wholesale from vllm-radiance, but only the patches that
cannot live in a source tree are applied at bootstrap time. Everything that edits vLLM source is
already inlined in this fork.

### Applied in-tree

`patch_gfx1201`, `patch_radiance_dispatch` (vLLM hunk), `patch_skinny_gemm`, `patch_gdn_wmma`,
`patch_gdn_aiter_prefill`, `patch_preshuffle`, `patch_mtp_loopbreak`, `patch_qwen3_toolparse`,
`patch_conv1d_blockn`, `patch_r4d`, `patch_dflash_fused_kv_fp8`, `patch_dflash_quant_exclusions`,
`patch_dflash_w4`, `patch_dflash_selector_topk`, `patch_gdn_metadata`, `patch_topk_composite`,
`patch_quark_mxfp4`, `patch_nvfp4_mxfp4`, `patch_quark_bf16_mtp`, `patch_kv_group_size`,
`patch_gdn_merge_inproj`, `patch_dynwidth`, `patch_verify_head`, `patch_kv_offload_registration`,
`patch_kv_offload_lifecycle`, `patch_kv_offload_restore`, `patch_fp8_kv_sidecar`,
`patch_qwen_open_object_schema`, `patch_qwen_tool_grammar`.

The `patch_ar_maxbytes` / `patch_ar_geometry` knobs are pre-applied to the vendored
`radiance_allreduce.py`, because an editable install imports that module from the source tree, not
from site-packages.

### Applied at bootstrap (out-of-tree)

`patch_radiance_dispatch` (aiter `SPLITK` hunk), `patch_unified_attention_lds` (aiter),
`patch_from_json_filter` (transformers), `patch_dynamo_metrics` (torch), and
`install_radiance_hooks` (the `vllm.general_plugins` entry point). Each patch reports drift instead
of corrupting a file; treat a reported failure as "this optimization is off" and re-anchor against
the installed versions.

### Not applied on this base

- `patch_gdn_shared_build` - the 0.34 `GDNAttentionMetadataBuilder.build` signature and internals
  changed, and the patch is not part of vllm-radiance's 0.34 image. It needs re-qualification on
  hardware before it can be carried; until then GDN metadata is built per KV-cache group as upstream.
- `patch_mtp_mm_mask` - the V1 multimodal MTP mask overlay was retired for vLLM 0.34.
- `patch_lfm3_backport` - applies to the vendored `upstream/vllm-lfm3` source, not this tree.

## Compatibility note

vLLM 0.34 pins `torch == 2.14.0`; the host's proven **ROCm 10.1 + AMD whl-next** flow
(`torch 2.14.0+rocm10.1.0`, precompiled Triton) supplies exactly that, so build against that stack.
`amd-wheel` mode encodes it; `RADIANCE_STACK_MODE=skip` is for hosts that already have a matched
stack.

Radiance's tuned GEMM/attention kernels target its qualified deployments (FP8 / MXFP4 quantized
Qwen/DeepSeek-style models). The overlays are generic and do not break other quantization paths
(e.g. GPTQ), but their performance tuning may not engage for a checkpoint the image did not qualify.

Baremetal installs were not exercised on the hardware used to port these overlays; the in-tree
source changes are syntax-checked, but the native builds, the out-of-tree patches, and the new
RDNA4 kernels need a serving qualification on the target GPUs before this is treated as equivalent
to the published image.
