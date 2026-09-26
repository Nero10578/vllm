# Radiance baremetal setup — from zero

This is the end-to-end runbook for building and serving **this fork** (the radiance-gfx1201 vLLM)
on a bare-metal ROCm 10 host with `uv` as the Python environment. It records the exact steps that
worked, and every failure we hit along the way with its fix, so you don't rediscover them.

Primary qualified target: AMD Radeon AI PRO R9700 / gfx1201, dual-GPU TP2, FP8 weights + FP8 KV.

If you only want the short version, it is: **clone → `./radiance/bootstrap.sh` → source env → serve**.
The rest of this file explains why, and what to do when a step errors.

---

## 0. What "the fork" contains

- `vllm/**` — the vLLM source with the radiance overlays baked in (30 of the 42 image overlays).
- `radiance_*.py` (repo root) — the runtime kernel/hook modules, shipped as top-level modules.
- `radiance/` — everything that isn't vLLM source:
  - `patches/` — the overlay scripts, including the out-of-tree ones (aiter/torch/transformers).
  - `configs/{fp8,moe,mxfp4}/` — tuned kernel configs.
  - `radiance-env.sh` — qualified `RADIANCE_*` runtime defaults + ROCm paths.
  - `bootstrap.sh` — the whole install flow.
  - `README.md` — port status (applied / retired / deferred / AITER gap).
- `radiance_mxfp4_fp8.hip` — the MXFP4/W4A8 HIP extension source.
- `radiance_amdsmi.pth` — interpreter-startup hook that initializes amdsmi before HIP.

The vLLM-side overlays are **already committed**; the `radiance/patches/` scripts are applied by
`bootstrap.sh` to the installed third-party packages (aiter, torch, transformers), which is why they
can't live in the vLLM source tree.

---

## 1. Host prerequisites (once)

```bash
# ROCm 10 userspace + developer headers (hipcc), plus the build toolchain
sudo apt update
sudo apt install -y amdrocm-core-dev10.0 build-essential python3.12-dev g++ git

which hipcc && hipcc --version          # must be on PATH
rocm-smi                                # confirm the R9700s are visible
find /opt/rocm -name 'libamdhip64*'     # note where the HIP runtime lives

# uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`python3.12-dev` and `g++` matter: aiter JIT-compiles kernels at first use, and the HIP extension
build needs the Python headers.

---

## 2. Clone the fork

```bash
cd ~
git clone -b radiance-gfx1201-baremetal https://github.com/Nero10578/vllm.git ~/vllm-radiance
cd ~/vllm-radiance
git log --oneline -1        # note the commit; verify it after any pull
```

---

## 3. Bootstrap

### Default (AMD whl-next ROCm 10 stack + wheel install)

```bash
./radiance/bootstrap.sh
```

What it does, in order:

1. Creates `.venv` (Python 3.12).
2. Installs **amdsmi** from a writable copy (see issue #2).
3. Installs the ROCm stack: `requirements/rocm.txt`, then AMD's prebuilt
   `torch 2.13.0+rocm10.0.0` / `torchvision 0.28.0+rocm10.0.0` / `torchaudio 2.11.0.2+rocm10.0.0`
   from `https://stable.repo.amd.com/rocm/whl-next/`, then removes `amd-quark`.
4. LLVM library dedup: symlinks `_rocm_sdk_core/lib` and `_rocm_sdk_libraries/lib` to
   `/opt/rocm/core-10.0/lib` (see issue #1).
5. Installs **this fork as a wheel** (non-editable) — see issue #4 — then removes `amd-quark` again
   (the install re-adds it).
6. Builds **libr4d** → `site-packages/r4d.so`.
7. Compiles **radiance_mxfp4_fp8.hip** → `site-packages/radiance_mxfp4_fp8.so`.
8. Copies tuned configs and the amdsmi `.pth` into the venv.
9. Applies the out-of-tree patches (aiter/torch/transformers).
10. Runs a smoke test.

Expected tail:

```
r4d 0.5.0 kernels 20
radiance_mxfp4_fp8 OK
configs installed; amdsmi .pth installed (amdsmi_init before HIP)
vllm … | torch 2.13.0+rocm10.0.0 | torchvision 0.28.0+rocm10.0.0 | triton 3.8.0+…rocm10.0.0
r4d 0.5.0 kernels 20
radiance runtime modules OK
```

### Stack modes

- `RADIANCE_STACK_MODE=amd-wheel` (default) — the ROCm 10 / AMD whl-next flow above.
- `RADIANCE_STACK_MODE=skip` — you already have a working torch stack; verifies it and skips all
  stack (re)installs. **Use this whenever the venv already went through the LLVM dedup** (issue #1).
- `RADIANCE_STACK_MODE=auto` — `uv --torch-backend` resolves the stack (not recommended on ROCm 10).

### Useful overrides

```
RADIANCE_VENV=/path/.venv
RADIANCE_ROCM_ROOT=/opt/rocm/core-10.0
RADIANCE_INSTALL_MODE=wheel|editable     # default wheel
RADIANCE_INSTALL_AITER=1                 # opt-in; see section 6
RADIANCE_AITER_VERSION / RADIANCE_AITER_COMMIT / RADIANCE_AITER_SPEC
RADIANCE_R4D_DIR=/path/to/libr4d
RADIANCE_SKIP_{DEPS,FORK,R4D,HIPEXT,PATCHES,CONFIGS,LLVM_LINK,SMOKE}=1
```

### Resuming / partial runs

Everything is idempotent. To run only part of the flow, combine skip flags. Examples:

```bash
# Resume native builds + configs + patches without rebuilding vLLM:
RADIANCE_STACK_MODE=skip RADIANCE_SKIP_FORK=1 ./radiance/bootstrap.sh

# Only install aiter, then re-apply configs/patches:
RADIANCE_STACK_MODE=skip RADIANCE_SKIP_FORK=1 RADIANCE_SKIP_R4D=1 \
RADIANCE_SKIP_HIPEXT=1 RADIANCE_SKIP_SMOKE=1 RADIANCE_INSTALL_AITER=1 ./radiance/bootstrap.sh
```

> **Always `git pull --ff-only` before re-running bootstrap after we push fixes, then confirm the
> pull landed** — e.g. `grep -c RADIANCE_SKIP_FORK radiance/bootstrap.sh` must be ≥ 1. Skipping this
> is how we once burned a 9-minute rebuild on an old checkout.

---

## 4. Verify

```bash
cd ~/vllm-radiance
source .venv/bin/activate
python -c "import torch, vllm, r4d, radiance_mxfp4_fp8, radiance_kernels; \
           import importlib.metadata as m; print(m.version('vllm'), torch.__version__, r4d.__version__)"
python -c "import amdsmi; amdsmi.amdsmi_init(); print(len(amdsmi.amdsmi_get_processor_handles()), 'GPUs')"
```

---

## 5. Serve

Create a launcher (radiance profile: FP8, FP8 KV, TP2, R4D attention):

```bash
cat > ~/vllm-radiance/start_vllm.sh <<'EOF'
#!/bin/bash
set -euo pipefail
cd ~/vllm-radiance
source .venv/bin/activate
source radiance/radiance-env.sh          # ROCm10 paths + qualified RADIANCE_* defaults

# host-specific extras (keep what you've proven)
export HIP_VISIBLE_DEVICES=0,1
export HIP_FORCE_DEV_KERNARG=1
export NCCL_MIN_NCHANNELS=112
export GPU_MAX_HW_QUEUES=1
export TRITON_CACHE_DIR="$HOME/vllm-nero/triton-cache"

vllm serve /home/arli/models/Qwen3.5-27B-FP8 --port 8000 \
  -tp 2 \
  --max-model-len 32768 --max-num-seqs 8 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --mamba-cache-mode=align \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8 \
  --attention-backend=R4D \
  --scheduling-policy priority \
  --served-model-name Qwen3.5-27B
EOF
chmod +x ~/vllm-radiance/start_vllm.sh
~/vllm-radiance/start_vllm.sh
```

Notes:

- **`--attention-backend=R4D` is what engages the libr4d path.** Without it you get stock ROCm
  attention and the radiance kernels stay idle.
- `radiance-env.sh` sets the qualified `RADIANCE_*` defaults and the ROCm 10 loader paths. Source it;
  don't hand-roll them.
- The qualified R4D envelope is TP2, FP8 KV, prefix caching + `--mamba-cache-mode=align`, 16K, 85%,
  8 seqs.
- **TP8 requires the AITER attention backend, not R4D.** R4D's kernel is compiled for the TP2 head
  geometry (`gqa == 6`); at TP8 the per-rank ratio is `gqa=3` and R4D refuses with
  `NotImplementedError … needs == 6`. For 8 GPUs either use 4× TP2 data-parallel replicas
  (`-tp 2 --data-parallel-size 4`, keeps R4D), or switch to AITER unified attention:

  ```bash
  export VLLM_ROCM_USE_AITER=1
  export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
  export VLLM_ROCM_USE_AITER_MHA=0
  export VLLM_ROCM_USE_AITER_MLA=0
  export VLLM_ROCM_USE_AITER_MOE=0
  ...
  vllm serve … -tp 8 … --attention-backend=ROCM_AITER_UNIFIED_ATTN …
  ```

  R4D's exact TP2 all-reduce falls back to RCCL at TP8; the rest of the radiance stack still runs.
  See section 6 for the aiter-status details.

Test:

```bash
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-27B","messages":[{"role":"user","content":"Say hi in five words."}],"max_tokens":32}'
```

### Healthy-startup evidence (grep the log)

- `R4D kernel selection: libr4d 0.5.0, 20 kernels built, 15 of 15 queries resolved`
- `Using R4D backend (selected via --attention-backend)`
- `attn_prefill_h256_gqa6_fp8kv` / `attn_decode_h256_gqa6_fp8kv`
- `[radiance] custom all-reduce INSTALLED` and `AR_QUANT ON`
- `[radiance.gdn] gdn_chunk_scan ENABLED` and `all-R4D prefill path live`
- `[radiance.gemm] skinny GEMM kernel ENABLED`
- `[radiance.verifyhead] …`, `[radiance.draft] RADIANCE_DYNAMIC_DRAFT=ON`
- `Application startup complete.`

### Known non-fatal warnings

| Warning | Meaning |
|---|---|
| `Checkpoint does not provide a q scaling factor… uncalibrated q_scale 1.0` | FP8-KV scales missing from the checkpoint; possible accuracy drift. Compare `--kv-cache-dtype bf16` if quality looks off. |
| `Failed to import the DeepSelect extension (vllm._deepselect_C)` | Optional DeepSeek sparse-routing kernel; irrelevant for Qwen. |
| `Falling back to the Triton GDN decode path: … fused_gdn_decode_post_conv_mtp is not built` | The fused GDN decode kernel isn't in this wheel; Triton path is used. Correct but a decode optimization is inactive. |
| `[radiance.gdnmerge] merged 0 GDN layers … 48 left unmerged` | In-proj merge found no eligible layers on this model; inert. |
| `Auto-disabled DeepGemm … on Blackwell` | Cosmetic platform text; GEMM falls back correctly. |
| `use_fast` / `tl.make_block_ptr` deprecations | Cosmetic. |

---

## 6. AITER — and the one thing still not fully solved

AITER is **not** installed by default and is **not** a vLLM dependency (its import is guarded).
The R4D path does not need it. Install it for the AITER fallback backend, the preshuffle FP8
blockscale GEMM (`RADIANCE_PRESHUFFLE`), and the GDN AITER knobs:

```bash
cd ~/vllm-radiance
RADIANCE_STACK_MODE=skip RADIANCE_SKIP_FORK=1 RADIANCE_SKIP_R4D=1 \
RADIANCE_SKIP_HIPEXT=1 RADIANCE_SKIP_SMOKE=1 RADIANCE_INSTALL_AITER=1 ./radiance/bootstrap.sh
```

Facts:

- **No `amd-aiter` wheel is published** (neither on the AMD whl-next index nor PyPI), so bootstrap
  builds it from source: `ROCm/aiter`, default tag **v0.1.23** (`50da036a`), with
  `GPU_ARCHS=gfx1201 PREBUILD_KERNELS=0 AITER_USE_SYSTEM_TRITON=1`. Override with
  `RADIANCE_AITER_VERSION` / `RADIANCE_AITER_COMMIT` / `RADIANCE_AITER_SPEC`.
- It JIT-builds `module_aiter_core` on first import (~25 s), visible as `[aiter] start build
  [module_aiter_core]`.
- The qualified image used AITER **0.1.20**; we default to 0.1.23 because 0.1.20 predates
  torch 2.13 / ROCm 10.

### The RDNA LDS-fit overlay (re-ported for aiter >= 0.1.21)

The image carried `patch_unified_attention_lds.py`, a **correctness** fix that shrinks AITER's
staged K/V tile to fit the R9700's 64 KiB LDS (AITER sizes it for CDNA's much larger LDS). Without
it, AITER's unified-attention backend can raise Triton `OutOfResources` at CUDA-graph capture:

```
head_size 256, 2-byte KV (bf16/fp16): 64*256*2*2 + 256 = 65792
head_size 512, fp8 KV              : 64*512*1*2 + 256 = 65792
```

aiter **0.1.21+ rewrote `unified_attention.py`**: `select_3d_config` / `select_2d_config` were
removed and config selection moved to `unified_attention_utils.get_unified_attention_config`
(`compute_tile_params` / `compute_segment_params`), so the original overlay's anchors no longer
exist. `patch_unified_attention_lds.py` now handles **both** layouts:

- **aiter <= 0.1.20** — edits `select_3d_config` / `select_2d_config` as before.
- **aiter >= 0.1.21** — injects a `_radiance_fit_lds` helper into `unified_attention.py` and clamps
  the staged tile (pipeline depth first, then tile width) in `_unified_attention_2d_triton` and
  `_unified_attention_3d_triton`, where both the tile and the stage count are in scope. The clamp is
  guarded to `gfx12` and skipped for shuffled/registered caches (their tile must equal the page).

**Status:** the >= 0.1.21 path was validated against the real v0.1.23 source (anchors match, AST
parses, idempotent) **and exercised on hardware**: AITER unified attention serves at TP8 on
Qwen3.5-27B-FP8 with no `OutOfResources`. Two other aiter-0.1.23 compatibility items are required
(and are handled in the branch/bootstrap):

- **vLLM import path.** aiter >= 0.1.21 moved the module to
  `aiter.ops.triton.attention.unified_attention`; vLLM's `rocm_aiter_unified_attn.py` imported only
  the old `aiter.ops.triton.unified_attention`, giving `ModuleNotFoundError`. vLLM now tries the new
  path first and falls back to the old one.
- **`flydsl` dependency.** aiter is installed with `--no-deps` (so it cannot replace torch/triton),
  but aiter >= 0.1.21's `__init__.py` imports `topk_select` → `flydsl`. bootstrap now installs
  `flydsl==0.3.4.1` (override with `RADIANCE_FLYDSL_SPEC`) plus aiter's other runtime deps.

If you ever see a Triton `OutOfResources` at capture again (e.g. a different block size / head),
the knob to tune is the clamp budget (currently 64 KiB) or the gfx1201 `TILE_SIZE_MIN/MAX` in the
aiter config.

**Options:**

1. **Keep using R4D** (default tuned path, TP2 geometry only) — the LDS overlay is dormant there.
2. **Use AITER unified attention** — enable `VLLM_ROCM_USE_AITER*` (below) and
   `--attention-backend=ROCM_AITER_UNIFIED_ATTN`. This is the path for geometries R4D can't serve,
   notably **TP8** (`gqa=3`) and the head-512 Gemma drafter. Validated at TP8.
3. **Pin aiter 0.1.20** (the qualified image version) so the old-layout overlay applies as written —
   but then you also lose the 0.1.23 support; risk: may not build against torch 2.13 / ROCm 10.

The aiter change that matters for GEMM applied regardless: `patch_radiance_dispatch`'s `SPLITK`
alignment fix (visible as `# --- radiance fix (patch_radiance_dispatch.py): scale-alignment guard ---`
in `aiter/ops/triton/utils/gemm_config_utils.py`).

Confirm which state you're in on any install:

```bash
cd ~/vllm-radiance/.venv/lib/python3.12/site-packages
grep -n "radiance" aiter/ops/triton/utils/gemm_config_utils.py            # SPLITK fix → expected
grep -n "_radiance_fit_lds" aiter/ops/triton/attention/unified_attention.py  # LDS fit → expected (>=0.1.21)
```

To use AITER attention, uncomment the `VLLM_ROCM_USE_AITER*` exports in `radiance/radiance-env.sh`
and pass `--attention-backend=ROCM_AITER_UNIFIED_ATTN`.

---

## 7. Every issue we hit, and its fix

These are all handled automatically by the current `bootstrap.sh`; this table is so you recognise
them if you see them, or if you're on an older checkout.

| # | Symptom | Cause | Fix (now automatic) |
|---|---|---|---|
| 1 | `error: failed to remove file …/_rocm_sdk_libraries/lib/…: Permission denied (os error 13)` | Ran a stack (re)install into a venv already LLVM-deduped; uv tried to remove root-owned `/opt/rocm` files | Bootstrap detects the dedup symlinks and skips the stack (re)install; or use `RADIANCE_STACK_MODE=skip` |
| 2 | `could not create 'amdsmi.egg-info': Permission denied` | `amd_smi` builds in place inside root-owned `/opt/rocm/share/amd_smi` | Copy to a writable temp dir and install from there |
| 3 | `amd-quark==0.12.post1` reappears after the vLLM build | `uv pip install` re-resolves `requirements/rocm.txt`, re-adding it | Uninstall `amd-quark` **after** the fork install |
| 4 | `ModuleNotFoundError: No module named 'vllm.model_executor.layers.quantization.utils.quant_utils'` under `vllm serve` | Editable install + config copy into `site-packages/vllm` confused the setuptools editable finder | Default install is now a **wheel** (matches the image); uninstalls any prior editable vLLM first |
| 5 | `/usr/bin/python3: No module named pybind11` during libr4d | `build.sh` used system python, not the venv | Run `build.sh` with the venv on `PATH` |
| 6 | `ImportError: libamdhip64.so.7: cannot open shared object file` after building r4d | ROCm lib dirs not on the loader path during verify | Export `/opt/rocm/lib` and the ROCm root lib in `LD_LIBRARY_PATH` |
| 7 | `RADIANCE_SKIP_FORK=1` ignored; vLLM rebuilt anyway | Ran an old checkout | `git pull --ff-only`; verify `grep -c RADIANCE_SKIP_FORK radiance/bootstrap.sh` ≥ 1 |
| 8 | Smoke test `module 'vllm' has no attribute '__version__'` | Script bug (fork may not expose it) | Smoke now uses `importlib.metadata.version("vllm")` |
| 9 | `patch_radiance_dispatch` reported the vLLM hunk missing | Editable install has no `site-packages/vllm`; the hunk is already baked in source | Treated as NOOP for editable installs |
| 10 | `patch_unified_attention_lds` reported `FAIL … missing` or drift | aiter absent, or aiter ≥0.1.21 rewrote the file | Skips cleanly, or applies the new-layout clamp; see section 6 |
| 11 | `ModuleNotFoundError: No module named 'aiter.ops.triton.unified_attention'` | aiter ≥0.1.21 moved the module under `attention/`; vLLM's backend imported only the old path | vLLM now tries `aiter.ops.triton.attention.unified_attention` first, falls back to the old path |
| 12 | `ModuleNotFoundError: No module named 'flydsl'` (from `aiter/ops/topk_select.py`) | aiter installed `--no-deps`, so aiter 0.1.23's own dependency `flydsl` was missing | bootstrap now installs aiter's runtime deps (`flydsl`, pandas, psutil, matplotlib, pyyaml, einops, pybind11, ninja) |

### Why the wheel install, not editable

The Docker image installs a vLLM **wheel**, then copies configs and applies patches to a real
`site-packages/vllm`. Editable installs leave the package in the source tree, so the config copy
and the editable finder disagree (issue #4). We mirror the image. The trade-off: source edits don't
take effect until you re-run bootstrap (it rebuilds). `RADIANCE_INSTALL_MODE=editable` still exists
if you need live source edits, but expect the import caveat.

---

## 8. Updating

```bash
cd ~/vllm-radiance
git pull --ff-only
# confirm the pull, then rebuild what changed:
RADIANCE_STACK_MODE=skip ./radiance/bootstrap.sh        # rebuilds vLLM + natives
```

Since the install is a wheel, `git pull` alone does **not** change the running vLLM — re-run
bootstrap. If you only changed `radiance/*.py` (no `vllm/**`), you can copy the specific module into
site-packages instead of a full rebuild.

---

## 9. Qualification status

| Area | State |
|---|---|
| vLLM source overlays (30) | Baked in, syntax-checked, idempotent |
| Out-of-tree torch / transformers patches | Applied (`from_json` filter, dynamo metrics) |
| libr4d (r4d.so, 20 kernels) | Built; all 15 runtime lookups resolved on Qwen3.5-27B-FP8 |
| MXFP4/W4A8 HIP extension | Built |
| Tuned FP8/MoE/MXFP4 configs | Installed and selected at runtime |
| R4D attention (fp8 KV), GDN, TP2 AR, verify head, dynamic draft | Live and serving (TP2) |
| `patch_kv_offload_restore` | **Deferred** — upstream rewrote hybrid cache annotation; needs re-qualification |
| AITER on 0.1.23 | **Qualified on hardware** — SPLITK fix + re-ported LDS clamp applied; vLLM import-path shim (`attention.unified_attention`) and aiter's `flydsl` dep installed; AITER unified attention serving at **TP8** (section 6) |
| Baremetal serving | Validated on R9700: Qwen3.5-27B-FP8 at TP2 (R4D) and TP8 (AITER unified attn) |

This fork is a forward-port of the radiance overlays onto a newer vLLM base; it is not byte-identical
to the published image. Treat per-path parity as something to confirm with matched benchmarks.