#!/usr/bin/env bash
# =============================================================================
# radiance baremetal bootstrap (uv) for the gfx1201 / Radeon AI PRO R9700 fork.
#
# Builds and installs this fork as the radiance vLLM distribution into a uv venv,
# without Docker. The fork source already carries the radiance vLLM patches; this
# script supplies what vllm-radiance supplied at image build time: the ROCm Python
# stack, libr4d, the MXFP4/W4A8 HIP extension, the runtime modules, the tuned
# configs, the out-of-tree patches, and the amdsmi startup hook.
#
# The default stack mode mirrors the proven ROCm 10 + AMD whl-next flow for
# gfx1201 (torch 2.13.0+rocm10.0.0), which matches this fork's `torch == 2.13.0`
# pin. Modes:
#   amd-wheel : system ROCm 10 dev headers + AMD whl-next torch/triton (default)
#   auto      : let uv --torch-backend resolve a ROCm stack
#   skip      : a matched torch/triton stack is already importable
#
# Prerequisites on the host:
#   * ROCm 10 userspace + `amdrocm-core-dev10.0` (hipcc on PATH)
#   * uv (https://docs.astral.sh/uv/)
#   * git, a C++ toolchain, python3.12 available to uv
#
# Usage:
#   ./radiance/bootstrap.sh
#   RADIANCE_STACK_MODE=skip ./radiance/bootstrap.sh   # you already have torch
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES="$ROOT/radiance/patches"
CONFIGS="$ROOT/radiance/configs"

VENV="${RADIANCE_VENV:-$ROOT/.venv}"
GFX_ARCH="${RADIANCE_GFX_ARCH:-gfx1201}"

STACK_MODE="${RADIANCE_STACK_MODE:-amd-wheel}"
TORCH_BACKEND="${RADIANCE_TORCH_BACKEND:-auto}"
TORCH_VERSION="${RADIANCE_TORCH_VERSION:-2.13.0}"
TORCHVISION_VERSION="${RADIANCE_TORCHVISION_VERSION:-0.28.0}"
TORCHAUDIO_VERSION="${RADIANCE_TORCHAUDIO_VERSION:-2.11.0.2}"
ROCM_WHEEL_INDEX="${RADIANCE_ROCM_WHEEL_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}"
INSTALL_MODE="${RADIANCE_INSTALL_MODE:-editable}"   # editable | wheel

# ROCm 10 keeps the SDK under /opt/rocm/core-10.0; fall back to /opt/rocm.
ROCM_ROOT="${RADIANCE_ROCM_ROOT:-}"
if [ -z "$ROCM_ROOT" ]; then
  if [ -d /opt/rocm/core-10.0 ]; then ROCM_ROOT=/opt/rocm/core-10.0; else ROCM_ROOT=/opt/rocm; fi
fi

R4D_REPO="${RADIANCE_R4D_REPO:-https://codeberg.org/StillDeadcode/libr4d.git}"
R4D_VERSION="${RADIANCE_R4D_VERSION:-v0.5.0}"
R4D_COMMIT="${RADIANCE_R4D_COMMIT:-e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7}"
R4D_DIR="${RADIANCE_R4D_DIR:-}"                     # optional existing clone

SKIP_DEPS="${RADIANCE_SKIP_DEPS:-0}"
SKIP_R4D="${RADIANCE_SKIP_R4D:-0}"
SKIP_HIPEXT="${RADIANCE_SKIP_HIPEXT:-0}"
SKIP_PATCHES="${RADIANCE_SKIP_PATCHES:-0}"
SKIP_CONFIGS="${RADIANCE_SKIP_CONFIGS:-0}"
SKIP_LLVM_LINK="${RADIANCE_SKIP_LLVM_LINK:-0}"
SKIP_SMOKE="${RADIANCE_SKIP_SMOKE:-0}"

log()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
log "preflight"
command -v uv  >/dev/null || die "uv not found on PATH"
command -v git >/dev/null || die "git not found on PATH"
[ -d "$ROCM_ROOT" ] || warn "ROCm root $ROCM_ROOT does not exist; native steps will fail"

export ROCM_PATH="$ROCM_ROOT"
export HIP_PATH="$ROCM_ROOT"
export PYTORCH_ROCM_ARCH="$GFX_ARCH"
export RADIANCE_GFX_ARCH="$GFX_ARCH"
export VLLM_TARGET_DEVICE=rocm
export HIP_ARCHITECTURES="$GFX_ARCH" AMDGPU_TARGETS="$GFX_ARCH" GPU_ARCHS="$GFX_ARCH"
export TRITON_USE_ROCM=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# -----------------------------------------------------------------------------
log "uv venv ($VENV)"
uv venv --python "${RADIANCE_PYTHON:-3.12}" "$VENV" 2>/dev/null || true
PY="$VENV/bin/python"
[ -x "$PY" ] || die "venv python missing at $PY"
SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "site-packages: $SP"

# -----------------------------------------------------------------------------
log "ROCm python stack (mode=$STACK_MODE)"
if [ "$STACK_MODE" = "skip" ]; then
  "$PY" -c 'import torch; print("torch", torch.__version__)' || die "torch not importable"
elif [ "$STACK_MODE" = "auto" ]; then
  uv pip install --python "$PY" "torch==${TORCH_VERSION}" --torch-backend="$TORCH_BACKEND" \
    || die "torch install failed; use RADIANCE_STACK_MODE=amd-wheel or skip"
  uv pip install --python "$PY" --torch-backend="$TORCH_BACKEND" triton torchvision 2>/dev/null \
    || warn "triton/torchvision install returned non-zero"
else
  # amd-wheel: the proven gfx1201 / ROCm 10 flow.
  uv pip install --python "$PY" --upgrade pip
  uv pip install --python "$PY" cmake ninja setuptools-rust wheel pybind11
  if [ "$SKIP_DEPS" != "1" ]; then
    [ -d "$ROCM_ROOT/share/amd_smi" ] && uv pip install --python "$PY" "$ROCM_ROOT/share/amd_smi" \
      || warn "amd_smi not found at $ROCM_ROOT/share/amd_smi"
    uv pip install --python "$PY" -r "$ROOT/requirements/rocm.txt"
  fi
  # AMD's prebuilt gfx1201 wheels. The extras pull the matching precompiled Triton,
  # so triton is never built from source here.
  uv pip install --python "$PY" --reinstall \
      --extra-index-url "$ROCM_WHEEL_INDEX" \
      "torch[device-${GFX_ARCH}]==${TORCH_VERSION}+rocm10.0.0"
  uv pip install --python "$PY" --index-url "$ROCM_WHEEL_INDEX" \
      "torch[device-${GFX_ARCH}]==${TORCH_VERSION}+rocm10.0.0" \
      "torchvision[device-${GFX_ARCH}]==${TORCHVISION_VERSION}+rocm10.0.0" \
      "torchaudio==${TORCHAUDIO_VERSION}+rocm10.0.0" || warn "torchvision/torchaudio install returned non-zero"
  # amd-quark currently circular-imports on vLLM main; the radiance Quark paths are
  # default-off and guarded, so removing it is safe for non-Quark checkpoints.
  uv pip uninstall --python "$PY" amd-quark 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
if [ "$SKIP_LLVM_LINK" != "1" ] && [ "$STACK_MODE" = "amd-wheel" ]; then
  log "LLVM library dedup (ROCm 10 SDK vs python _rocm_sdk_* copies)"
  # ROCm 10 ships the same libraries in /opt/rocm and in the python _rocm_sdk_*
  # packages; the duplicate LLVM crashes triton's spirv-expand-step. Point the
  # python copies at the system tree.
  for d in _rocm_sdk_core _rocm_sdk_libraries; do
    if [ -d "$SP/$d" ]; then
      rm -rf "$SP/$d/lib" "$SP/$d/lib64"
      ln -sfn "$ROCM_ROOT/lib" "$SP/$d/lib"
      echo "linked $SP/$d/lib -> $ROCM_ROOT/lib"
    fi
  done
fi

# -----------------------------------------------------------------------------
log "install this fork (mode=$INSTALL_MODE)"
rm -rf "$ROOT/build" "$ROOT/CMakeCache.txt"
if [ "$INSTALL_MODE" = "editable" ]; then
  VLLM_TARGET_DEVICE=rocm uv pip install --python "$PY" --no-build-isolation -e "$ROOT"
else
  VLLM_TARGET_DEVICE=rocm uv pip install --python "$PY" --no-build-isolation "$ROOT"
fi
SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# -----------------------------------------------------------------------------
if [ "$SKIP_R4D" != "1" ]; then
  log "libr4d ($R4D_VERSION @ ${R4D_COMMIT:0:12})"
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  if [ -n "$R4D_DIR" ]; then
    SRC="$R4D_DIR"
  else
    SRC="$WORK/libr4d"
    git clone --filter=blob:none --no-checkout "$R4D_REPO" "$SRC"
    git -C "$SRC" fetch --depth 1 origin tag "$R4D_VERSION" || true
    git -C "$SRC" checkout --detach "$R4D_COMMIT"
    [ "$(git -C "$SRC" rev-parse HEAD)" = "$R4D_COMMIT" ] || die "libr4d checkout mismatch"
  fi
  git -C "$SRC" apply --check "$PATCHES/r4d_radiance_extras.patch" 2>/dev/null \
    && git -C "$SRC" apply "$PATCHES/r4d_radiance_extras.patch" \
    || warn "r4d extras patch already applied or not applicable; continuing"
  ( cd "$SRC" && GFX_ARCH="$GFX_ARCH" OUT="$SP/r4d.so" ./build.sh )
  "$PY" - <<PY
import r4d
print("r4d", r4d.__version__, "kernels", len(r4d.kernels()))
PY
else
  log "libr4d skipped"
fi

# -----------------------------------------------------------------------------
if [ "$SKIP_HIPEXT" != "1" ]; then
  log "radiance MXFP4/W4A8 HIP extension"
  command -v hipcc >/dev/null || die "hipcc not found (needed for radiance_mxfp4_fp8.so)"
  INC="$("$PY" -m pybind11 --includes)"
  hipcc -O3 -std=c++17 -fPIC -shared --offload-arch="$GFX_ARCH" -Wno-unused-result \
      $INC "$ROOT/radiance_mxfp4_fp8.hip" -o "$SP/radiance_mxfp4_fp8.so"
  "$PY" - <<PY
import radiance_mxfp4_fp8 as m
assert all(hasattr(m, n) for n in (
    "launch", "launch_at", "set_decode_scratch",
    "launch_add_rms_quant", "launch_silu_mul_quant", "launch_gdn_norm_quant",
))
print("radiance_mxfp4_fp8 OK")
PY
else
  log "HIP extension skipped"
fi

# -----------------------------------------------------------------------------
if [ "$SKIP_CONFIGS" != "1" ]; then
  log "tuned configs + amdsmi startup hook"
  install -d "$SP/vllm/model_executor/layers/quantization/utils/configs"
  install -d "$SP/vllm/model_executor/layers/fused_moe/configs"
  install -d "$SP/aiter/ops/triton/configs/gemm" 2>/dev/null || true
  cp -f "$CONFIGS"/fp8/*.json  "$SP/vllm/model_executor/layers/quantization/utils/configs/"
  cp -f "$CONFIGS"/moe/*.json  "$SP/vllm/model_executor/layers/fused_moe/configs/"
  cp -f "$CONFIGS"/mxfp4/*.json "$SP/aiter/ops/triton/configs/gemm/" 2>/dev/null \
    || warn "aiter config dir absent; MXFP4 tiles not installed (aiter not in this env?)"
  cp -f "$ROOT/radiance_amdsmi.pth" "$SP/radiance_amdsmi.pth"
  echo "configs installed; amdsmi .pth installed (amdsmi_init before HIP)"
else
  log "configs/hook skipped"
fi

# -----------------------------------------------------------------------------
if [ "$SKIP_PATCHES" != "1" ]; then
  log "out-of-tree patches (torch / transformers / aiter)"
  cd "$PATCHES"
  # Each script reports drift instead of corrupting a file; a reported failure
  # simply means that optimization is off for the installed package version.
  for p in patch_radiance_dispatch patch_unified_attention_lds patch_from_json_filter patch_dynamo_metrics; do
    echo "-- $p"
    "$PY" "$p.py" || warn "$p reported a drift/failure; inspect before relying on it"
  done
else
  log "out-of-tree patches skipped"
fi

# -----------------------------------------------------------------------------
if [ "$SKIP_SMOKE" != "1" ]; then
  log "smoke test"
  "$PY" - <<'PY'
import importlib.metadata as md
import torch, vllm, amdsmi
print("vllm", vllm.__version__, "| torch", torch.__version__,
      "| torchvision", md.version("torchvision"), "| triton", md.version("triton"))
try:
    import r4d
    print("r4d", r4d.__version__)
except Exception as e:
    print("r4d not loaded:", e)
PY
fi

log "done"
echo "Source the qualified defaults before serving:"
echo "  source \"$ROOT/radiance/radiance-env.sh\""
echo "  vllm serve <model> --tensor-parallel-size 2 ..."