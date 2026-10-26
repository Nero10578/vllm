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
# The default stack mode mirrors the proven ROCm 10.1 + AMD whl-next flow for
# gfx1201 (torch 2.14.0+rocm10.1.0), which matches this fork's `torch == 2.14.0`
# pin. Modes:
#   amd-wheel : system ROCm 10.1 dev headers + AMD whl-next torch/triton (default)
#   auto      : let uv --torch-backend resolve a ROCm stack
#   skip      : a matched torch/triton stack is already importable
#
# Prerequisites on the host:
#   * ROCm 10.1 userspace + `amdrocm-core-dev10.1` (hipcc on PATH, and a
#     hip-lang-config.cmake the vLLM CMake build can find)
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
TORCH_VERSION="${RADIANCE_TORCH_VERSION:-2.14.0}"
TORCHVISION_VERSION="${RADIANCE_TORCHVISION_VERSION:-0.29.0a0}"
TORCHAUDIO_VERSION="${RADIANCE_TORCHAUDIO_VERSION:-2.11.0.3}"
ROCM_WHEEL_INDEX="${RADIANCE_ROCM_WHEEL_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}"
ROCM_WHEEL_TAG="${RADIANCE_ROCM_WHEEL_TAG:-rocm10.1.0}"
# editable (default): edits under vllm/** take effect with no rebuild. The config copy
# below is redirected to the live package (VLLM_PKG) so it never materializes a
# site-packages/vllm that would shadow the editable package. Set
# RADIANCE_INSTALL_MODE=wheel to instead build a real site-packages/vllm (image-style).
INSTALL_MODE="${RADIANCE_INSTALL_MODE:-editable}"   # editable | wheel

# ROCm 10 keeps the SDK under /opt/rocm/core-10.<minor> (core-10.1 on current
# hosts); prefer the newest such tree, else fall back to /opt/rocm.
ROCM_ROOT="${RADIANCE_ROCM_ROOT:-}"
if [ -z "$ROCM_ROOT" ]; then
  ROCM_ROOT=/opt/rocm
  for cand in $(ls -d /opt/rocm/core-10.* 2>/dev/null | sort -V); do
    [ -d "$cand" ] && ROCM_ROOT="$cand"
  done
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

# CMake's HIP language support requires hip-lang-config.cmake, which ROCm 10 can keep
# outside the versioned SDK tree (the cmake package often lives in the unversioned
# /opt/rocm while the libs live in core-10.<minor>). ROCM_PATH/HIP_PATH must name the
# root that actually has it; ROCM_ROOT keeps naming the tree used for libraries
# (LLVM dedup, loader path).
ROCM_HOME="${RADIANCE_ROCM_HOME:-}"
if [ -z "$ROCM_HOME" ]; then
  for root in "$ROCM_ROOT" /opt/rocm; do
    for d in lib lib64 lib/x86_64-unknown-linux-gnu; do
      if [ -f "$root/$d/cmake/hip-lang/hip-lang-config.cmake" ]; then
        ROCM_HOME="$root"
        break 2
      fi
    done
  done
fi
if [ -z "$ROCM_HOME" ]; then
  if [ "${RADIANCE_SKIP_FORK:-0}" = "1" ]; then
    warn "no hip-lang-config.cmake under $ROCM_ROOT or /opt/rocm; the fork build is skipped anyway"
    ROCM_HOME="$ROCM_ROOT"
  else
    die "no hip-lang-config.cmake under $ROCM_ROOT or /opt/rocm; install the ROCm dev package (e.g. amdrocm-core-dev10.1) or set RADIANCE_ROCM_HOME"
  fi
fi
echo "ROCm libs: $ROCM_ROOT | ROCm cmake/HIP: $ROCM_HOME"

export ROCM_PATH="$ROCM_HOME"
export HIP_PATH="$ROCM_HOME"
export PYTORCH_ROCM_ARCH="$GFX_ARCH"
export RADIANCE_GFX_ARCH="$GFX_ARCH"
export VLLM_TARGET_DEVICE=rocm
export HIP_ARCHITECTURES="$GFX_ARCH" AMDGPU_TARGETS="$GFX_ARCH" GPU_ARCHS="$GFX_ARCH"
export TRITON_USE_ROCM=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
# Native artifacts (r4d.so, radiance_mxfp4_fp8.so) link libamdhip64.so.7; without both
# ROCm lib dirs on the loader path, importing them fails during the build/verify steps.
ROCM_LIBS="/opt/rocm/lib:$ROCM_ROOT/lib"
[ "$ROCM_HOME" = "$ROCM_ROOT" ] || ROCM_LIBS="$ROCM_LIBS:$ROCM_HOME/lib"
export LD_LIBRARY_PATH="$ROCM_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -----------------------------------------------------------------------------
log "uv venv ($VENV)"
uv venv --python "${RADIANCE_PYTHON:-3.12}" "$VENV" 2>/dev/null || true
PY="$VENV/bin/python"
[ -x "$PY" ] || die "venv python missing at $PY"
SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "site-packages: $SP"

# The vLLM build needs the CPython development headers: CMake's FindPython asks for
# Development.Module/Development.SABIModule, and without Python.h the configure step
# fails after the whole stack download. uv-managed pythons ship headers; distro
# pythons need the -dev package.
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if ! "$PY" -c 'import os, sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)'; then
  die "python${PYVER} development headers missing (Python.h not found); install them (e.g. sudo apt install python${PYVER}-dev) or point RADIANCE_PYTHON at a python that ships headers"
fi

# A venv that already went through the ROCm LLVM dedup has _rocm_sdk_*/lib symlinked
# into the (root-owned) system ROCm tree. uv can no longer uninstall/reinstall the
# SDK packages that own those paths, so never run a stack (re)install on such a venv.
# Remember the target: after a ROCm upgrade the links point at the old tree and the
# venv must be recreated instead of silently reusing the old stack.
DEDUPED=0
DEDUP_TARGET=""
if [ -L "$SP/_rocm_sdk_libraries/lib" ] || [ -L "$SP/_rocm_sdk_core/lib" ]; then
  DEDUPED=1
  DEDUP_TARGET="$(readlink "$SP/_rocm_sdk_core/lib" 2>/dev/null \
    || readlink "$SP/_rocm_sdk_libraries/lib" 2>/dev/null || true)"
fi

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# amdsmi backs the radiance .pth startup hook (amdsmi_init before HIP). It is not a
# vLLM dependency. Install from a WRITABLE copy: the ROCm share dir is root-owned and
# setuptools writes egg-info in place, which fails with "Permission denied".
if ! "$PY" -c 'import amdsmi' >/dev/null 2>&1; then
  log "amdsmi"
  AMD_SMI_SRC=""
  # The module is `amdsmi`; the source/package directory is `amd_smi` in some ROCm
  # layouts and `amdsmi` in others. Accept a source tree (setup.py/pyproject.toml) or
  # a directory that only holds the package (share/amd_smi, or a lib/python*/site-packages).
  for cand in "$ROCM_ROOT/share/amd_smi" "$ROCM_HOME/share/amd_smi" /opt/rocm/share/amd_smi \
              "$ROCM_ROOT"/lib/python*/site-packages; do
    [ -d "$cand" ] || continue
    if [ -f "$cand/setup.py" ] || [ -f "$cand/pyproject.toml" ] \
       || [ -d "$cand/amd_smi" ] || [ -d "$cand/amdsmi" ]; then
      AMD_SMI_SRC="$cand"
      break
    fi
  done
  if [ -n "$AMD_SMI_SRC" ]; then
    echo "amdsmi source: $AMD_SMI_SRC"
    TMP_SMI="$(mktemp -d)"
    if [ -f "$AMD_SMI_SRC/setup.py" ] || [ -f "$AMD_SMI_SRC/pyproject.toml" ]; then
      cp -r "$AMD_SMI_SRC/." "$TMP_SMI/"
      uv pip install --python "$PY" "$TMP_SMI" || warn "amdsmi install failed"
    else
      # Package-only layout: copy just the package and synthesize the missing build
      # metadata so uv records a dist-info instead of an untracked directory.
      for pkg in amd_smi amdsmi; do
        [ -d "$AMD_SMI_SRC/$pkg" ] && cp -r "$AMD_SMI_SRC/$pkg" "$TMP_SMI/"
      done
      if [ -n "$(ls -A "$TMP_SMI")" ]; then
        cat > "$TMP_SMI/setup.py" <<'PY'
from setuptools import find_packages, setup

setup(name="amdsmi", version="0.0.0", packages=find_packages())
PY
        uv pip install --python "$PY" "$TMP_SMI" || warn "amdsmi install failed"
      else
        warn "no amdsmi python package in $AMD_SMI_SRC (contents: $(ls -A "$AMD_SMI_SRC" | tr '\n' ' '))"
      fi
    fi
    rm -rf "$TMP_SMI"
    "$PY" -c 'import amdsmi; print("amdsmi OK")' || warn "amdsmi import failed"
  else
    warn "no amd_smi tree under $ROCM_ROOT/share, $ROCM_HOME/share or /opt/rocm/share"
    warn "check: ls -A $ROCM_ROOT/share/amd_smi; ls -d $ROCM_ROOT/lib/python*/site-packages/amdsmi"
  fi
else
  log "amdsmi already importable"
fi

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
  # amd-wheel: the proven gfx1201 / ROCm 10.1 flow.
  if [ "$DEDUPED" = "1" ]; then
    if [ -n "$DEDUP_TARGET" ] && [ "$DEDUP_TARGET" != "$ROCM_ROOT/lib" ]; then
      die "venv is deduped to $DEDUP_TARGET but the ROCm root is $ROCM_ROOT; delete $VENV and rerun"
    fi
    warn "venv already has the ROCm LLVM dedup symlinks; skipping the torch stack"
    warn "(re)install so uv does not try to remove root-owned /opt/rocm files."
    warn "Use RADIANCE_STACK_MODE=skip to silence this."
    "$PY" -c 'import torch; print("torch", torch.__version__)' || die "torch not importable"
  else
    uv pip install --python "$PY" --upgrade pip
    uv pip install --python "$PY" cmake ninja setuptools-rust wheel pybind11
    if [ "$SKIP_DEPS" != "1" ]; then
      uv pip install --python "$PY" -r "$ROOT/requirements/rocm.txt"
    fi
    # AMD's prebuilt gfx1201 wheels. The extras pull the matching precompiled Triton,
    # so triton is never built from source here.
    uv pip install --python "$PY" --reinstall \
        --extra-index-url "$ROCM_WHEEL_INDEX" \
        "torch[device-${GFX_ARCH}]==${TORCH_VERSION}+${ROCM_WHEEL_TAG}"
    uv pip install --python "$PY" --index-url "$ROCM_WHEEL_INDEX" \
        "torch[device-${GFX_ARCH}]==${TORCH_VERSION}+${ROCM_WHEEL_TAG}" \
        "torchvision[device-${GFX_ARCH}]==${TORCHVISION_VERSION}+${ROCM_WHEEL_TAG}" \
        "torchaudio==${TORCHAUDIO_VERSION}+${ROCM_WHEEL_TAG}" || warn "torchvision/torchaudio install returned non-zero"
    # amd-quark currently circular-imports on vLLM main; the radiance Quark paths are
    # default-off and guarded, so removing it is safe for non-Quark checkpoints.
    uv pip uninstall --python "$PY" amd-quark 2>/dev/null || true
  fi
fi

# -----------------------------------------------------------------------------
if [ "$SKIP_LLVM_LINK" != "1" ] && [ "$STACK_MODE" = "amd-wheel" ]; then
  log "LLVM library dedup (ROCm 10.1 SDK vs python _rocm_sdk_* copies)"
  # ROCm 10.1 ships the same libraries in /opt/rocm and in the python _rocm_sdk_*
  # packages; the duplicate LLVM crashes triton's spirv-expand-step. Point the
  # python copies at the system tree. Idempotent: an already-correct link is left
  # untouched. Run this only once per venv and never (re)install the stack after.
  for d in _rocm_sdk_core _rocm_sdk_libraries; do
    if [ -d "$SP/$d" ] || [ -L "$SP/$d/lib" ]; then
      if [ "$(readlink "$SP/$d/lib" 2>/dev/null)" = "$ROCM_ROOT/lib" ]; then
        echo "$d/lib already linked -> $ROCM_ROOT/lib"
        continue
      fi
      rm -rf "$SP/$d/lib" "$SP/$d/lib64"
      ln -sfn "$ROCM_ROOT/lib" "$SP/$d/lib"
      echo "linked $SP/$d/lib -> $ROCM_ROOT/lib"
    fi
  done
fi

# -----------------------------------------------------------------------------
# AITER is NOT a vLLM requirement (vllm/_aiter_ops.py guards the import), so the
# fork builds and serves without it on the R4D path. But the qualified image builds
# AITER for gfx1201: it provides the ROCM_AITER_UNIFIED_ATTN fallback backend, the
# preshuffle FP8 blockscale GEMM (RADIANCE_PRESHUFFLE), and the GDN AITER knobs, and
# it is the target of patch_unified_attention_lds + patch_radiance_dispatch's SPLITK
# hunk. Opt in with RADIANCE_INSTALL_AITER=1.
if [ "${RADIANCE_INSTALL_AITER:-0}" = "1" ]; then
  log "AITER (gfx1201, source build; kernels JIT at runtime)"
  # The qualified image pinned AITER 0.1.20 (fc2e5d57) for torch 2.12 / ROCm 7.14. This
  # host is torch 2.14 / ROCm 10.1, so default to the current ROCm/aiter tag and let the
  # operator override. Changing the AITER version changes the AITER-backed tuning paths
  # (preshuffle FP8 GEMM, unified attention, GDN knobs), so treat it as a re-qualification.
  AITER_VERSION="${RADIANCE_AITER_VERSION:-0.1.23}"
  AITER_COMMIT="${RADIANCE_AITER_COMMIT:-50da036acdedec2dd596f93188d6c615e2561672}"
  AITER_SPEC="${RADIANCE_AITER_SPEC:-}"
  # No amd-aiter wheel is published on the AMD whl-next index or PyPI (verified against
  # https://stable.repo.amd.com/rocm/whl-next/), so the index paths are best-effort and the
  # source build below is the real path.
  if [ -n "$AITER_SPEC" ] && uv pip install --python "$PY" "$AITER_SPEC" 2>/dev/null; then
    echo "installed $AITER_SPEC"
  elif uv pip install --python "$PY" --index-url "$ROCM_WHEEL_INDEX" \
        --extra-index-url https://pypi.org/simple "amd-aiter==${AITER_VERSION}" 2>/dev/null; then
    echo "installed amd-aiter==${AITER_VERSION} from an index"
  else
    warn "no amd-aiter wheel found; building from ROCm/aiter @ ${AITER_COMMIT:0:12} (v${AITER_VERSION})"
    AW="$(mktemp -d)"
    git clone --filter=blob:none --no-checkout https://github.com/ROCm/aiter.git "$AW/aiter"
    git -C "$AW/aiter" fetch --depth 1 origin tag "v${AITER_VERSION}" || \
      git -C "$AW/aiter" fetch --depth 1 origin "$AITER_COMMIT" || true
    git -C "$AW/aiter" checkout --detach "$AITER_COMMIT" || die "aiter commit $AITER_COMMIT not found"
    [ "$(git -C "$AW/aiter" rev-parse HEAD)" = "$AITER_COMMIT" ] || die "aiter checkout mismatch"
    ( cd "$AW/aiter" && GPU_ARCHS="$GFX_ARCH" PREBUILD_KERNELS=0 AITER_USE_SYSTEM_TRITON=1 \
        SETUPTOOLS_SCM_PRETEND_VERSION="$AITER_VERSION" \
        uv pip install --python "$PY" --no-build-isolation --no-deps . )
    rm -rf "$AW"
  fi
  # aiter is installed with --no-deps (so it cannot replace torch/triton), but aiter >= 0.1.21
  # needs its own Python deps — notably `flydsl`, which aiter/__init__.py imports via
  # topk_select. Install them explicitly; unpinned so already-satisfied ones are left alone.
  AITER_FLYDSL="${RADIANCE_FLYDSL_SPEC:-flydsl==0.3.4.1}"
  uv pip install --python "$PY" pandas psutil matplotlib pyyaml einops pybind11 ninja \
    "$AITER_FLYDSL" \
    || warn "installing aiter runtime deps failed; aiter import may fail until they are present"
  "$PY" -c 'import importlib.metadata as m; print("aiter", m.version("amd-aiter"))' \
    || warn "aiter version check failed"
  "$PY" -c 'import flydsl; print("flydsl OK")' || warn "flydsl import failed"
else
  log "AITER skipped (set RADIANCE_INSTALL_AITER=1 for the AITER fallback backends/GEMM)"
fi

# -----------------------------------------------------------------------------
if [ "${RADIANCE_SKIP_FORK:-0}" = "1" ]; then
  log "fork install skipped (RADIANCE_SKIP_FORK=1)"
else
  log "install this fork (mode=$INSTALL_MODE)"
  rm -rf "$ROOT/build" "$ROOT/CMakeCache.txt"
  # Remove any prior install (including a stale editable finder/.pth) before reinstalling.
  uv pip uninstall --python "$PY" vllm 2>/dev/null || true
  if [ "$INSTALL_MODE" = "editable" ]; then
    # A leftover real site-packages/vllm (from a prior wheel install) shadows the
    # editable package; drop it before reinstalling.
    rm -rf "$SP/vllm"
    VLLM_TARGET_DEVICE=rocm uv pip install --python "$PY" --no-build-isolation -e "$ROOT"
  else
    VLLM_TARGET_DEVICE=rocm uv pip install --python "$PY" --no-build-isolation "$ROOT"
  fi
  # vLLM's install re-resolves requirements/rocm.txt, which re-adds amd-quark. The
  # image removes it (circular import on vLLM main), so remove it again after the build.
  uv pip uninstall --python "$PY" amd-quark 2>/dev/null || true
fi
SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# Where the *live* vllm package lives: the source tree for an editable install, the
# site-packages copy for a wheel. The config copy targets this so an editable install
# is never shadowed by a real site-packages/vllm.
if [ "$INSTALL_MODE" = "editable" ]; then
  VLLM_PKG="$ROOT"
else
  VLLM_PKG="$SP"
fi

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
  ( cd "$SRC" && PATH="$VENV/bin:$PATH" GFX_ARCH="$GFX_ARCH" OUT="$SP/r4d.so" ./build.sh )
  "$PY" - <<PY || warn "r4d import check failed (see error above); continuing"
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
  "$PY" - <<PY || warn "radiance_mxfp4_fp8 import check failed (see error above); continuing"
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
  install -d "$VLLM_PKG/vllm/model_executor/layers/quantization/utils/configs"
  install -d "$VLLM_PKG/vllm/model_executor/layers/fused_moe/configs"
  install -d "$SP/aiter/ops/triton/configs/gemm" 2>/dev/null || true
  cp -f "$CONFIGS"/fp8/*.json  "$VLLM_PKG/vllm/model_executor/layers/quantization/utils/configs/"
  cp -f "$CONFIGS"/moe/*.json  "$VLLM_PKG/vllm/model_executor/layers/fused_moe/configs/"
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
  # Let vllm-targeting patches resolve the live package (source tree when editable).
  export RADIANCE_VLLM_ROOT="$VLLM_PKG"
  cd "$PATCHES"
  # Each script reports drift instead of corrupting a file; a reported failure
  # simply means that optimization is off for the installed package version.
  for p in patch_radiance_dispatch patch_unified_attention_lds patch_from_json_filter patch_dynamo_metrics; do
    echo "-- $p"
    "$PY" "$p.py" || warn "$p reported a drift/failure; inspect before relying on it"
  done
  cd "$ROOT"
else
  log "out-of-tree patches skipped"
fi

# -----------------------------------------------------------------------------
if [ "$SKIP_SMOKE" != "1" ]; then
  log "smoke test"
  "$PY" - <<'PY'
import importlib.metadata as md
import torch, vllm, amdsmi
vver = getattr(vllm, "__version__", None) or md.version("vllm")
print("vllm", vver, "| torch", torch.__version__,
      "| torchvision", md.version("torchvision"), "| triton", md.version("triton"))
try:
    import r4d
    print("r4d", r4d.__version__, "kernels", len(r4d.kernels()))
except Exception as e:
    print("r4d not loaded:", e)
try:
    import radiance_mxfp4_fp8  # noqa: F401
    import radiance_kernels  # noqa: F401
    print("radiance runtime modules OK")
except Exception as e:
    print("radiance runtime modules FAILED:", e)
PY
fi

log "done"
echo "Source the qualified defaults before serving:"
echo "  source \"$ROOT/radiance/radiance-env.sh\""
echo "  vllm serve <model> --tensor-parallel-size 2 ..."