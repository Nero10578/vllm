#!/usr/bin/env bash
# Radiance qualified runtime defaults for baremetal ROCm / gfx1201 (R9700).
#
# These mirror the ENV block of the vllm-radiance Dockerfile's release stage.
# Source this before launching the server:
#
#     source radiance/radiance-env.sh
#     vllm serve <model> --tensor-parallel-size 2 ...
#
# Every value is overridable: export your override AFTER sourcing this file.
# See the vllm-radiance DOCKERHUB.md for the full knob reference.

# --- ROCm / device targeting -------------------------------------------------
# ROCm 10 keeps the SDK under /opt/rocm/core-10.0; older layouts use /opt/rocm.
if [ -z "${ROCM_PATH:-}" ]; then
  if [ -d /opt/rocm/core-10.0 ]; then ROCM_PATH=/opt/rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export HIP_PLATFORM="${HIP_PLATFORM:-amd}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1201}"
export RADIANCE_GFX_ARCH="${RADIANCE_GFX_ARCH:-gfx1201}"
export HIP_ARCHITECTURES="${HIP_ARCHITECTURES:-gfx1201}"
export AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1201}"
export GPU_ARCHS="${GPU_ARCHS:-gfx1201}"

# --- runtime library/service paths (required: triton under ROCm 10) ----------
# Both the system tree and the core-10.0 tree must be on the loader path so
# triton/vLLM find the ROCm 10 kernel and BLAS libraries.
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/core-10.0/lib:${LD_LIBRARY_PATH:-}"
export TRITON_USE_ROCM="${TRITON_USE_ROCM:-1}"
# AMD-recommended FP16 GEMM path on the R9700.
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-1}"

# --- runtime hygiene ---------------------------------------------------------
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRITON_CACHE_AUTOTUNING="${TRITON_CACHE_AUTOTUNING:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

# --- Radiance feature flags (qualified defaults) -----------------------------
export RADIANCE_USE_R4D="${RADIANCE_USE_R4D:-1}"
export RADIANCE_USE_R4D_GDN="${RADIANCE_USE_R4D_GDN:-1}"
export RADIANCE_R4D_REPORT="${RADIANCE_R4D_REPORT:-1}"
export RADIANCE_USE_R4D_AR="${RADIANCE_USE_R4D_AR:-1}"
export RADIANCE_USE_R4D_AR_QUANT="${RADIANCE_USE_R4D_AR_QUANT:-1}"
export RADIANCE_SKINNY_GEMM="${RADIANCE_SKINNY_GEMM:-1}"
export RADIANCE_GDN_META="${RADIANCE_GDN_META:-1}"
export RADIANCE_GDN_MERGE_INPROJ="${RADIANCE_GDN_MERGE_INPROJ:-1}"
export RADIANCE_GDN_FUSED_UPDATE="${RADIANCE_GDN_FUSED_UPDATE:-1}"
export RADIANCE_GDN_FUSED_MAX_ITEMS="${RADIANCE_GDN_FUSED_MAX_ITEMS:-32}"
export RADIANCE_GDN_SHARED_BUILD="${RADIANCE_GDN_SHARED_BUILD:-1}"
export RADIANCE_TOPK_TRITON_MIN_ROWS="${RADIANCE_TOPK_TRITON_MIN_ROWS:-1}"
export RADIANCE_TOPK_COMPOSITE="${RADIANCE_TOPK_COMPOSITE:-1}"
export RADIANCE_TOPK_COMPOSITE_KCAP="${RADIANCE_TOPK_COMPOSITE_KCAP:-64}"

# MXFP4 / W4A8 are opt-in: the qualified default is native FP8 weights.
export RADIANCE_MXFP4="${RADIANCE_MXFP4:-0}"
export RADIANCE_MXFP4_W4A8="${RADIANCE_MXFP4_W4A8:-0}"
export RADIANCE_MXFP4_W4A8_MIN_M="${RADIANCE_MXFP4_W4A8_MIN_M:-0}"
export RADIANCE_QUARK_BF16_MTP="${RADIANCE_QUARK_BF16_MTP:-0}"
export RADIANCE_MXFP4_DECODE_MAX_M="${RADIANCE_MXFP4_DECODE_MAX_M:-64}"
export RADIANCE_MXFP4_TN4_MIN_M="${RADIANCE_MXFP4_TN4_MIN_M:-2048}"
export RADIANCE_MXFP4_EPIFAST="${RADIANCE_MXFP4_EPIFAST:-1}"
export RADIANCE_MXFP4_WPERM="${RADIANCE_MXFP4_WPERM:-0}"
export RADIANCE_MXFP4_DECODE_NT="${RADIANCE_MXFP4_DECODE_NT:-0}"
export RADIANCE_MXFP4_A_TILED_MIN_M="${RADIANCE_MXFP4_A_TILED_MIN_M:-0}"
export RADIANCE_GDN_NORM_QUANT="${RADIANCE_GDN_NORM_QUANT:-0}"
export RADIANCE_NORMQUANT_FUSION="${RADIANCE_NORMQUANT_FUSION:-0}"
export RADIANCE_MXFP4_HOIST_QUANT="${RADIANCE_MXFP4_HOIST_QUANT:-0}"
export RADIANCE_MXFP4_TRACED_QUANT="${RADIANCE_MXFP4_TRACED_QUANT:-0}"
export RADIANCE_FP8_STREAM="${RADIANCE_FP8_STREAM:-0}"

export RADIANCE_KV_GROUP_OPT="${RADIANCE_KV_GROUP_OPT:-1}"
export RADIANCE_AR_QNT="${RADIANCE_AR_QNT:-1024}"
export RADIANCE_AR_QNB="${RADIANCE_AR_QNB:-96}"
export RADIANCE_PRESHUFFLE="${RADIANCE_PRESHUFFLE:-1}"
export RADIANCE_ATTN_TUNE="${RADIANCE_ATTN_TUNE:-1}"
export RADIANCE_FUSE_RMS_QUANT="${RADIANCE_FUSE_RMS_QUANT:-1}"

export RADIANCE_DYNAMIC_DRAFT="${RADIANCE_DYNAMIC_DRAFT:-1}"
export RADIANCE_DRAFT_SCHEDULE="${RADIANCE_DRAFT_SCHEDULE:-1:8,2:7,4:6,8:5,16:4}"
export RADIANCE_DRAFT_TAU="${RADIANCE_DRAFT_TAU:-0.28}"
export RADIANCE_DYNAMIC_WIDTH="${RADIANCE_DYNAMIC_WIDTH:-1}"
export RADIANCE_DYNW_MIN_BATCH="${RADIANCE_DYNW_MIN_BATCH:-5}"
export RADIANCE_DRAFT_RERANK="${RADIANCE_DRAFT_RERANK:-64}"
export RADIANCE_VERIFY_HEAD="${RADIANCE_VERIFY_HEAD:-1}"
export R4D_ATTN_FP8="${R4D_ATTN_FP8:-0}"
export RADIANCE_FAST_DRAFT="${RADIANCE_FAST_DRAFT:-0}"
export RADIANCE_RUN_BWTEST="${RADIANCE_RUN_BWTEST:-1}"