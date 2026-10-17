"""RADIANCE gfx1201 block-FP8 (W8A8, 128x128) GEMM dispatch: routes vLLM's block-scaled GEMM
through the hand-written fp8-WMMA kernel.

Why: the ROCm block-FP8 route is `_w8a8_triton_block_scaled_mm`, i.e. `tl.dot` on e4m3 operands.
Triton will not emit gfx1201's fp8 matrix instruction -- it upconverts and runs the 16-bit WMMA.
Measured here, register-resident: fp8 WMMA 325.2 TFLOP/s vs f16 160.2, while Triton's own fp8
tl.dot manages only 43.3. So the GEMM runs on the conversion path, at a fraction of the card's fp8
matrix throughput. `radiance_fp8_block_wmma.hip` is the same math on `v_wmma_f32_16x16x16_fp8_fp8`.

Numeric contract: identical semantics to the Triton kernel (per-(1,128) activation scale,
per-(128,128) weight scale, fp32 accumulation, applied inside the K loop). Accumulation order
differs, so output is not required to be bit-identical; RADIANCE_FP8_WMMA_VERIFY=1 checks the
relative error against the Triton path on the first calls.

Opt-in: RADIANCE_FP8_WMMA=1 (default off). When unset, or the extension is missing, or the shape is
not one the kernel can serve, this returns None and the caller keeps its existing route.
"""
import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_FP8_WMMA", "0") == "1"
# Smallest M the kernel serves. Below this the tile wastes rows (BMF=128 computes 128 rows for a
# handful) and the tuned Triton/aiter split-K paths are already good, so the default leaves the
# decode band entirely to them and takes only the prefill band.
MIN_M = int(os.environ.get("RADIANCE_FP8_WMMA_MIN_M", "128"))
VERIFY = os.environ.get("RADIANCE_FP8_WMMA_VERIFY", "0") == "1"
_VERIFY_EVERY = int(os.environ.get("RADIANCE_FP8_WMMA_VERIFY_EVERY", "1"))
_VERIFY_TOL = float(os.environ.get("RADIANCE_FP8_WMMA_VERIFY_TOL", "0.02"))

_ext_state = {"ext": None, "tried": False}
_verify_calls = 0


def _ext():
    """Import the HIP extension on first use, never at module scope.

    The compiled module links libamdhip64.so; importing it while vLLM is still parsing config in
    the parent process initialises HIP early and forces the engine core to spawn instead of fork.
    radiance_kernels is imported by the plugin loader in the worker, and the first GEMM is at model
    load, so a lazy import here is always after HIP is up.
    """
    if not _ext_state["tried"]:
        _ext_state["tried"] = True
        try:
            import radiance_fp8_block_wmma as m

            _ext_state["ext"] = m
            sys.stderr.write(
                f"[radiance.fp8_block] block-FP8 fp8-WMMA GEMM ON (M>={MIN_M}); "
                f"kernel: {getattr(m, '__file__', '?')}\n")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[radiance.fp8_block] extension unavailable, Triton stays: {e!r}\n")
    return _ext_state["ext"]


def eligible(A, B, As, Bs, out_dtype) -> bool:
    if not ENABLED or _ext() is None:
        return False
    if out_dtype is not torch.bfloat16:
        return False
    if A.dim() != 2 or B.dim() != 2 or As.dim() != 2 or Bs.dim() != 2:
        return False
    M, K = int(A.shape[0]), int(A.shape[1])
    N = int(B.shape[0])
    if M < MIN_M or K % 128 or N % 128:
        return False
    if int(B.shape[1]) != K:
        return False
    if A.dtype is not torch.float8_e4m3fn or B.dtype is not torch.float8_e4m3fn:
        return False
    # As: [M, K/128] (activation group shape (1,128)); Bs: [N/128, K/128] (weight block (128,128)).
    if int(As.shape[0]) != M or int(As.shape[1]) != K // 128:
        return False
    if int(Bs.shape[0]) != N // 128 or int(Bs.shape[1]) != K // 128:
        return False
    return True


@torch.library.custom_op("radiance::fp8_block_wmma", mutates_args=())
def fp8_block_wmma(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor,
                   Bs: torch.Tensor) -> torch.Tensor:
    ext = _ext()
    M, K = int(A.shape[0]), int(A.shape[1])
    N = int(B.shape[0])
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    asc = As.float().contiguous()
    bsc = Bs.float().contiguous()
    out = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
    ext.launch(A.data_ptr(), B.data_ptr(), asc.data_ptr(), bsc.data_ptr(), out.data_ptr(),
               M, N, K, torch.cuda.current_stream().cuda_stream)
    return out


@fp8_block_wmma.register_fake
def _(A, B, As, Bs):
    return torch.empty((A.shape[0], B.shape[0]), device=A.device, dtype=torch.bfloat16)


def _triton_reference(A, B, As, Bs, out_dtype):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    return w8a8_triton_block_scaled_mm(A, B, As, Bs, [128, 128], out_dtype)


def _maybe_verify(y, A, B, As, Bs, out_dtype):
    global _verify_calls
    _verify_calls += 1
    if _verify_calls > _VERIFY_EVERY * 40 or (_verify_calls - 1) % max(1, _VERIFY_EVERY):
        return y
    try:
        ref = _triton_reference(A, B, As, Bs, out_dtype)
        rel = ((y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)).item()
        M = int(A.shape[0])
        N = int(B.shape[0])
        K = int(A.shape[1])
        flag = "**WRONG**" if rel > _VERIFY_TOL else "ok"
        sys.stderr.write(f"[radiance.fp8_block.verify] N={N} K={K} M={M} "
                         f"rel_vs_triton={rel:.5f} {flag}\n")
        sys.stderr.flush()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[radiance.fp8_block.verify] failed: {e!r}\n")
    return y


def block_scaled_mm(A, B, As, Bs, out_dtype):
    """Hand-written fp8-WMMA block-FP8 GEMM, or None to leave the caller's route in place."""
    if not eligible(A, B, As, Bs, out_dtype):
        return None
    y = torch.ops.radiance.fp8_block_wmma(A, B, As, Bs)
    if VERIFY:
        y = _maybe_verify(y, A, B, As, Bs, out_dtype)
    return y
