#!/usr/bin/env python3
"""Correctness + speed gate for the gfx1201 block-FP8 fp8-WMMA GEMM (radiance_fp8_block_wmma).

Runs on the serving box, inside the vLLM venv, against the SAME Triton kernel it replaces. The
kernel's accumulation order differs from Triton's, so it is not bit-identical by construction --
this checks the relative error stays at rounding level and reports the speedup on the real Gemma
shapes. Nothing here needs a loaded model: the data is synthetic block-FP8.

    source <repo>/radiance/radiance-env.sh
    python <repo>/radiance/gate_fp8_block.py [--tp-size 4] [--tol 0.02]

Exit status is non-zero if any shape fails the tolerance, so it can gate a serve script.
"""
import argparse
import sys

import torch

# Gemma-4-31B-it TP4 per-rank linear shapes, read from the running server's W8A8 config lookup:
# (N, K) pairs. Kept in sync with benchmark_w8a8_block_fp8.py's GEMMA4_31B_TP4_SHAPES.
GEMMA4_31B_TP4_SHAPES = [
    (4096, 5376),
    (5376, 2048),
    (10752, 5376),
    (5376, 5376),
    (5120, 5376),
    (5376, 4096),
]
BLOCK = 128
M_VALUES = [128, 256, 512, 1024, 2048, 2496]


def make_inputs(N, K, M, device="cuda"):
    A = (torch.randn(M, K, device=device) * 0.5).to(torch.float8_e4m3fn)
    As = (torch.rand(M, K // BLOCK, device=device) * 0.02 + 1e-3).float()
    W = (torch.randn(N, K, device=device) * 0.05).to(torch.float8_e4m3fn)
    Bs = (torch.rand(N // BLOCK, K // BLOCK, device=device) * 0.02 + 1e-3).float()
    return A, W, As, Bs


def timeit(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0    # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    if args.tp_size != 4:
        print(f"note: shapes are the TP4 list; tp-size={args.tp_size} is not TP4", file=sys.stderr)

    import radiance_fp8_block
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    assert radiance_fp8_block._ext() is not None, "extension missing (build it first)"

    def ref(A, W, As, Bs):
        return w8a8_triton_block_scaled_mm(A, W, As, Bs, [BLOCK, BLOCK], torch.bfloat16)

    def wmma(A, W, As, Bs):
        return torch.ops.radiance.fp8_block_wmma(A, W, As, Bs)

    print(f"device={torch.cuda.get_device_name(0)} tol={args.tol}")
    print(f"{'N':>7} {'K':>6} {'M':>6} {'rel':>10} {'triton_us':>10} {'wmma_us':>9} {'speedup':>8}")
    worst = 0.0
    failures = []
    for N, K in GEMMA4_31B_TP4_SHAPES:
        for M in M_VALUES:
            A, W, As, Bs = make_inputs(N, K, M)
            r = ref(A, W, As, Bs).float()
            y = wmma(A, W, As, Bs).float()
            rel = ((y - r).norm() / r.norm().clamp_min(1e-9)).item()
            tt = timeit(lambda: ref(A, W, As, Bs), args.iters)
            tw = timeit(lambda: wmma(A, W, As, Bs), args.iters)
            worst = max(worst, rel)
            ok = "" if rel <= args.tol else "  **WRONG**"
            print(f"{N:>7} {K:>6} {M:>6} {rel:>10.5f} {tt:>10.1f} {tw:>9.1f} {tt / tw:>7.2f}x{ok}")
            if rel > args.tol:
                failures.append((N, K, M, rel))
            del A, W, As, Bs, r, y
    print(f"worst rel={worst:.5f} over {len(GEMMA4_31B_TP4_SHAPES) * len(M_VALUES)} cells")
    if failures:
        print(f"FAIL: {len(failures)} cells over tol {args.tol}: {failures}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
