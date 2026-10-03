#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the native RDNA W4A16 MoE kernels.

Times ``moe_gptq_gemm_rdna3`` (scalar, decode) and optionally
``moe_gptq_gemm_rdna4_wmma`` (gfx12 WMMA, prefill) at fixed GLM-4.7 shapes so
wave-size / tiling / pipelining changes can be A/B'd at the kernel level. This
is far more sensitive than end-to-end tok/s, which hides a few percent in
run-to-run noise and in the non-MoE part of the step.

Run on each commit and diff the printed microseconds.

Example (GLM-4.7, TP8):
    .venv/bin/python benchmarks/kernels/benchmark_moe_rdna.py --dtype bfloat16
    .venv/bin/python benchmarks/kernels/benchmark_moe_rdna.py --wmma
"""

import argparse
import statistics

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

device = "cuda"


def _make_packed_weights(E, K, N, seed=0):
    """Random 4-bit weights [E, K/8, N] int32, GPTQ-shuffled like the loader."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 16, (E, K, N), dtype=torch.int32, generator=g).to(device)
    packed = torch.zeros(E, K // 8, N, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (w[:, i::8, :] & 0xF) << (i * 4)
    for e in range(E):
        we = packed[e].contiguous()
        ops.gptq_shuffle(we, 4)
        packed[e] = we
    return packed


def _make_scales(E, groups, N, dtype, seed=0):
    g = torch.Generator().manual_seed(seed + 1)
    s = torch.rand(E, groups, N, generator=g, dtype=torch.float32) * 0.1
    return s.to(dtype).to(device)


def _make_qzeros(E, groups, N):
    zeros = torch.full(
        (groups, N),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device=device,
    )
    qz = pack_quantized_values_into_int32(
        zeros, scalar_types.uint4b8, packed_dim=1
    )
    return qz.unsqueeze(0).expand(E, -1, -1).contiguous()


def _time_us(fn, warmup, iters):
    """Mean microseconds per call, measured over an amortized loop."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _median_us(fn, warmup, iters, repeats):
    return statistics.median(
        _time_us(fn, warmup, iters) for _ in range(repeats)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--hidden-size", type=int, default=5120)
    p.add_argument("--intermediate-size", type=int, default=1536)
    p.add_argument("--num-experts", type=int, default=160)
    p.add_argument("--tp-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--wmma", action="store_true", help="also time the gfx12 WMMA op")
    args = p.parse_args()

    if not current_platform.is_rocm():
        raise SystemExit("ROCm only")
    if not hasattr(torch.ops, "_rocm_C") or not hasattr(
        torch.ops._rocm_C, "moe_gptq_gemm_rdna3"
    ):
        raise SystemExit("moe_gptq_gemm_rdna3 not registered in this build")

    dtype = getattr(torch, args.dtype)
    E = max(1, args.num_experts // args.tp_size)
    inter = args.intermediate_size // args.tp_size
    K_hidden = args.hidden_size
    N_gate_up = 2 * inter
    groups_w1 = K_hidden // args.group_size
    groups_w2 = inter // args.group_size

    torch.manual_seed(0)
    w1 = _make_packed_weights(E, K_hidden, N_gate_up)
    w1_s = _make_scales(E, groups_w1, N_gate_up, dtype)
    w1_z = _make_qzeros(E, groups_w1, N_gate_up)
    w2 = _make_packed_weights(E, inter, K_hidden, seed=17)
    w2_s = _make_scales(E, groups_w2, K_hidden, dtype, seed=17)
    w2_z = _make_qzeros(E, groups_w2, K_hidden)

    has_wmma = args.wmma and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna4_wmma")

    print(
        f"dtype={args.dtype} E_local={E} hidden={K_hidden} "
        f"inter_local={inter} N_gate_up={N_gate_up} top_k={args.top_k} "
        f"group={args.group_size}"
    )
    header = f"{'M':>6} {'bsm':>4} {'w1 us':>10} {'w2 us':>10} {'moe us':>10}"
    if has_wmma:
        header += f" {'wmma us':>10}"
    print(header)

    empty = torch.empty(0, device=device)
    for M in args.tokens:
        bsm = 1 if M <= 4 else 4
        topk_ids = torch.randint(0, E, (M, args.top_k), dtype=torch.int32, device=device)
        si, ei, ntp = moe_align_block_size(topk_ids, bsm, E)

        x1 = torch.randn(M, K_hidden, dtype=dtype, device=device)
        out1 = torch.zeros(M * args.top_k, N_gate_up, dtype=dtype, device=device)

        def run_w1():
            ops.moe_gptq_gemm_rdna3(
                x1, out1, w1, w1_s, w1_z, empty, si, ei, ntp,
                args.top_k, bsm, False, 0,
            )

        act = torch.zeros(M * args.top_k, inter, dtype=dtype, device=device)
        out2 = torch.zeros(M, K_hidden, dtype=dtype, device=device)

        def run_w2():
            ops.moe_gptq_gemm_rdna3(
                act, out2, w2, w2_s, w2_z, empty, si, ei, ntp,
                1, bsm, False, args.top_k,
            )

        t1 = _median_us(run_w1, args.warmup, args.iters, args.repeats)
        t2 = _median_us(run_w2, args.warmup, args.iters, args.repeats)

        line = f"{M:>6} {bsm:>4} {t1:>10.2f} {t2:>10.2f} {t1 + t2:>10.2f}"

        if has_wmma:
            bsm_w = 16
            si_w, ei_w, ntp_w = moe_align_block_size(topk_ids, bsm_w, E)
            out1_w = torch.zeros(M * args.top_k, N_gate_up, dtype=dtype, device=device)
            out2_w = torch.zeros(M, K_hidden, dtype=dtype, device=device)

            def run_wmma1():
                ops.moe_gptq_gemm_rdna4_wmma(
                    x1, out1_w, w1, w1_s, w1_z, empty, si_w, ei_w, ntp_w,
                    args.top_k, bsm_w, False, 0,
                )

            def run_wmma2():
                ops.moe_gptq_gemm_rdna4_wmma(
                    act, out2_w, w2, w2_s, w2_z, empty, si_w, ei_w, ntp_w,
                    1, bsm_w, False, args.top_k,
                )

            tw = _median_us(lambda: (run_wmma1(), run_wmma2()), args.warmup, args.iters, args.repeats)
            line += f" {tw:>10.2f}"

        print(line)


if __name__ == "__main__":
    main()