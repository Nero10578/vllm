#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tune the vLLM Punica LoRA shrink/expand Triton kernels for a device.

The runtime loader (`vllm/lora/ops/triton_ops/utils.py`) reads three JSON
files from ``$VLLM_TUNED_CONFIG_FOLDER``:

    {gpu}_SHRINK.json
    {gpu}_EXPAND_TRUE.json
    {gpu}_EXPAND_FALSE.json

with the schema ``data[max_loras][num_slices][m][k][n] = config`` where, for
shrink, ``k=hidden_size`` and ``n=rank``; for expand, ``k=rank`` and
``n=hidden_size``. This script produces those files.

It drives the production ops (``torch.ops.vllm.lora_shrink`` /
``lora_expand``) and swaps the launch config by patching the module-level
``get_lora_op_configs``, so the measured timings come from the exact kernel
the server runs.

Run on the serving box, one GPU visible::

    HIP_VISIBLE_DEVICES=0 \
    VLLM_TUNED_CONFIG_FOLDER=/home/arli/vllm-radiance-bm \
    .venv/bin/python radiance/tune_lora.py \
        --hidden-sizes 5376 --lora-ranks 16 32 64 128 \
        --max-loras 3 --batch-sizes 1 2 4 8 16 24 32
"""

import argparse
import json
import os
from pathlib import Path

import torch

from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
from vllm.lora.ops.triton_ops import lora_expand_op, lora_shrink_op
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def shrink_space(quick: bool):
    blocks = (16, 32, 64, 128) if quick else (16, 32, 64, 128, 256)
    for block_m in blocks:
        for block_n in (16, 32, 64, 128):
            for block_k in (128, 256):
                for split_k in ((1, 64) if quick else (1, 8, 64)):
                    for num_warps in (4, 8):
                        for num_stages in ((2, 3) if quick else (2, 3, 4)):
                            for group_size_m in (1, 8):
                                yield {
                                    "block_m": block_m,
                                    "block_n": block_n,
                                    "block_k": block_k,
                                    "split_k": split_k,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "num_ctas": 1,
                                    "group_size_m": group_size_m,
                                }


def expand_space(quick: bool):
    blocks = (16, 32, 64, 128) if quick else (16, 32, 64, 128, 256)
    for block_m in blocks:
        for block_n in (16, 32, 64, 128):
            for block_k in (16, 32, 64, 128):
                for num_warps in (4, 8):
                    for num_stages in ((2, 3) if quick else (2, 3, 4)):
                        yield {
                            "block_m": block_m,
                            "block_n": block_n,
                            "block_k": block_k,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "num_ctas": 1,
                        }


def make_meta(max_loras: int, num_tokens: int, device: str) -> LoRAKernelMeta:
    meta = LoRAKernelMeta.make(max_loras, num_tokens, device)
    mapping = (
        torch.arange(num_tokens, device=device, dtype=torch.int32) % max_loras
    )
    meta.prepare_tensors(mapping)
    return meta


def time_fn(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) / iters * 1000.0


def tune_shrink(
    hidden: int,
    rank: int,
    num_slices: int,
    m: int,
    max_loras: int,
    dtype: torch.dtype,
    device: str,
    space,
    warmup: int,
    iters: int,
) -> dict:
    inputs = torch.randn(m, hidden, dtype=dtype, device=device)
    lora_a = [
        torch.randn(max_loras, rank, hidden, dtype=dtype, device=device)
        for _ in range(num_slices)
    ]
    output = torch.empty(num_slices, m, rank, dtype=torch.float32, device=device)
    meta = make_meta(max_loras, m, device)
    metadata = meta.meta_args(m, specialize_active_lora=True)

    best = None
    best_us = float("inf")
    for config in space:
        lora_shrink_op.get_lora_op_configs = lambda *a, _c=config, **k: _c
        try:
            fn = lambda: lora_shrink(
                inputs, lora_a, output, *metadata, 1.0
            )
            fn()
            torch.cuda.synchronize()
        except Exception:
            continue
        us = time_fn(fn, warmup, iters)
        if us < best_us:
            best_us, best = us, config
    lora_shrink_op.get_lora_op_configs = _ORIG_SHRINK_CFG
    return best


def tune_expand(
    hidden: int,
    rank: int,
    num_slices: int,
    m: int,
    max_loras: int,
    add_inputs: bool,
    dtype: torch.dtype,
    device: str,
    space,
    warmup: int,
    iters: int,
) -> dict:
    inputs = torch.randn(num_slices, m, rank, dtype=torch.float32, device=device)
    lora_b = [
        torch.randn(max_loras, hidden, rank, dtype=dtype, device=device)
        for _ in range(num_slices)
    ]
    output = torch.empty(m, hidden * num_slices, dtype=dtype, device=device)
    meta = make_meta(max_loras, m, device)
    metadata = meta.meta_args(m, specialize_active_lora=True)

    best = None
    best_us = float("inf")
    for config in space:
        lora_expand_op.get_lora_op_configs = lambda *a, _c=config, **k: _c
        try:
            fn = lambda: lora_expand(
                inputs, lora_b, output, *metadata, 0, add_inputs
            )
            fn()
            torch.cuda.synchronize()
        except Exception:
            continue
        us = time_fn(fn, warmup, iters)
        if us < best_us:
            best_us, best = us, config
    lora_expand_op.get_lora_op_configs = _ORIG_EXPAND_CFG
    return best


_ORIG_SHRINK_CFG = lora_shrink_op.get_lora_op_configs
_ORIG_EXPAND_CFG = lora_expand_op.get_lora_op_configs


def nested(data: dict, path: list) -> dict:
    node = data
    for key in path:
        node = node.setdefault(str(key), {})
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[5376])
    parser.add_argument("--lora-ranks", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--num-slices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-loras", type=int, default=3)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32]
    )
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("VLLM_TUNED_CONFIG_FOLDER"),
    )
    args = parser.parse_args()

    if not args.output_dir:
        parser.error("set --output-dir or VLLM_TUNED_CONFIG_FOLDER")

    device = "cuda"
    dtype = DTYPE_MAP[args.dtype]
    torch.cuda.init()
    torch.accelerator.set_device_index(0)

    shrink_data: dict = {}
    expand_true: dict = {}
    expand_false: dict = {}

    for hidden in args.hidden_sizes:
        for rank in args.lora_ranks:
            for num_slices in args.num_slices:
                for m in args.batch_sizes:
                    cfg = tune_shrink(
                        hidden, rank, num_slices, m, args.max_loras, dtype,
                        device, shrink_space(args.quick), args.warmup, args.iters,
                    )
                    if cfg is not None:
                        nested(shrink_data, [args.max_loras, num_slices, m, hidden, rank]).update(cfg)
                        print(f"[shrink] h={hidden} r={rank} slices={num_slices} m={m} -> {cfg}")
                    for add_inputs, target in ((True, expand_true), (False, expand_false)):
                        cfg = tune_expand(
                            hidden, rank, num_slices, m, args.max_loras, add_inputs,
                            dtype, device, expand_space(args.quick), args.warmup, args.iters,
                        )
                        if cfg is not None:
                            nested(target, [args.max_loras, num_slices, m, rank, hidden]).update(cfg)
                            print(f"[expand={add_inputs}] h={hidden} r={rank} slices={num_slices} m={m} -> {cfg}")

    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (
        (f"{gpu}_SHRINK.json", shrink_data),
        (f"{gpu}_EXPAND_TRUE.json", expand_true),
        (f"{gpu}_EXPAND_FALSE.json", expand_false),
    ):
        path = out / name
        path.write_text(json.dumps(data, indent=4) + "\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
