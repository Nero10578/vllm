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

Every candidate config is a distinct Triton program and must be JIT-compiled
before it can be timed, so the runtime is dominated by the size of the search
space (not by the kernels). Use ``--smoke`` to confirm the path works, then
``--quick`` or a narrowed ``--lora-ranks`` for a fast real run.

Run on the serving box, one GPU visible::

    HIP_VISIBLE_DEVICES=0 \
    VLLM_TUNED_CONFIG_FOLDER=/home/arli/vllm-radiance-bm \
    .venv/bin/python radiance/tune_lora.py \
        --hidden-sizes 5376 --lora-ranks 16 32 64 128 \
        --max-loras 3 --batch-sizes 1 2 4 8 16 32
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
from vllm.lora.ops.triton_ops import lora_expand_op, lora_shrink_op
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_ORIG_SHRINK_CFG = lora_shrink_op.get_lora_op_configs
_ORIG_EXPAND_CFG = lora_expand_op.get_lora_op_configs
_FIRST_ERROR: dict[str, str] = {}


def shrink_space(quick: bool):
    if quick:
        blocks_m, blocks_n, split_ks = (32, 64, 128), (16, 32), (64,)
    else:
        blocks_m, blocks_n, split_ks = (16, 32, 64, 128), (16, 32, 64), (1, 64)
    for block_m in blocks_m:
        for block_n in blocks_n:
            for block_k in (128, 256):
                for split_k in split_ks:
                    for num_warps in (4, 8):
                        yield {
                            "block_m": block_m,
                            "block_n": block_n,
                            "block_k": block_k,
                            "split_k": split_k,
                            "num_warps": num_warps,
                            "num_stages": 2,
                            "num_ctas": 1,
                            "group_size_m": 8,
                        }


def expand_space(quick: bool):
    if quick:
        blocks_m, blocks_n = (32, 64, 128), (64, 128)
    else:
        blocks_m, blocks_n = (16, 32, 64, 128), (32, 64, 128)
    for block_m in blocks_m:
        for block_n in blocks_n:
            for block_k in (32, 64):
                for num_warps in (4, 8):
                    for num_stages in (2, 3):
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
    mapping = torch.arange(num_tokens, device=device, dtype=torch.int32) % max_loras
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


def _attempt(tag: str, fn) -> float | None:
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - first failure is reported, rest skipped
        _FIRST_ERROR.setdefault(tag, f"{type(exc).__name__}: {exc}")
        return None
    return 0.0


def tune_shrink(
    hidden, rank, num_slices, m, max_loras, dtype, device, space, warmup, iters
):
    inputs = torch.randn(m, hidden, dtype=dtype, device=device)
    lora_a = [
        torch.randn(max_loras, rank, hidden, dtype=dtype, device=device)
        for _ in range(num_slices)
    ]
    output = torch.empty(num_slices, m, rank, dtype=torch.float32, device=device)
    metadata = make_meta(max_loras, m, device).meta_args(m, True)

    best, best_us = None, float("inf")
    for config in space:
        lora_shrink_op.get_lora_op_configs = lambda *a, _c=config, **k: _c
        fn = lambda: lora_shrink(inputs, lora_a, output, *metadata, 1.0)
        us = _attempt("shrink", fn)
        if us is None:
            continue
        us = time_fn(fn, warmup, iters)
        if us < best_us:
            best_us, best = us, config
    lora_shrink_op.get_lora_op_configs = _ORIG_SHRINK_CFG
    return best


def tune_expand(
    hidden, rank, num_slices, m, max_loras, add_inputs, dtype, device,
    space, warmup, iters,
):
    inputs = torch.randn(num_slices, m, rank, dtype=torch.float32, device=device)
    lora_b = [
        torch.randn(max_loras, hidden, rank, dtype=dtype, device=device)
        for _ in range(num_slices)
    ]
    output = torch.empty(m, hidden * num_slices, dtype=dtype, device=device)
    metadata = make_meta(max_loras, m, device).meta_args(m, True)

    best, best_us = None, float("inf")
    for config in space:
        lora_expand_op.get_lora_op_configs = lambda *a, _c=config, **k: _c
        fn = lambda: lora_expand(inputs, lora_b, output, *metadata, 0, add_inputs)
        us = _attempt("expand", fn)
        if us is None:
            continue
        us = time_fn(fn, warmup, iters)
        if us < best_us:
            best_us, best = us, config
    lora_expand_op.get_lora_op_configs = _ORIG_EXPAND_CFG
    return best


def nested(data: dict, path: list) -> dict:
    node = data
    for key in path:
        node = node.setdefault(str(key), {})
    return node


def ranks_from_dir(path: str) -> list[int]:
    ranks = set()
    for cfg in Path(path).glob("*/adapter_config.json"):
        data = json.loads(cfg.read_text())
        if "r" in data:
            ranks.add(int(data["r"]))
    return sorted(ranks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[5376])
    parser.add_argument("--lora-ranks", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument(
        "--lora-dir",
        help="Read adapter ranks from */adapter_config.json under this dir "
        "(overrides --lora-ranks).",
    )
    parser.add_argument("--num-slices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-loras", type=int, default=3)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one config on one key per op, print the result, and exit.",
    )
    parser.add_argument(
        "--output-dir", default=os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
    )
    args = parser.parse_args()

    if not args.output_dir:
        parser.error("set --output-dir or VLLM_TUNED_CONFIG_FOLDER")
    if args.lora_dir:
        found = ranks_from_dir(args.lora_dir)
        if found:
            args.lora_ranks = found

    dtype = DTYPE_MAP[args.dtype]
    torch.cuda.init()
    torch.accelerator.set_device_index(0)
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")

    print(f"[tune_lora] device={gpu} dtype={args.dtype}")
    print(f"[tune_lora] hidden={args.hidden_sizes} ranks={args.lora_ranks} "
          f"slices={args.num_slices} max_loras={args.max_loras} m={args.batch_sizes}")
    print(f"[tune_lora] output_dir={args.output_dir}")

    shrink_data: dict = {}
    expand_data = {True: {}, False: {}}

    if args.smoke:
        h = args.hidden_sizes[0]
        r = args.lora_ranks[0]
        print("[smoke] running one shrink + one expand config...")
        cfg = tune_shrink(h, r, 1, 1, args.max_loras, dtype, "cuda",
                          list(shrink_space(True))[:1], 1, 1)
        print(f"[smoke] shrink {'-> ' + str(cfg) if cfg else 'FAILED: ' + _FIRST_ERROR.get('shrink', '')}")
        cfg = tune_expand(h, r, 1, 1, args.max_loras, True, dtype, "cuda",
                          list(expand_space(True))[:1], 1, 1)
        print(f"[smoke] expand {'-> ' + str(cfg) if cfg else 'FAILED: ' + _FIRST_ERROR.get('expand', '')}")
        print("[smoke] done")
        return

    jobs = []
    for hidden in args.hidden_sizes:
        for rank in args.lora_ranks:
            for num_slices in args.num_slices:
                for m in args.batch_sizes:
                    jobs.append(("shrink", hidden, rank, num_slices, m, None))
                    jobs.append(("expand", hidden, rank, num_slices, m, True))
                    jobs.append(("expand", hidden, rank, num_slices, m, False))

    n_shrink = len(list(shrink_space(args.quick)))
    n_expand = len(list(expand_space(args.quick)))
    print(f"[tune_lora] {len(jobs)} jobs; ~{n_shrink} shrink + ~{n_expand} expand "
          f"configs/compile each (this is the runtime driver)")

    for kind, hidden, rank, num_slices, m, add_inputs in tqdm(jobs, desc="tuning"):
        if kind == "shrink":
            cfg = tune_shrink(hidden, rank, num_slices, m, args.max_loras, dtype,
                              "cuda", shrink_space(args.quick), args.warmup, args.iters)
            if cfg is not None:
                nested(shrink_data, [args.max_loras, num_slices, m, hidden, rank]).update(cfg)
                tqdm.write(f"[shrink] h={hidden} r={rank} slices={num_slices} m={m} -> {cfg}")
        else:
            cfg = tune_expand(hidden, rank, num_slices, m, args.max_loras, add_inputs,
                              dtype, "cuda", expand_space(args.quick), args.warmup, args.iters)
            if cfg is not None:
                nested(expand_data[add_inputs], [args.max_loras, num_slices, m, rank, hidden]).update(cfg)
                tqdm.write(f"[expand={add_inputs}] h={hidden} r={rank} slices={num_slices} m={m} -> {cfg}")

    for tag, msg in _FIRST_ERROR.items():
        print(f"[tune_lora] WARNING all {tag} attempts failed; first error: {msg}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (
        (f"{gpu}_SHRINK.json", shrink_data),
        (f"{gpu}_EXPAND_TRUE.json", expand_data[True]),
        (f"{gpu}_EXPAND_FALSE.json", expand_data[False]),
    ):
        path = out / name
        path.write_text(json.dumps(data, indent=4) + "\n")
        print(f"[tune_lora] wrote {path}")


if __name__ == "__main__":
    main()
