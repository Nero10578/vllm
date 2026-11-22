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
before it can be timed, so runtime is dominated by the size of the search
space. Candidates whose staged tiles exceed the device LDS budget would fail
to compile (`OutOfResources`), so they are pruned up front with an LDS
estimate; the runtime catch remains as a backstop. Jobs are independent and
are sharded across all visible GPUs with ``multiprocessing`` (spawn).

Run on the serving box with all target GPUs visible::

    VLLM_TUNED_CONFIG_FOLDER=/home/arli/vllm-radiance-bm \
    .venv/bin/python radiance/tune_lora.py \
        --hidden-sizes 5376 --lora-dir /home/arli/loras-gemma \
        --max-loras 3 --batch-sizes 1 2 4 8 16 32
"""

import argparse
import json
import multiprocessing as mp
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

# Staged operand element sizes: shrink A/B are the activation/weight (2 B each
# for bf16/fp16); expand stages A as fp32 and B as bf16.
_ELEM_A_SHRINK = 2
_ELEM_B_SHRINK = 2
_ELEM_A_EXPAND = 4
_ELEM_B_EXPAND = 2


def _is_oom(exc: BaseException) -> bool:
    return "OutOfResources" in type(exc).__name__ or "out of resource" in str(exc)


def shrink_space(quick: bool):
    # split_k matters with m: vLLM's default uses split_k=64 for small batches
    # and 8 for m>=128, so both must be in the grid for prefill-sized m.
    if quick:
        blocks_m, blocks_n = (32, 64, 128), (16, 32)
        blocks_k, split_ks, stages = (128, 256), (8, 64), (2,)
    else:
        blocks_m, blocks_n = (16, 32, 64, 128), (16, 32, 64, 128)
        blocks_k, split_ks, stages = (64, 128, 256), (1, 8, 64), (1, 2, 3)
    for block_m in blocks_m:
        for block_n in blocks_n:
            for block_k in blocks_k:
                for split_k in split_ks:
                    for num_warps in (4, 8):
                        for num_stages in stages:
                            yield {
                                "block_m": block_m,
                                "block_n": block_n,
                                "block_k": block_k,
                                "split_k": split_k,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                                "num_ctas": 1,
                                "group_size_m": 8,
                            }


def expand_space(quick: bool):
    if quick:
        blocks_m, blocks_n = (32, 64, 128), (64, 128)
        blocks_k, stages = (32, 64), (2, 3)
    else:
        blocks_m, blocks_n = (16, 32, 64, 128), (32, 64, 128, 256)
        blocks_k, stages = (16, 32, 64, 128), (1, 2, 3)
    for block_m in blocks_m:
        for block_n in blocks_n:
            for block_k in blocks_k:
                for num_warps in (4, 8):
                    for num_stages in stages:
                        yield {
                            "block_m": block_m,
                            "block_n": block_n,
                            "block_k": block_k,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "num_ctas": 1,
                        }


def lds_bytes_shrink(c: dict) -> int:
    tile = c["block_m"] * c["block_k"] * _ELEM_A_SHRINK
    tile += c["block_k"] * c["block_n"] * _ELEM_B_SHRINK
    return c["num_stages"] * tile


def lds_bytes_expand(c: dict) -> int:
    tile = c["block_m"] * c["block_k"] * _ELEM_A_EXPAND
    tile += c["block_k"] * c["block_n"] * _ELEM_B_EXPAND
    return c["num_stages"] * tile


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
    ok, failed, first_err = 0, 0, None
    for config in space:
        lora_shrink_op.get_lora_op_configs = lambda *a, _c=config, **k: _c
        fn = lambda: lora_shrink(inputs, lora_a, output, *metadata, 1.0)
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if first_err is None and not _is_oom(exc):
                first_err = f"{type(exc).__name__}: {exc}"
            continue
        ok += 1
        us = time_fn(fn, warmup, iters)
        if us < best_us:
            best_us, best = us, config
    lora_shrink_op.get_lora_op_configs = _ORIG_SHRINK_CFG
    return best, best_us, ok, failed, first_err


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
    ok, failed, first_err = 0, 0, None
    for config in space:
        lora_expand_op.get_lora_op_configs = lambda *a, _c=config, **k: _c
        fn = lambda: lora_expand(inputs, lora_b, output, *metadata, 0, add_inputs)
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if first_err is None and not _is_oom(exc):
                first_err = f"{type(exc).__name__}: {exc}"
            continue
        ok += 1
        us = time_fn(fn, warmup, iters)
        if us < best_us:
            best_us, best = us, config
    lora_expand_op.get_lora_op_configs = _ORIG_EXPAND_CFG
    return best, best_us, ok, failed, first_err


def nested(data: dict, path: list) -> dict:
    node = data
    for key in path:
        node = node.setdefault(str(key), {})
    return node


def merge(dst: dict, src: dict) -> None:
    for key, val in src.items():
        if isinstance(val, dict):
            merge(dst.setdefault(key, {}), val)
        else:
            dst[key] = val


def ranks_from_dir(path: str) -> list[int]:
    ranks = set()
    for cfg in Path(path).glob("*/adapter_config.json"):
        data = json.loads(cfg.read_text())
        if "r" in data:
            ranks.add(int(data["r"]))
    return sorted(ranks)


def run_jobs(gpu_id: int, jobs: list, params: dict):
    torch.cuda.init()
    torch.cuda.set_device(gpu_id)
    dtype = DTYPE_MAP[params["dtype"]]
    lds = params["lds_bytes"]

    shrink_space_list = [c for c in shrink_space(params["quick"]) if lds_bytes_shrink(c) <= lds]
    expand_space_list = [c for c in expand_space(params["quick"]) if lds_bytes_expand(c) <= lds]

    shrink_data: dict = {}
    expand_data = {True: {}, False: {}}
    stats = {
        "shrink": {"ok": 0, "failed": 0, "err": None, "us": []},
        "expand": {"ok": 0, "failed": 0, "err": None, "us": []},
    }

    for kind, hidden, rank, num_slices, m, add_inputs in tqdm(
        jobs, desc=f"GPU {gpu_id}", leave=False
    ):
        if kind == "shrink":
            cfg, us, ok, failed, err = tune_shrink(
                hidden, rank, num_slices, m, params["max_loras"], dtype, "cuda",
                shrink_space_list, params["warmup"], params["iters"])
            stats["shrink"]["ok"] += ok
            stats["shrink"]["failed"] += failed
            stats["shrink"]["err"] = stats["shrink"]["err"] or err
            if cfg is not None:
                stats["shrink"]["us"].append(us)
            if cfg is not None:
                nested(shrink_data, [params["max_loras"], num_slices, m, hidden, rank]).update(cfg)
        else:
            cfg, us, ok, failed, err = tune_expand(
                hidden, rank, num_slices, m, params["max_loras"], add_inputs, dtype,
                "cuda", expand_space_list, params["warmup"], params["iters"])
            stats["expand"]["ok"] += ok
            stats["expand"]["failed"] += failed
            stats["expand"]["err"] = stats["expand"]["err"] or err
            if cfg is not None:
                stats["expand"]["us"].append(us)
            if cfg is not None:
                nested(expand_data[add_inputs], [params["max_loras"], num_slices, m, rank, hidden]).update(cfg)

    return shrink_data, expand_data[True], expand_data[False], stats


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
    parser.add_argument(
        "--prefill",
        action="store_true",
        help="Also tune prefill token counts: extends --batch-sizes with a "
        "spread up to --prefill-max (chunked-prefill m can reach "
        "max_num_batched_tokens).",
    )
    parser.add_argument(
        "--prefill-max",
        type=int,
        default=2496,
        help="Upper prefill token count to tune (default: the server's "
        "max_num_batched_tokens).",
    )
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced candidate grid (fewer Triton compiles) at the "
        "risk of missing the fastest tile. Omit for a best-effort search.",
    )
    parser.add_argument(
        "--lds-kb",
        type=int,
        default=64,
        help="Per-CU LDS budget in KiB used to prune candidate tiles (R9700: 64).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPUs to shard across (default: all visible).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one config on one key per op on GPU 0, print, and exit.",
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
    if args.prefill:
        spread = [m for m in (64, 128, 256, 512, 1024, 1536, 2048)
                  if m < args.prefill_max]
        spread.append(args.prefill_max)
        args.batch_sizes = sorted(set(args.batch_sizes) | set(spread))

    torch.cuda.init()
    torch.accelerator.set_device_index(0)
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    lds_bytes = args.lds_kb * 1024

    shrink_n = len([c for c in shrink_space(args.quick) if lds_bytes_shrink(c) <= lds_bytes])
    expand_n = len([c for c in expand_space(args.quick) if lds_bytes_expand(c) <= lds_bytes])
    print(f"[tune_lora] device={gpu} dtype={args.dtype} lds={args.lds_kb}KiB")
    print(f"[tune_lora] hidden={args.hidden_sizes} ranks={args.lora_ranks} "
          f"slices={args.num_slices} max_loras={args.max_loras} m={args.batch_sizes}")
    print(f"[tune_lora] {shrink_n} shrink + {expand_n} expand candidate configs "
          f"fit the LDS budget (larger tiles pruned)")
    print(f"[tune_lora] output_dir={args.output_dir}")

    if args.smoke:
        h, r = args.hidden_sizes[0], args.lora_ranks[0]
        print("[smoke] running one shrink + one expand config on GPU 0...")
        cfg, us, ok, failed, err = tune_shrink(
            h, r, 1, 1, args.max_loras, DTYPE_MAP[args.dtype], "cuda",
            list(shrink_space(True))[:1], 1, 1)
        print(f"[smoke] shrink ok={ok} failed={failed} best={us:.1f}us -> {cfg} {err or ''}")
        cfg, us, ok, failed, err = tune_expand(
            h, r, 1, 1, args.max_loras, True, DTYPE_MAP[args.dtype], "cuda",
            list(expand_space(True))[:1], 1, 1)
        print(f"[smoke] expand ok={ok} failed={failed} best={us:.1f}us -> {cfg} {err or ''}")
        return

    jobs = []
    for hidden in args.hidden_sizes:
        for rank in args.lora_ranks:
            for num_slices in args.num_slices:
                for m in args.batch_sizes:
                    jobs.append(("shrink", hidden, rank, num_slices, m, None))
                    jobs.append(("expand", hidden, rank, num_slices, m, True))
                    jobs.append(("expand", hidden, rank, num_slices, m, False))

    num_gpus = args.num_gpus or torch.cuda.device_count()
    num_gpus = max(1, min(num_gpus, torch.cuda.device_count()))
    shards = [jobs[i::num_gpus] for i in range(num_gpus)]
    print(f"[tune_lora] {len(jobs)} jobs across {num_gpus} GPUs")
    for i, shard in enumerate(shards):
        print(f"[tune_lora]   GPU {i}: {len(shard)} jobs")

    params = {
        "dtype": args.dtype,
        "max_loras": args.max_loras,
        "warmup": args.warmup,
        "iters": args.iters,
        "quick": args.quick,
        "lds_bytes": lds_bytes,
    }

    shrink_data: dict = {}
    expand_true: dict = {}
    expand_false: dict = {}
    total = {"shrink": {"ok": 0, "failed": 0, "err": None, "us": []},
             "expand": {"ok": 0, "failed": 0, "err": None, "us": []}}

    if num_gpus == 1:
        results = [run_jobs(0, shards[0], params)]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_gpus) as pool:
            results = pool.starmap(
                run_jobs, [(i, shards[i], params) for i in range(num_gpus)]
            )

    for s, et, ef, stats in results:
        merge(shrink_data, s)
        merge(expand_true, et)
        merge(expand_false, ef)
        for tag, st in stats.items():
            total[tag]["ok"] += st["ok"]
            total[tag]["failed"] += st["failed"]
            total[tag]["err"] = total[tag]["err"] or st["err"]
            total[tag]["us"].extend(st["us"])

    for tag, st in total.items():
        us = sorted(st["us"])
        med = us[len(us) // 2] if us else 0.0
        lo = us[0] if us else 0.0
        print(f"[tune_lora] {tag}: {st['ok']} evals OK, {st['failed']} skipped "
              f"(OOM/compile); best {lo:.1f}us, median {med:.1f}us over "
              f"{len(us)} jobs")
        if st["ok"] == 0:
            print(f"[tune_lora] WARNING no {tag} config compiled successfully"
                  f"{'; first error: ' + st['err'] if st['err'] else ''}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (
        (f"{gpu}_SHRINK.json", shrink_data),
        (f"{gpu}_EXPAND_TRUE.json", expand_true),
        (f"{gpu}_EXPAND_FALSE.json", expand_false),
    ):
        path = out / name
        path.write_text(json.dumps(data, indent=4) + "\n")
        print(f"[tune_lora] wrote {path}")


if __name__ == "__main__":
    main()
