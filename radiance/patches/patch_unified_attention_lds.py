#!/usr/bin/env python3
"""Keep both RDNA4 attention paths inside 64 KiB after AITER's config-API migration.

AITER 0.1.26 moved the selectors into a shared JSON configuration reader.
Apply the correctness clamp there, independently of the optional runtime tune.

The reader can return a *partial* config: some `attn_3d` shapes come back with
only the launch keys and no `TILE_SIZE` (the caller keeps its own default), so
the clamp must skip those rather than index a missing key. The first cut of
this patch indexed `config["TILE_SIZE"]` unconditionally, which crashed graph
capture with `KeyError: 'TILE_SIZE'`; this script repairs such an install when
it finds the old block.
"""
import ast
import sysconfig
from pathlib import Path

F = (
    Path(sysconfig.get_paths()["purelib"])
    / "aiter/ops/triton/utils/unified_attention_utils.py"
)

OLD = "    return compute_segment_params(copy.deepcopy(config), params)\n"

HEAD = (
    "    config = compute_segment_params(copy.deepcopy(config), params)\n"
    "    # RADIANCE LDS fit: the configuration reader serves both attention paths.\n"
)

# What the previous version of this patch installed. Kept so an already-patched
# AITER (the usual case after a bootstrap run) is repaired, not silently left
# broken: the sentinel below is only absent until this rewrite lands.
BUGGY = HEAD + (
    '    if ((arch or arch_info.get_arch()) in ("gfx1200", "gfx1201")\n'
    '            and backend == "triton" and op in ("attn_2d", "attn_3d")):\n'
    '        tile = config["TILE_SIZE"]\n'
    '        stages = config.get("num_stages", 1)\n'
    "        width = triton.next_power_of_2(max(params.head_size, params.head_size_v))\n"
    "        element_bytes = params.kv_cache_dtype.itemsize\n"
    "        while stages > 1 and tile * width * element_bytes * stages + 256 > 65536:\n"
    "            stages -= 1\n"
    "        while tile > 16 and tile * width * element_bytes * stages + 256 > 65536:\n"
    "            tile //= 2\n"
    "        if tile * width * element_bytes * stages + 256 > 65536:\n"
    '            raise ValueError("RADIANCE: attention configuration exceeds RDNA4 LDS")\n'
    '        config["TILE_SIZE"] = tile\n'
    '        config["num_stages"] = stages\n'
    "    return config\n"
)

NEW = HEAD + (
    "    # AITER 0.1.26 returns a partial config for some attn_3d shapes (no\n"
    "    # TILE_SIZE); the clamp only applies to configs that carry the tile.\n"
    '    if ("TILE_SIZE" in config\n'
    '            and (arch or arch_info.get_arch()) in ("gfx1200", "gfx1201")\n'
    '            and backend == "triton" and op in ("attn_2d", "attn_3d")):\n'
    '        kv_dtype = getattr(params, "kv_cache_dtype", None)\n'
    '        head_size = int(getattr(params, "head_size", 0) or 0)\n'
    '        head_size_v = int(getattr(params, "head_size_v", head_size) or head_size)\n'
    '        if kv_dtype is not None and hasattr(kv_dtype, "itemsize") and head_size:\n'
    '            tile = int(config["TILE_SIZE"])\n'
    '            stages = int(config.get("num_stages", 1) or 1)\n'
    "            width = triton.next_power_of_2(max(head_size, head_size_v))\n"
    "            element_bytes = kv_dtype.itemsize\n"
    "            while (\n"
    "                stages > 1\n"
    "                and tile * width * element_bytes * stages + 256 > 65536\n"
    "            ):\n"
    "                stages -= 1\n"
    "            while (\n"
    "                tile > 16\n"
    "                and tile * width * element_bytes * stages + 256 > 65536\n"
    "            ):\n"
    "                tile //= 2\n"
    "            if tile * width * element_bytes * stages + 256 > 65536:\n"
    '                raise ValueError("RADIANCE: attention configuration exceeds RDNA4 LDS")\n'
    '            config["TILE_SIZE"] = tile\n'
    '            config["num_stages"] = stages\n'
    "    return config\n"
)

SENTINEL = "# AITER 0.1.26 returns a partial config"


def main():
    if not F.exists():
        raise SystemExit(f"  FAIL  RDNA4 shared LDS clamp: {F} missing")
    s = F.read_text()
    if SENTINEL in s:
        print("  NOOP  RDNA4 shared LDS clamp already applied")
        return
    if BUGGY in s:
        s = s.replace(BUGGY, NEW, 1)
        ast.parse(s)
        F.write_text(s)
        print("  OK    RDNA4 shared LDS clamp repaired (guard partial configs)")
        return
    n = s.count(OLD)
    if n != 1:
        raise SystemExit(
            f"  FAIL  RDNA4 shared LDS clamp: anchor matched {n}x, expected 1 ({F})"
        )
    s = s.replace(OLD, NEW, 1)
    ast.parse(s)
    F.write_text(s)
    print("  OK    RDNA4 shared LDS clamp")


if __name__ == "__main__":
    main()
