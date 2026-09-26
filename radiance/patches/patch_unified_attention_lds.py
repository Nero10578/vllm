#!/usr/bin/env python3
"""LDS-fit overlay for AITER's unified attention on RDNA (gfx1201 / R9700).

The attention kernels stage a TILE_SIZE x next_pow2(head_size) K/V tile num_stages deep in shared
memory at the KV-cache element size (+~256 B). AITER's gfx1201 config lets TILE_SIZE reach 64 with
2 pipeline stages, which exceeds the R9700's 64 KiB LDS for some shapes and makes Triton raise
OutOfResources at cudagraph capture:

    head_size 256, 2-byte KV (bf16/fp16): 64*256*2*2 + 256 = 65792
    head_size 512, fp8 KV              : 64*512*1*2 + 256 = 65792

This patch clamps the staged tile (pipeline depth first, then tile width) until it fits 64 KiB, at
whichever launch site the installed aiter uses:

  * aiter <= 0.1.20: string replacements on the `select_3d_config` / `select_2d_config` selectors.
  * aiter >= 0.1.21: the selectors were replaced by `unified_attention_utils.get_unified_attention_config`;
    here we inject a `_radiance_fit_lds` helper and clamp in `_unified_attention_2d_triton` and
    `_unified_attention_3d_triton` (where both the tile and the stage count are in scope).

The clamp is a hard requirement (correctness), so it lives in source; the runtime RADIANCE_ATTN_TUNE
hook stays a separate, disableable tune. Registered/chunked/shuffled caches are left untouched (their
tile must equal the page).
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
F = SP / "aiter/ops/triton/attention/unified_attention.py"

# ---------------------------------------------------------------------------
# Old layout (aiter <= 0.1.20): edit the selector functions directly.
# ---------------------------------------------------------------------------


def _fit(tag, stages_var):
    return (
        f"    # --- RADIANCE LDS fit ({tag}): shrink the staged K/V tile into the R9700's 64 KiB LDS ---\n"
        "    _rad_el = 2 if kv_cache_dtype in (torch.bfloat16, torch.float16) else 1\n"
        "    _rad_hs = triton.next_power_of_2(head_size)\n"
        f"    while {stages_var} > 1 and TILE_SIZE * _rad_hs * _rad_el * {stages_var} + 256 > 65536:\n"
        f"        {stages_var} -= 1\n"
        f"    while TILE_SIZE > 16 and TILE_SIZE * _rad_hs * _rad_el * {stages_var} + 256 > 65536:\n"
        "        TILE_SIZE //= 2\n"
        "\n"
    )


A3 = (
    "    if NUM_BLOCKS_GATHER_PER_TILE > 1:\n"
    "        # force gather mode\n"
)
A2 = (
    "    return {\n"
    '        "BLOCK_M": BLOCK_M,\n'
    '        "BLOCK_Q": BLOCK_Q,\n'
)
ANCHOR = (
    "        elif q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:\n"
    "            TILE_SIZE = max(32, TILE_SIZE)\n"
)
INSERT = (
    "        elif kv_cache_dtype in (torch.bfloat16, torch.float16):\n"
    "            # --- RADIANCE 2-byte (bf16/fp16, incl. --kv-cache-dtype auto) KV, gfx1201 ---\n"
    "            TILE_SIZE = 16\n"
    "            attn_warps = 4\n"
    "            attn_stages = 2\n"
    "            waves_per_eu = 2\n"
    "            reduce_num_warps = 4\n"
)

# ---------------------------------------------------------------------------
# New layout (aiter >= 0.1.21): inject the helper and clamp at the launch sites.
# ---------------------------------------------------------------------------
HELPER_ANCHOR = "def _gfx950_gluon_supported(params: _UAParams):\n"
HELPER_INSERT = '''_LDS_BUDGET_BYTES = 64 * 1024


def _radiance_fit_lds(tile, stages, head_size, elem_size, budget=_LDS_BUDGET_BYTES):
    """Clamp a staged K/V tile into the R9700's 64 KiB LDS.

    The kernel stages TILE_SIZE x next_pow2(head_size) at elem_size bytes, num_stages
    deep, plus ~256 B. Reduce the pipeline depth first, then the tile width, until the
    tile fits. Returns (tile, stages); a tile that already fits is unchanged.
    """
    head_padded = triton.next_power_of_2(int(head_size))
    elem = 2 if elem_size > 1 else 1
    tile = int(tile)
    stages = max(1, int(stages))
    while stages > 1 and tile * head_padded * elem * stages + 256 > budget:
        stages -= 1
    while tile > 16 and tile * head_padded * elem * stages + 256 > budget:
        tile //= 2
    return tile, stages


'''
NEW_HELPER_SENTINEL = "def _radiance_fit_lds("

P2D_ANCHOR = (
    '    assert config["BLOCK_Q"] >= 1\n'
    '    if params.shuffled_kv_cache:\n'
    '        config["TILE_SIZE"] = params.block_size\n'
    '    if params.all_decode:\n'
)
P2D_INSERT = (
    '    assert config["BLOCK_Q"] >= 1\n'
    '    if params.shuffled_kv_cache:\n'
    '        config["TILE_SIZE"] = params.block_size\n'
    '    if DEVICE_ARCH.startswith("gfx12") and not params.shuffled_kv_cache:\n'
    '        config["TILE_SIZE"], config["num_stages"] = _radiance_fit_lds(\n'
    '            config["TILE_SIZE"], config.get("num_stages", 1),\n'
    '            params.head_size, params.k.element_size(),\n'
    '        )\n'
    '    if params.all_decode:\n'
)
NEW_P2D_SENTINEL = 'config["TILE_SIZE"], config["num_stages"] = _radiance_fit_lds'

P3D_ANCHOR = (
    '    config = get_unified_attention_config("attn_3d", params, backend="triton")\n'
    '    config["BLOCK_M"] = max(\n'
    '        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)\n'
    '    )\n'
    '    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv\n'
    '    assert config["BLOCK_Q"] >= 1\n'
    '\n'
    '    if params.all_decode:\n'
)
P3D_INSERT = (
    '    config = get_unified_attention_config("attn_3d", params, backend="triton")\n'
    '    config["BLOCK_M"] = max(\n'
    '        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)\n'
    '    )\n'
    '    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv\n'
    '    assert config["BLOCK_Q"] >= 1\n'
    '    if DEVICE_ARCH.startswith("gfx12") and not params.shuffled_kv_cache:\n'
    '        TILE_SIZE, config["num_stages"] = _radiance_fit_lds(\n'
    '            TILE_SIZE, config.get("num_stages", 1),\n'
    '            params.head_size, params.k.element_size(),\n'
    '        )\n'
    '\n'
    '    if params.all_decode:\n'
)
NEW_P3D_SENTINEL = 'TILE_SIZE, config["num_stages"] = _radiance_fit_lds'


def _main_old() -> None:
    apply(F, A3, _fit("3D", "attn_stages") + A3, "RADIANCE LDS fit (3D)", "unified_attention LDS fit (3D)")
    apply(F, A2, _fit("2D", "num_stages_2d") + A2, "RADIANCE LDS fit (2D)", "unified_attention LDS fit (2D)")
    apply(F, ANCHOR, ANCHOR + INSERT, "RADIANCE 2-byte", "unified_attention bf16 3D-decode tune")


def _main_new() -> None:
    apply(F, HELPER_ANCHOR, HELPER_INSERT + HELPER_ANCHOR, NEW_HELPER_SENTINEL,
          "unified_attention LDS fit helper")
    apply(F, P2D_ANCHOR, P2D_INSERT, NEW_P2D_SENTINEL,
          "unified_attention LDS fit (2D)")
    apply(F, P3D_ANCHOR, P3D_INSERT, NEW_P3D_SENTINEL,
          "unified_attention LDS fit (3D)")


def main() -> None:
    if not F.exists():
        print(f"  SKIP  {F} not found (aiter not installed)")
        raise SystemExit(0)
    src = F.read_text()
    if "def select_3d_config" in src:
        _main_old()
        return
    if "get_unified_attention_config" in src and "_unified_attention_3d_triton" in src:
        _main_new()
        return
    print("  N/A   unrecognized aiter unified_attention layout; LDS-fit overlay not applied")
    raise SystemExit(0)


if __name__ == "__main__":
    main()