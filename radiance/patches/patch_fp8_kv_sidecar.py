#!/usr/bin/env python3
"""Add an opt-in, model-bound FP8 KV-scale sidecar to DefaultModelLoader."""

import sysconfig
from pathlib import Path

from _patchlib import apply


LIB = Path(sysconfig.get_paths()["purelib"])
F = LIB / "vllm/model_executor/model_loader/default_loader.py"

ANCHOR = '''        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
'''

NEW = '''        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

        # Radiance FP8-KV calibration is a separate immutable artifact rather
        # than an in-place checkpoint mutation. Load it last so its scalar
        # q/k/v/prob scales replace absent or placeholder checkpoint values.
        scale_sidecar = os.environ.get("RADIANCE_FP8_KV_SCALES", "").strip()
        if scale_sidecar:
            from radiance_kv_calibration import iter_scale_sidecar

            verify = os.environ.get("RADIANCE_FP8_KV_SCALES_VERIFY", "1") != "0"
            logger.info_once(
                "Loading Radiance FP8-KV calibration sidecar %s (verify=%s)",
                scale_sidecar,
                verify,
            )
            yield from iter_scale_sidecar(
                scale_sidecar,
                source_model=model_config.model if verify else None,
                verify_manifest=verify,
            )

    def download_model(self, model_config: ModelConfig) -> None:
'''

SENTINEL = "Loading Radiance FP8-KV calibration sidecar"


def main() -> None:
    apply(F, ANCHOR, NEW, SENTINEL, "fp8-kv: immutable calibrated-scale sidecar")


if __name__ == "__main__":
    main()
