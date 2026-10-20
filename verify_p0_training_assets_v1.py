"""Read-only identity and parameter-budget check for Protocol-V2 P0 training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "DehazeFormer-main"))

from crossmodal_dehazeformer_residual_v2 import (  # noqa: E402
    CrossModalDehazeFormerResidualV2,
    load_dehazeformer_base_weights,
)


EXPECTED_BASE_PARAMETERS = 2_517_612
EXPECTED_TRAINABLE_PARAMETERS = 47_255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=Path, required=True)
    args = parser.parse_args()

    model = CrossModalDehazeFormerResidualV2(
        backbone="b",
        init_gamma=0.01,
        ablation="wo_ir_physics",
        wo_gate=True,
        use_attnres=False,
        refiner_width=25,
        refiner_blocks=4,
        adapter_enabled=True,
        adapter_variant="plain",
    )
    missing, unexpected = load_dehazeformer_base_weights(
        model, args.pretrained, strict=True
    )
    if missing or unexpected:
        raise RuntimeError("Strict base load failed")
    model.freeze_base()

    base_parameters = sum(parameter.numel() for parameter in model.base.parameters())
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    base_trainable = sum(
        parameter.numel()
        for parameter in model.base.parameters()
        if parameter.requires_grad
    )
    if base_parameters != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(
            "Base parameter mismatch: {} != {}".format(
                base_parameters, EXPECTED_BASE_PARAMETERS
            )
        )
    if trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            "P0 parameter mismatch: {} != {}".format(
                trainable_parameters, EXPECTED_TRAINABLE_PARAMETERS
            )
        )
    if base_trainable != 0:
        raise RuntimeError("P0 backbone is not fully frozen")

    selection_hash = hashlib.sha256("\n".join(trainable).encode("utf-8")).hexdigest()
    print(
        json.dumps(
            {
                "status": "P0_TRAINING_ASSETS_V1_PASS",
                "base_parameters": base_parameters,
                "base_trainable": base_trainable,
                "adapter_trainable": trainable_parameters,
                "trainable_tensor_count": len(trainable),
                "trainable_names_sha256": selection_hash,
                "strict_base_load": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
