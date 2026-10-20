"""Offline self-check for the locked Restormer RGB-only final-test code."""

import hashlib
from pathlib import Path

import numpy as np
import torch

import analyze_post_ft_rgb_restormer_final_v1 as analysis
import eval_post_ft_rgb_restormer_final_v1 as evaluator
from post_ft_rgb_residual_v1 import build_model


ROOT = Path(__file__).resolve().parent
EXPECTED_SOURCE_HASHES = {
    "post_ft_rgb_residual_v1.py": "7ce05a16b62b462ee5b8cd129c45cdab4bc3d891abc16fa39a583dd3f773f952",
    "native_partial_ft_v1.py": "0fcbfaa74890f727b4b30ce7c0aaee0451bac3f8c6448bdd98117d9bf30de6e8",
    "restormer_arch_protocol_v2.py": "3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        actual = sha256_file(ROOT / name)
        if actual != expected:
            raise RuntimeError("Source hash mismatch: {} {}".format(name, actual))

    model = build_model("restormer", ROOT / "restormer_arch_protocol_v2.py", 0.01)
    counts = model.parameter_counts()
    if counts["base_total"] != evaluator.EXPECTED_BASE_PARAMETERS:
        raise RuntimeError("Restormer parameter count mismatch")
    if counts["adapter_total"] != evaluator.EXPECTED_ADAPTER_PARAMETERS:
        raise RuntimeError("Adapter parameter count mismatch")
    if counts["base_trainable"] != 0 or model.refiner.gate is not None:
        raise RuntimeError("The final model must use a frozen base and no gate")

    saved = {
        "model": {},
        "epoch": 6,
        "args": {
            "backbone": "restormer",
            "seed": 3407,
            "epochs": 15,
            "eval_every": 3,
            "size": 256,
            "init_gamma": 0.01,
        },
    }
    evaluator.validate_residual_checkpoint(saved, 3407)
    saved["epoch"] = 9
    try:
        evaluator.validate_residual_checkpoint(saved, 3407)
    except ValueError:
        pass
    else:
        raise RuntimeError("Wrong selected epoch was not rejected")

    image = torch.rand(2, 3, 32, 32)
    if not torch.allclose(evaluator.ssim_per_sample(image, image), torch.ones(2), atol=1e-5):
        raise RuntimeError("SSIM identity check failed")

    values = np.asarray([[0.1, 0.01, -0.01], [0.2, 0.02, -0.02]], dtype=np.float64)
    original_resamples = analysis.BOOTSTRAP_RESAMPLES
    analysis.BOOTSTRAP_RESAMPLES = 200
    first = analysis.bootstrap(values, np.random.default_rng(7))
    second = analysis.bootstrap(values, np.random.default_rng(7))
    analysis.BOOTSTRAP_RESAMPLES = original_resamples
    if first != second or first["psnr"]["mean"] <= 0 or first["lpips"]["mean"] >= 0:
        raise RuntimeError("Deterministic paired bootstrap check failed")

    print("POST_FT_RGB_RESTORMER_FINAL_V1_SELF_CHECK_PASS")


if __name__ == "__main__":
    main()
