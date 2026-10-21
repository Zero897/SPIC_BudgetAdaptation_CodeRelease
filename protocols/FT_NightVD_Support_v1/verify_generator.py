"""Check paired smoke generation without starting training."""

import argparse
import hashlib
import importlib.util
import random
from pathlib import Path

import numpy as np


ORIGINAL_SHA256 = "5e8a7c975a3f90c54c393c53ab7528eacc70e5b95aff4b8d9009107456daa34d"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, expanded):
    random.seed(3407)
    np.random.seed(3407)
    image = np.full((32, 32, 3), 0.2, dtype=np.float32)
    output = module.add_synthetic_smoke(image, is_night=True, return_params=True,
                                        **({"support_expanded": expanded} if expanded is not None else {}))
    return output, random.getstate(), np.random.get_state()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original-dataset", type=Path, required=True)
    args = p.parse_args()
    assert hashlib.sha256(args.original_dataset.read_bytes()).hexdigest() == ORIGINAL_SHA256
    original = load(args.original_dataset, "original_dataset")
    patched = load(Path(__file__).parent / "source/data/dataset_crossmodal.py", "patched_dataset")
    old, old_py, old_np = run(original, None)
    default, default_py, default_np = run(patched, False)
    expanded, expanded_py, expanded_np = run(patched, True)
    assert all(np.array_equal(x, y) for x, y in zip(old, default))
    assert old_py == default_py == expanded_py
    assert all(np.array_equal(x, y) for x, y in zip(old_np, default_np))
    assert all(np.array_equal(x, y) for x, y in zip(old_np, expanded_np))
    assert not np.array_equal(default[0], expanded[0])
    print("GENERATOR_PAIRING_PASS: default bit-exact; expanded RNG stream unchanged")


if __name__ == "__main__":
    main()
