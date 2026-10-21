"""Count the frozen hash assignment among original nighttime training images."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from data.dataset_crossmodal import CrossModalUAVDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["FT_NIGHT_VD_ARM"] = "default"
    data = CrossModalUAVDataset(str(args.train_root), size=256)
    if len(data) != 12000:
        raise ValueError(f"Expected 12000 training sources, found {len(data)}")
    names = []
    for raw_name in data.train_list:
        stem = os.path.splitext(raw_name)[0]
        gt = data.load_img(data.find_image_path(data.rgb_dir, stem))
        if np.mean(gt) * 255.0 < 70:
            names.append(stem)
    counts = {str(seed): sum(int(hashlib.sha256(f"{seed}|{stem}".encode()).hexdigest(), 16) % 5 == 0
                             for stem in names) for seed in (3407, 2026, 8888)}
    report = {"train_count": len(data), "night_count": len(names), "hash_modulus": 5,
              "selected_night_count_by_seed": counts,
              "selected_night_fraction_by_seed": {seed: count / len(names) for seed, count in counts.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NIGHT_SELECTION_AUDIT_PASS", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
