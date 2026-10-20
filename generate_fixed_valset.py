#!/usr/bin/env python3
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from generate_fixed_testset import add_synthetic_smoke_controlled


MASTER_SEED = 20260923
EXPECTED_PAIRS = 1469
EXPECTED_GENERATOR_SHA256 = (
    "327b20d75352c831c9c7cf29530795fab7c8feb040499c7c916f9225af1a0f7d"
)

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_ROOT / "data/dataset/uav/val"
OUTPUT_ROOT = PROJECT_ROOT / "data/dataset/uav/synthetic_smoke_val_v2"
GENERATOR_PATH = PROJECT_ROOT / "generate_fixed_testset.py"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_map(directory):
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        if path.stem in result:
            raise RuntimeError(f"Duplicate stem in {directory}: {path.stem}")
        result[path.stem] = path
    return result


def sample_seed(source_stem, level):
    text = f"{MASTER_SEED}|{source_stem}|{level}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big")


def refuse_nonempty_output():
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {OUTPUT_ROOT}\n"
            "Refusing to overwrite it. Move the old directory aside before retrying."
        )


def main():
    if not GENERATOR_PATH.is_file():
        raise FileNotFoundError(GENERATOR_PATH)

    generator_sha256 = sha256_file(GENERATOR_PATH)
    if generator_sha256 != EXPECTED_GENERATOR_SHA256:
        raise RuntimeError(
            "generate_fixed_testset.py has changed.\n"
            f"Expected: {EXPECTED_GENERATOR_SHA256}\n"
            f"Actual:   {generator_sha256}"
        )

    gt_map = image_map(INPUT_ROOT / "gt")
    ir_map = image_map(INPUT_ROOT / "ir")
    if set(gt_map) != set(ir_map):
        raise RuntimeError(
            f"Unpaired validation files: "
            f"GT-only={len(set(gt_map) - set(ir_map))}, "
            f"IR-only={len(set(ir_map) - set(gt_map))}"
        )
    if len(gt_map) != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} validation pairs, found {len(gt_map)}"
        )

    refuse_nonempty_output()
    out_lq = OUTPUT_ROOT / "lq"
    out_gt = OUTPUT_ROOT / "gt"
    out_ir = OUTPUT_ROOT / "ir"
    for directory in (out_lq, out_gt, out_ir):
        directory.mkdir(parents=True, exist_ok=False)

    total = len(gt_map)
    num_light = int(total * 0.5)
    num_dense = int(total * 0.3)
    levels = (
        ["light"] * num_light
        + ["dense"] * num_dense
        + ["very_dense"] * (total - num_light - num_dense)
    )

    stems = sorted(gt_map)
    assignment_rng = random.Random(MASTER_SEED)
    assignment_rng.shuffle(stems)
    assignment_rng.shuffle(levels)

    cv2.setNumThreads(1)
    cv2.setRNGSeed(MASTER_SEED)
    records = []

    for index, (stem, level) in enumerate(zip(stems, levels)):
        gt_path = gt_map[stem]
        ir_path = ir_map[stem]
        image_bgr = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Cannot read GT image: {gt_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        brightness = float(np.mean(image_rgb))
        is_night = brightness < 70.0
        time_tag = "night" if is_night else "day"

        seed = sample_seed(stem, level)
        random.seed(seed)
        np.random.seed(seed)
        smoke_rgb = add_synthetic_smoke_controlled(
            image_rgb, level=level, is_night=is_night
        )
        smoke_bgr = cv2.cvtColor(smoke_rgb, cv2.COLOR_RGB2BGR)

        output_stem = f"{index:04d}_val_{stem}_{time_tag}_{level}"
        lq_path = out_lq / f"{output_stem}.png"
        gt_output_path = out_gt / f"{output_stem}{gt_path.suffix.lower()}"
        ir_output_path = out_ir / f"{output_stem}{ir_path.suffix.lower()}"

        if not cv2.imwrite(str(lq_path), smoke_bgr):
            raise RuntimeError(f"Failed to write: {lq_path}")
        shutil.copy2(gt_path, gt_output_path)
        shutil.copy2(ir_path, ir_output_path)

        records.append(
            {
                "index": index,
                "output_stem": output_stem,
                "source_stem": stem,
                "level": level,
                "is_night": is_night,
                "brightness": round(brightness, 6),
                "sample_seed": seed,
                "source_gt": str(gt_path.relative_to(PROJECT_ROOT)),
                "source_ir": str(ir_path.relative_to(PROJECT_ROOT)),
                "source_gt_sha256": sha256_file(gt_path),
                "source_ir_sha256": sha256_file(ir_path),
                "lq_sha256": sha256_file(lq_path),
            }
        )

        if (index + 1) % 100 == 0 or index + 1 == total:
            print(f"generated {index + 1}/{total}")

    manifest_path = OUTPUT_ROOT / "generation_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "purpose": "fixed synthetic-smoke validation set for checkpoint selection",
        "master_seed": MASTER_SEED,
        "source_pair_count": total,
        "generated_count": len(records),
        "unique_source_count": len({record["source_stem"] for record in records}),
        "severity_counts": dict(sorted(Counter(levels).items())),
        "day_night_counts": dict(
            sorted(Counter("night" if r["is_night"] else "day" for r in records).items())
        ),
        "generator_file": GENERATOR_PATH.name,
        "generator_sha256": generator_sha256,
        "wrapper_file": Path(__file__).name,
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_rule": "checkpoint selection only; never final-test reporting",
    }
    summary_path = OUTPUT_ROOT / "generation_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")
    print(f"summary:  {summary_path}")


if __name__ == "__main__":
    main()
