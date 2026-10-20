from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
VALID_SEVERITIES = {"light", "dense", "very_dense"}
EXPECTED_SEVERITY_COUNTS = {"light": 384, "dense": 230, "very_dense": 154}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_image(folder: Path, stem: str, allow_ir_suffix: bool = False) -> Path | None:
    stems = (stem, f"{stem}_ir") if allow_ir_suffix else (stem,)
    for candidate_stem in stems:
        for extension in IMAGE_EXTENSIONS:
            for candidate_extension in (extension, extension.upper()):
                candidate = folder / f"{candidate_stem}{candidate_extension}"
                if candidate.is_file():
                    return candidate
    return None


def perlin_noise(width: int, height: int, scale: float = 5.0) -> np.ndarray:
    grid_width = int(scale)
    grid_height = int(scale)
    if grid_width < 1 or grid_height < 1:
        raise ValueError(f"Perlin scale must be >= 1, got {scale}")

    linear_x = np.linspace(0, grid_width, width, endpoint=False)
    linear_y = np.linspace(0, grid_height, height, endpoint=False)
    x_grid, y_grid = np.meshgrid(linear_x, linear_y)

    def fade(value: np.ndarray) -> np.ndarray:
        return 6 * value**5 - 15 * value**4 + 10 * value**3

    random_grid = np.random.rand(grid_height + 1, grid_width + 1)
    fractional_x = x_grid % 1
    fractional_y = y_grid % 1
    integer_x = x_grid.astype(int) % grid_width
    integer_y = y_grid.astype(int) % grid_height

    upper = (
        random_grid[integer_y, integer_x] * (1 - fractional_x)
        + random_grid[integer_y, integer_x + 1] * fractional_x
    )
    lower = (
        random_grid[integer_y + 1, integer_x] * (1 - fractional_x)
        + random_grid[integer_y + 1, integer_x + 1] * fractional_x
    )
    noise = upper * (1 - fade(fractional_y)) + lower * fade(fractional_y)
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)


def fbm_noise(
    width: int,
    height: int,
    octaves: int = 4,
    base_scale: float = 2.0,
) -> np.ndarray:
    noise = np.zeros((height, width), dtype=np.float32)
    amplitude = 0.5
    frequency = base_scale
    for _ in range(octaves):
        noise += amplitude * perlin_noise(width, height, scale=frequency)
        amplitude *= 0.5
        frequency *= 2
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)


def add_synthetic_smoke_controlled(
    image: np.ndarray,
    severity: str,
    is_night: bool,
    synthesis_seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if severity not in VALID_SEVERITIES:
        raise ValueError(f"Unknown severity: {severity}")

    random.seed(synthesis_seed)
    np.random.seed(synthesis_seed)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    height, width = image.shape[:2]
    texture_scale = random.uniform(2.0, 4.5)
    density_scale = random.uniform(1.0, 2.0)

    base_texture = fbm_noise(
        width,
        height,
        octaves=5,
        base_scale=texture_scale,
    )
    displacement_x = fbm_noise(width, height, octaves=2, base_scale=10) * 40 - 20
    displacement_y = fbm_noise(width, height, octaves=2, base_scale=10) * 40 - 20
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    map_x = np.clip(x_grid + displacement_x, 0, width - 1).astype(np.float32)
    map_y = np.clip(y_grid + displacement_y, 0, height - 1).astype(np.float32)
    smoke_texture = cv2.remap(base_texture, map_x, map_y, cv2.INTER_LINEAR)
    global_density = fbm_noise(
        width,
        height,
        octaves=2,
        base_scale=density_scale,
    )

    if severity == "light":
        constant_bias = random.uniform(0.4, 0.6)
        transmission_power = random.uniform(1.0, 1.3)
    elif severity == "dense":
        constant_bias = random.uniform(0.15, 0.3)
        transmission_power = random.uniform(1.8, 2.5)
    else:
        constant_bias = random.uniform(0.0, 0.05)
        transmission_power = random.uniform(2.8, 4.0)

    base_transmission = (1.0 - global_density * 0.5) * (
        constant_bias + (1.0 - constant_bias) * smoke_texture
    )
    transmission = np.power(base_transmission, transmission_power)
    transmission = np.clip(transmission, 0.01, 1.0)[..., None]

    is_black_smoke = random.random() < 0.25
    if is_night:
        atmospheric_light = (
            random.uniform(0.15, 0.35)
            if is_black_smoke
            else random.uniform(0.5, 0.8)
        )
    else:
        atmospheric_light = (
            random.uniform(0.1, 0.3)
            if is_black_smoke
            else random.uniform(0.8, 0.95)
        )

    atmospheric_base = np.array(
        [atmospheric_light, atmospheric_light, atmospheric_light],
        dtype=np.float32,
    )
    light_variance = cv2.resize(perlin_noise(8, 8, 2), (width, height))
    light_variance = 0.9 + 0.2 * light_variance
    atmospheric_map = atmospheric_base[None, None, :] * light_variance[:, :, None]
    degraded = image * transmission + atmospheric_map * (1 - transmission)

    if is_night:
        grayscale = np.mean(image, axis=2)
        bright_mask = np.clip((grayscale - 0.3) * 2, 0, 1)
        bright_boost = 1 + 0.4 * bright_mask[..., None] * (1 - transmission)
        degraded = degraded * bright_boost

    degraded = np.clip(degraded, 0.0, 1.0)
    metadata: dict[str, object] = {
        "synthesis_seed": synthesis_seed,
        "severity": severity,
        "is_night": is_night,
        "texture_scale": texture_scale,
        "density_scale": density_scale,
        "constant_bias": constant_bias,
        "transmission_power": transmission_power,
        "is_black_smoke": is_black_smoke,
        "atmospheric_light": atmospheric_light,
        "transmission_min": float(transmission.min()),
        "transmission_mean": float(transmission.mean()),
        "transmission_max": float(transmission.max()),
    }
    # Keep the original generator's uint8 conversion exactly (truncate, do not round).
    return (degraded * 255.0).astype(np.uint8), metadata


def load_protocol(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    required_columns = {"source_stem", "severity", "synthesis_seed"}
    if not rows:
        raise ValueError(f"Protocol is empty: {path}")
    if not required_columns.issubset(rows[0]):
        raise ValueError(
            f"Protocol must contain columns {sorted(required_columns)}, got {sorted(rows[0])}"
        )

    normalized: list[dict[str, object]] = []
    source_stems: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        source_stem = str(row["source_stem"]).strip()
        severity = str(row["severity"]).strip()
        synthesis_seed = int(str(row["synthesis_seed"]).strip())

        if not source_stem:
            raise ValueError(f"Empty source_stem at CSV row {row_number}")
        if source_stem in source_stems:
            raise ValueError(f"Duplicate source_stem in protocol: {source_stem}")
        if severity not in VALID_SEVERITIES:
            raise ValueError(f"Invalid severity at CSV row {row_number}: {severity}")
        if not 0 <= synthesis_seed <= 0xFFFFFFFF:
            raise ValueError(f"Seed outside uint32 range at CSV row {row_number}")

        source_stems.add(source_stem)
        normalized.append(
            {
                "source_stem": source_stem,
                "severity": severity,
                "synthesis_seed": synthesis_seed,
            }
        )

    severity_counts = Counter(str(row["severity"]) for row in normalized)
    if len(normalized) != 768:
        raise ValueError(f"Expected 768 protocol rows, got {len(normalized)}")
    if dict(severity_counts) != EXPECTED_SEVERITY_COUNTS:
        raise ValueError(
            f"Expected severity counts {EXPECTED_SEVERITY_COUNTS}, got {dict(severity_counts)}"
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the frozen F-ScatterRes V2 synthetic-smoke test set."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("data/dataset/uav/test"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("data/split_v2_manifests/final_test_protocol.csv"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/dataset/uav/synthetic_smoke_test_v2"),
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Validate the protocol and source pairs without writing images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    protocol_path = args.protocol.resolve()
    output_root = args.output_root.resolve()
    gt_dir = input_root / "gt"
    ir_dir = input_root / "ir"

    protocol_rows = load_protocol(protocol_path)
    resolved_rows: list[dict[str, object]] = []
    for row in protocol_rows:
        source_stem = str(row["source_stem"])
        gt_path = find_image(gt_dir, source_stem)
        ir_path = find_image(ir_dir, source_stem, allow_ir_suffix=True)
        if gt_path is None:
            raise FileNotFoundError(f"Missing GT source: {source_stem} in {gt_dir}")
        if ir_path is None:
            raise FileNotFoundError(f"Missing IR source: {source_stem} in {ir_dir}")
        resolved_rows.append({**row, "gt_path": gt_path, "ir_path": ir_path})

    print(f"Protocol: {protocol_path}")
    print(f"Protocol SHA256: {sha256_file(protocol_path)}")
    print(f"Resolved paired sources: {len(resolved_rows)}")
    print(f"Severity counts: {dict(Counter(str(row['severity']) for row in resolved_rows))}")

    if args.preflight_only:
        print("Preflight passed. No output files were written.")
        return

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Use a new directory for each run."
        )

    lq_output = output_root / "lq"
    gt_output = output_root / "gt"
    ir_output = output_root / "ir"
    for folder in (lq_output, gt_output, ir_output):
        folder.mkdir(parents=True, exist_ok=True)

    partial_manifest = output_root / "generation_manifest.jsonl.partial"
    final_manifest = output_root / "generation_manifest.jsonl"
    generated_records: list[dict[str, object]] = []

    with partial_manifest.open("w", encoding="utf-8") as manifest_stream:
        for index, row in enumerate(tqdm(resolved_rows, desc="Generating V2 test set")):
            source_stem = str(row["source_stem"])
            severity = str(row["severity"])
            synthesis_seed = int(row["synthesis_seed"])
            gt_path = Path(row["gt_path"])
            ir_path = Path(row["ir_path"])

            gt_bgr = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
            if gt_bgr is None:
                raise FileNotFoundError(f"Failed to decode GT: {gt_path}")
            gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
            is_night = float(np.mean(gt_rgb)) < 70.0
            smoke_rgb, synthesis_metadata = add_synthetic_smoke_controlled(
                gt_rgb,
                severity=severity,
                is_night=is_night,
                synthesis_seed=synthesis_seed,
            )

            time_tag = "night" if is_night else "day"
            output_stem = f"{index:04d}_{source_stem}_{time_tag}_{severity}"
            lq_path = lq_output / f"{output_stem}.png"
            gt_copy = gt_output / f"{output_stem}{gt_path.suffix.lower()}"
            ir_copy = ir_output / f"{output_stem}{ir_path.suffix.lower()}"

            write_ok = cv2.imwrite(
                str(lq_path),
                cv2.cvtColor(smoke_rgb, cv2.COLOR_RGB2BGR),
            )
            if not write_ok:
                raise OSError(f"Failed to write LQ image: {lq_path}")
            shutil.copy2(gt_path, gt_copy)
            shutil.copy2(ir_path, ir_copy)

            record: dict[str, object] = {
                "index": index,
                "source_stem": source_stem,
                "output_stem": output_stem,
                "source_gt": str(gt_path),
                "source_ir": str(ir_path),
                "source_gt_sha256": sha256_file(gt_path),
                "source_ir_sha256": sha256_file(ir_path),
                "lq_sha256": sha256_file(lq_path),
                "gt_copy_sha256": sha256_file(gt_copy),
                "ir_copy_sha256": sha256_file(ir_copy),
                **synthesis_metadata,
            }
            manifest_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_stream.flush()
            generated_records.append(record)

    partial_manifest.replace(final_manifest)
    summary = {
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "sample_count": len(generated_records),
        "severity_counts": dict(
            Counter(str(row["severity"]) for row in generated_records)
        ),
        "manifest_sha256": sha256_file(final_manifest),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
    }
    summary_path = output_root / "generation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Generated samples: {len(generated_records)}")
    print(f"Output: {output_root}")
    print(f"Manifest: {final_manifest}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
