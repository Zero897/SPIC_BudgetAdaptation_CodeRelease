"""Paired three-seed analysis for the locked Restormer RGB-only final test."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEEDS = (3407, 2026, 8888)
METRICS = ("psnr", "ssim", "lpips")
GROUPS = ("all", "light", "dense", "very_dense", "day", "night")
EXPECTED_COUNT = 768
BOOTSTRAP_SEED = 20261015
BOOTSTRAP_RESAMPLES = 50_000
EXPECTED_MANIFEST_SHA256 = "4f1782e5d4c6e946d7819c5bb647d7f1a5f49355414b54a37cd935d5e4a3d0e2"
EXPECTED_BASE_SHA256 = "a272ba1c49254e9b59ad7eb3299a16e8c169ab3f939e31171716064ddd1a5e37"
EXPECTED_ARCHITECTURE_SHA256 = "3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a"
EXPECTED_SELECTED_EPOCH = {3407: 6, 2026: 6, 8888: 12}
EXPECTED_CHECKPOINT_SHA256 = {
    3407: "54c03b54309a7d7760c78fe6b3bd457afb6998bf720ab27f917e4d84555acadb",
    2026: "b436eb209015042fb338225465a20dda32f03322348009471e2c4a1317ec5e4a",
    8888: "d419c2212834f46c4cf03a41ac2b034eb4ad7630be2a8a627f9dddb6c864d944",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run(path, mode, seed):
    metrics = read_json(path / "metrics.json")
    rows = read_jsonl(path / "per_sample.jsonl")
    if metrics.get("protocol") != "post_ft_rgb_restormer_final_v1":
        raise RuntimeError("Protocol mismatch: {}".format(path))
    if metrics.get("mode") != mode or metrics.get("seed") != seed:
        raise RuntimeError("Run identity mismatch: {}".format(path))
    if metrics.get("sample_count") != EXPECTED_COUNT or len(rows) != EXPECTED_COUNT:
        raise RuntimeError("Sample-count mismatch: {}".format(path))
    if metrics.get("ir_used") is not False or metrics.get("rgb_only") is not True:
        raise RuntimeError("Run is not RGB-only: {}".format(path))
    if metrics.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Manifest identity mismatch: {}".format(path))
    if metrics.get("base_checkpoint_sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError("FT base identity mismatch: {}".format(path))
    if metrics.get("architecture_sha256") != EXPECTED_ARCHITECTURE_SHA256:
        raise RuntimeError("Architecture identity mismatch: {}".format(path))
    expected_epoch = None if mode == "base" else EXPECTED_SELECTED_EPOCH[seed]
    if metrics.get("selected_epoch") != expected_epoch:
        raise RuntimeError("Selected epoch mismatch: {}".format(path))
    expected_checkpoint = None if mode == "base" else EXPECTED_CHECKPOINT_SHA256[seed]
    if metrics.get("checkpoint_sha256") != expected_checkpoint:
        raise RuntimeError("Residual checkpoint identity mismatch: {}".format(path))
    by_name = {}
    for row in rows:
        if row["name"] in by_name:
            raise RuntimeError("Duplicate sample: {}".format(row["name"]))
        if any(not math.isfinite(float(row[key])) for key in METRICS):
            raise RuntimeError("Non-finite metric: {}".format(row["name"]))
        by_name[row["name"]] = row
    return metrics, by_name


def bootstrap(values, rng):
    count = values.shape[0]
    estimates = np.empty((BOOTSTRAP_RESAMPLES, values.shape[1]), dtype=np.float64)
    chunk = 500
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        stop = min(start + chunk, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, count, size=(stop - start, count))
        estimates[start:stop] = values[indices].mean(axis=1)
    return {
        metric: {
            "mean": float(values[:, index].mean()),
            "ci95_low": float(np.percentile(estimates[:, index], 2.5)),
            "ci95_high": float(np.percentile(estimates[:, index], 97.5)),
        }
        for index, metric in enumerate(METRICS)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.output_root / "final_summary.json"
    paired_path = args.output_root / "paired_deltas.jsonl"
    if summary_path.exists() or paired_path.exists():
        raise FileExistsError("Refusing to overwrite final analysis in {}".format(args.output_root))
    base_metrics, base = load_run(args.output_root / "base", "base", 3407)
    residuals = {}
    run_metrics = {}
    for seed in SEEDS:
        run_metrics[seed], residuals[seed] = load_run(
            args.output_root / "residual_seed{}".format(seed), "residual", seed
        )

    names = sorted(base)
    for seed in SEEDS:
        if set(residuals[seed]) != set(names):
            raise RuntimeError("Seed {} sample identities differ from base".format(seed))
    paired_rows = []
    source_stems = set()
    seed_summaries = {}
    seed_values = {seed: {metric: [] for metric in METRICS} for seed in SEEDS}
    mean_delta_by_sample = []
    for name in names:
        base_row = base[name]
        if base_row["source_stem"] in source_stems:
            raise RuntimeError("Duplicate bootstrap unit: {}".format(base_row["source_stem"]))
        source_stems.add(base_row["source_stem"])
        row = {
            "name": name,
            "source_stem": base_row["source_stem"],
            "severity": base_row["severity"],
            "time": base_row["time"],
            "base": {metric: base_row[metric] for metric in METRICS},
            "seed_deltas": {},
        }
        seed_delta_matrix = []
        for seed in SEEDS:
            candidate = residuals[seed][name]
            identity = ("source_stem", "severity", "time")
            if any(candidate[key] != base_row[key] for key in identity):
                raise RuntimeError("Metadata mismatch for {} seed {}".format(name, seed))
            deltas = {metric: candidate[metric] - base_row[metric] for metric in METRICS}
            row["seed_deltas"][str(seed)] = deltas
            seed_delta_matrix.append([deltas[metric] for metric in METRICS])
            for metric in METRICS:
                seed_values[seed][metric].append(deltas[metric])
        mean_delta = np.asarray(seed_delta_matrix, dtype=np.float64).mean(axis=0)
        row["mean_delta"] = {metric: float(mean_delta[index]) for index, metric in enumerate(METRICS)}
        mean_delta_by_sample.append(mean_delta)
        paired_rows.append(row)

    for seed in SEEDS:
        seed_summaries[str(seed)] = {
            metric: float(np.mean(seed_values[seed][metric])) for metric in METRICS
        }

    all_values = np.asarray(mean_delta_by_sample, dtype=np.float64)
    grouped = {}
    for group_index, group in enumerate(GROUPS):
        if group == "all":
            selected = all_values
        elif group in {"light", "dense", "very_dense"}:
            selected = all_values[[row["severity"] == group for row in paired_rows]]
        else:
            selected = all_values[[row["time"] == group for row in paired_rows]]
        if not len(selected):
            raise RuntimeError("Frozen stratum is empty: {}".format(group))
        grouped[group] = {
            "count": int(len(selected)),
            "paired_bootstrap": bootstrap(selected, np.random.default_rng(BOOTSTRAP_SEED + group_index)),
        }

    primary = grouped["all"]["paired_bootstrap"]
    checks = {
        "all_seed_psnr_means_positive": all(
            seed_summaries[str(seed)]["psnr"] > 0.0 for seed in SEEDS
        ),
        "mean_psnr_positive": primary["psnr"]["mean"] > 0.0,
        "psnr_ci95_low_positive": primary["psnr"]["ci95_low"] > 0.0,
        "mean_ssim_nonnegative": primary["ssim"]["mean"] >= 0.0,
        "mean_lpips_nonpositive": primary["lpips"]["mean"] <= 0.0,
    }
    summary = {
        "protocol": "post_ft_rgb_restormer_final_v1",
        "comparison": "three-seed RGB residual minus identical frozen FT base",
        "seeds": list(SEEDS),
        "sample_count": EXPECTED_COUNT,
        "bootstrap": {
            "unit": "source_stem",
            "unique_sources_required": EXPECTED_COUNT,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "base_metrics": base_metrics["aggregates"],
        "residual_metrics": {str(seed): run_metrics[seed]["aggregates"] for seed in SEEDS},
        "seed_mean_deltas": seed_summaries,
        "groups": grouped,
        "checks": checks,
        "pass": all(checks.values()),
        "post_test_selection_allowed": False,
    }
    summary["decision"] = (
        "FINAL_TEST_SUPPORTS_RGB_RESIDUAL"
        if summary["pass"]
        else "FINAL_TEST_DOES_NOT_SUPPORT_RGB_RESIDUAL"
    )
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with paired_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in paired_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
