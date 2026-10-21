"""Frozen source-stem split and two checkpoint-selection rules for FT validation."""

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = "943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128"
EPOCHS = (2, 4, 6, 8, 10)
STRATA = tuple((time, level) for time in ("day", "night") for level in ("light", "dense", "very_dense"))
SPLIT_SALT = "spic-ft-selection-20261021-v1"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def summarize(rows):
    return {"count": len(rows), **{key: mean(rows, key) for key in ("psnr", "ssim", "lpips")}}


def partition(manifest):
    by_stratum = {stratum: [] for stratum in STRATA}
    seen_source = set()
    for row in manifest:
        source = row["source_stem"]
        if source in seen_source:
            raise ValueError("Repeated source_stem; scene grouping needs a separate audit: " + source)
        seen_source.add(source)
        stratum = ("night" if row["is_night"] else "day", row["level"])
        by_stratum[stratum].append(row)
    selection = set()
    counts = {}
    for stratum, rows in by_stratum.items():
        if len(rows) < 2:
            raise ValueError("Cannot split stratum: " + str(stratum))
        rows.sort(key=lambda row: hashlib.sha256(
            (SPLIT_SALT + "|" + row["source_stem"]).encode("utf-8")
        ).digest())
        selection.update(row["output_stem"] for row in rows[:len(rows) // 2])
        counts["/".join(stratum)] = {"selection": len(rows) // 2, "diagnostic": len(rows) - len(rows) // 2}
    return selection, counts


def checkpoint_metrics(rows, names):
    subset = [row for row in rows if row["name"] in names]
    if len(subset) != len(names):
        raise ValueError("Incomplete metric rows")
    strata = {"/".join(stratum): summarize([
        row for row in subset if (row["time"], row["level"]) == stratum
    ]) for stratum in STRATA}
    return {
        "all": summarize(subset),
        "day_very_dense": summarize([row for row in subset if row["time"] == "day" and row["level"] == "very_dense"]),
        "night_very_dense": summarize([row for row in subset if row["time"] == "night" and row["level"] == "very_dense"]),
        "balanced_psnr": sum(strata[key]["psnr"] for key in strata) / 6,
        "strata": strata,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("preflight", "analyze"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--metrics-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if "test" in str(args.manifest).lower().replace("\\", "/"):
        raise ValueError("Refusing a test manifest")
    actual_hash = file_hash(args.manifest)
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Manifest identity mismatch: " + actual_hash)
    manifest = read_jsonl(args.manifest)
    if len(manifest) != 1469 or len({row["output_stem"] for row in manifest}) != 1469:
        raise ValueError("Expected 1469 unique validation outputs")
    selection, counts = partition(manifest)
    all_names = {row["output_stem"] for row in manifest}
    metadata = {row["output_stem"]: (
        row["source_stem"], "night" if row["is_night"] else "day", row["level"]
    ) for row in manifest}
    diagnostic = all_names - selection
    split = {
        "manifest_sha256": actual_hash,
        "split_salt": SPLIT_SALT,
        "stratum_counts": counts,
        "selection_names": sorted(selection),
        "diagnostic_names": sorted(diagnostic),
        "scene_identity_note": "source_stem disjoint only; scene/video identities are unavailable",
    }
    if args.mode == "preflight":
        result = {"status": "PREFLIGHT_PASS", "split": split, "epochs": EPOCHS}
    else:
        if args.metrics_root is None:
            raise ValueError("--metrics-root is required for analyze")
        by_epoch = {}
        for epoch in EPOCHS:
            path = args.metrics_root / ("epoch" + str(epoch)) / "FT_per_sample.jsonl"
            rows = read_jsonl(path)
            if len(rows) != 1469 or {row["name"] for row in rows} != all_names:
                raise ValueError("Metric identity mismatch: " + str(path))
            for row in rows:
                if (row["source_stem"], row["time"], row["level"]) != metadata[row["name"]]:
                    raise ValueError("Metric stratum mismatch: " + row["name"])
                if not all(math.isfinite(float(row[key])) for key in ("psnr", "ssim", "lpips")):
                    raise ValueError("Nonfinite metric: " + row["name"])
            by_epoch[str(epoch)] = {
                "metrics_sha256": file_hash(path),
                "selection": checkpoint_metrics(rows, selection),
                "diagnostic": checkpoint_metrics(rows, diagnostic),
                "full_validation": checkpoint_metrics(rows, all_names),
            }
        aggregate_epoch = max(EPOCHS, key=lambda e: by_epoch[str(e)]["selection"]["all"]["psnr"])
        balanced_epoch = max(EPOCHS, key=lambda e: by_epoch[str(e)]["selection"]["balanced_psnr"])
        a = by_epoch[str(aggregate_epoch)]["diagnostic"]
        b = by_epoch[str(balanced_epoch)]["diagnostic"]
        differences = {group: {key: b[group][key] - a[group][key] for key in ("psnr", "ssim", "lpips")}
                       for group in ("all", "day_very_dense", "night_very_dense")}
        result = {
            "status": "EXPLORATORY_ONLY_SAME_VALIDATION_SCENES",
            "split": split,
            "aggregate_epoch": aggregate_epoch,
            "balanced_epoch": balanced_epoch,
            "diagnostic_balanced_minus_aggregate": differences,
            "practical_gate": (differences["night_very_dense"]["psnr"] >= 0.10 and
                               differences["all"]["psnr"] >= -0.05 and
                               differences["day_very_dense"]["psnr"] >= -0.05),
            "by_epoch": by_epoch,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["status"] + " output=" + str(args.output))
    if args.mode == "analyze":
        print("selected aggregate epoch=" + str(aggregate_epoch) + " balanced epoch=" + str(balanced_epoch))
        print("diagnostic balanced-minus-aggregate PSNR=" + json.dumps({
            group: differences[group]["psnr"] for group in differences}, sort_keys=True))
        print("PRACTICAL_GATE=" + str(result["practical_gate"]))


if __name__ == "__main__":
    main()
