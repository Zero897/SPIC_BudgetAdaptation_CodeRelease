"""Paired default-versus-expanded FT comparison after all six runs finish."""

import argparse
import json
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8") as f:
        return {r["name"]: r for r in map(json.loads, f)}


def compare(root, seed, rule):
    arms = {}
    for arm in ("default", "expanded"):
        home = root / f"FT_{arm}_seed{seed}"
        report = json.loads((home / "selection_comparison.json").read_text())
        epoch = report[f"{rule}_epoch"]
        arms[arm] = (report, rows(home / f"epoch{epoch}" / "FT_per_sample.jsonl"))
    default, drows = arms["default"]
    expanded, erows = arms["expanded"]
    assert default["split"] == expanded["split"]
    names = set(default["split"]["diagnostic_names"])
    assert names <= drows.keys() and names <= erows.keys()
    groups = {
        "all": names,
        "night_very_dense": {n for n in names if drows[n]["time"] == "night" and drows[n]["level"] == "very_dense"},
        "day_very_dense": {n for n in names if drows[n]["time"] == "day" and drows[n]["level"] == "very_dense"},
    }
    for name in names:
        assert (drows[name]["source_stem"], drows[name]["time"], drows[name]["level"]) == (
            erows[name]["source_stem"], erows[name]["time"], erows[name]["level"])
    delta = {group: {key: sum(erows[n][key] - drows[n][key] for n in group_names) / len(group_names)
                     for key in ("psnr", "ssim", "lpips")}
             for group, group_names in groups.items()}
    night = groups["night_very_dense"]
    gate = (delta["night_very_dense"]["psnr"] >= 0.10 and
            delta["all"]["psnr"] >= -0.05 and
            delta["day_very_dense"]["psnr"] >= -0.05)
    return {
        "seed": seed,
        "rule": rule,
        "default_epoch": default[f"{rule}_epoch"],
        "expanded_epoch": expanded[f"{rule}_epoch"],
        "diagnostic_delta_expanded_minus_default": delta,
        "night_improved_images": sum(erows[n]["psnr"] > drows[n]["psnr"] for n in night),
        "night_regressed_images": sum(erows[n]["psnr"] < drows[n]["psnr"] for n in night),
        "night_count": len(night),
        "practical_gate": gate,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = [compare(args.root, seed, rule) for rule in ("aggregate", "balanced")
               for seed in (3407, 2026, 8888)]
    summary = {"primary_rule": "aggregate", "results": results,
               "aggregate_three_seed_gate": all(r["practical_gate"] for r in results if r["rule"] == "aggregate")}
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(result["rule"], result["seed"], result["diagnostic_delta_expanded_minus_default"],
              "gate", result["practical_gate"])
    print("AGGREGATE_THREE_SEED_GATE", summary["aggregate_three_seed_gate"])


if __name__ == "__main__":
    main()
