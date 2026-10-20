"""Validation-only stratified evaluation for Protocol V2 checkpoints.

This script must only receive the fixed validation directories. It reports the
same PSNR, SSIM, and LPIPS definitions used by the Protocol V2 trainer, grouped
by smoke severity and day/night metadata from the generation manifest.
"""

import argparse
import hashlib
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DEHAZEFORMER_ROOT = SCRIPT_DIR / "DehazeFormer-main"
if LOCAL_DEHAZEFORMER_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_DEHAZEFORMER_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import lpips

# The server repository uses data/dataset_crossmodal.py, while the portable
# Windows folder keeps the same locked file beside this evaluator.
if not (SCRIPT_DIR / "data" / "dataset_crossmodal.py").is_file():
    import dataset_crossmodal as local_dataset_crossmodal

    data_package = sys.modules.setdefault("data", types.ModuleType("data"))
    if not hasattr(data_package, "__path__"):
        data_package.__path__ = []
    sys.modules["data.dataset_crossmodal"] = local_dataset_crossmodal

from crossmodal_dehazeformer_residual_v2 import (
    CrossModalDehazeFormerResidualV2,
    load_dehazeformer_base_weights,
    load_v2_checkpoint,
)
from train_cm_dehazeformer_v2_residual import (
    FixedEvalSmokeDataset,
    seed_everything,
    seed_worker,
    to_01,
)


EXPECTED_MANIFEST_SHA256 = (
    "943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128"
)
EXPECTED_SAMPLE_COUNT = 1469

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "B0": {
        "train_mode": "frozen_baseline",
        "ablation": "wo_ir_physics",
        "wo_gate": True,
        "use_attnres": False,
        "adapter_variant": "fscatterres",
        "refiner_width": 24,
        "refiner_blocks": 4,
        "adapter_enabled": False,
        "checkpoint_type": "base",
    },
    "A0": {
        "train_mode": "adapter",
        "ablation": "wo_ir_physics",
        "wo_gate": True,
        "use_attnres": False,
        "adapter_variant": "fscatterres",
        "refiner_width": 24,
        "refiner_blocks": 4,
        "adapter_enabled": True,
        "checkpoint_type": "v2",
    },
    "P0": {
        "train_mode": "adapter",
        "ablation": "wo_ir_physics",
        "wo_gate": True,
        "use_attnres": False,
        "adapter_variant": "plain",
        "refiner_width": 25,
        "refiner_blocks": 4,
        "adapter_enabled": True,
        "checkpoint_type": "v2",
    },
    "A2": {
        "train_mode": "adapter",
        "ablation": "wo_ir",
        "wo_gate": True,
        "use_attnres": True,
        "adapter_variant": "fscatterres",
        "refiner_width": 24,
        "refiner_blocks": 4,
        "adapter_enabled": True,
        "checkpoint_type": "v2",
    },
    "FT": {
        "train_mode": "base_only",
        "ablation": "wo_ir_physics",
        "wo_gate": True,
        "use_attnres": False,
        "adapter_variant": "fscatterres",
        "refiner_width": 24,
        "refiner_blocks": 4,
        "adapter_enabled": False,
        "checkpoint_type": "v2",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_lq", required=True)
    parser.add_argument("--val_ir", required=True)
    parser.add_argument("--val_gt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def reject_test_paths(args: argparse.Namespace) -> None:
    for key in ("val_lq", "val_ir", "val_gt", "manifest"):
        normalized = str(Path(getattr(args, key)).resolve()).replace("\\", "/").lower()
        if "/test/" in normalized or "synthetic_smoke_test" in normalized:
            raise ValueError(f"Refusing test path in --{key}: {getattr(args, key)}")


def load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Validation manifest hash mismatch: {actual_hash} != "
            f"{EXPECTED_MANIFEST_SHA256}"
        )

    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            stem = record["output_stem"]
            if stem in records:
                raise ValueError(f"Duplicate output_stem at line {line_number}: {stem}")
            if record["level"] not in {"light", "dense", "very_dense"}:
                raise ValueError(f"Invalid severity at line {line_number}: {record['level']}")
            records[stem] = record

    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} manifest records, found {len(records)}"
        )
    return records


def validate_checkpoint_args(
    checkpoint: Path, experiment_id: str, expected_seed: int
) -> None:
    expected = EXPERIMENTS[experiment_id]
    saved = torch.load(checkpoint, map_location="cpu")
    saved_args = saved.get("args", {}) if isinstance(saved, dict) else {}
    if not saved_args:
        raise ValueError(f"{experiment_id} checkpoint has no saved args: {checkpoint}")

    keys = (
        "experiment_id",
        "train_mode",
        "ablation",
        "wo_gate",
        "use_attnres",
        "adapter_variant",
        "refiner_width",
        "refiner_blocks",
        "size",
        "seed",
    )
    required = {
        "experiment_id": experiment_id,
        "size": 256,
        "seed": expected_seed,
        **{key: expected[key] for key in keys if key in expected},
    }
    mismatches = {
        key: {"expected": value, "actual": saved_args.get(key)}
        for key, value in required.items()
        if saved_args.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint config mismatch: {json.dumps(mismatches, sort_keys=True)}")


def build_model(
    experiment_id: str,
    checkpoint: Path,
    device: torch.device,
    expected_seed: int,
) -> torch.nn.Module:
    config = EXPERIMENTS[experiment_id]
    model = CrossModalDehazeFormerResidualV2(
        backbone="b",
        init_gamma=0.01,
        ablation=config["ablation"],
        wo_gate=config["wo_gate"],
        use_attnres=config["use_attnres"],
        refiner_width=config["refiner_width"],
        refiner_blocks=config["refiner_blocks"],
        adapter_enabled=config["adapter_enabled"],
        adapter_variant=config["adapter_variant"],
    ).to(device)

    if config["checkpoint_type"] == "base":
        missing, unexpected = load_dehazeformer_base_weights(
            model, str(checkpoint), strict=True
        )
    else:
        validate_checkpoint_args(checkpoint, experiment_id, expected_seed)
        missing, unexpected = load_v2_checkpoint(model, str(checkpoint), strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load was not exact: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    model.set_adapter_enabled(config["adapter_enabled"])
    model.eval()
    return model


def ssim_per_sample(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, 1, window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, window_size // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, 1, window_size // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, window_size // 2) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return score.mean(dim=(1, 2, 3))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "count": len(rows),
        "psnr": sum(row["psnr"] for row in rows) / len(rows),
        "ssim": sum(row["ssim"] for row in rows) / len(rows),
        "lpips": sum(row["lpips"] for row in rows) / len(rows),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("batch must be positive and workers must be non-negative")
    reject_test_paths(args)
    seed_everything(args.seed, deterministic=True)

    checkpoint = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(manifest_path)

    dataset = FixedEvalSmokeDataset(
        args.val_lq, args.val_ir, args.val_gt, size=args.size
    )
    dataset_stems = {Path(name).stem for name in dataset.names}
    if dataset_stems != set(records):
        raise ValueError(
            f"Dataset/manifest mismatch: dataset_only={len(dataset_stems - set(records))}, "
            f"manifest_only={len(set(records) - dataset_stems)}"
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed + 1)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.experiment_id, checkpoint, device, args.seed)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    rows: List[Dict[str, Any]] = []
    groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for batch in tqdm(loader, desc=f"Stratified validation {args.experiment_id}"):
        lq = to_01(batch["lq"].to(device, non_blocking=True))
        gt = to_01(batch["gt"].to(device, non_blocking=True))
        ir = batch["ir"].to(device, non_blocking=True)
        pred = model(lq, ir).clamp(0, 1)

        mse = F.mse_loss(pred, gt, reduction="none").mean(dim=(1, 2, 3))
        psnr_values = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        ssim_values = ssim_per_sample(pred, gt)
        lpips_values = lpips_fn(pred * 2 - 1, gt * 2 - 1).view(-1)

        for index, stem in enumerate(batch["name"]):
            metadata = records[stem]
            time_tag = "night" if metadata["is_night"] else "day"
            row = {
                "name": stem,
                "source_stem": metadata["source_stem"],
                "level": metadata["level"],
                "time": time_tag,
                "psnr": float(psnr_values[index].item()),
                "ssim": float(ssim_values[index].item()),
                "lpips": float(lpips_values[index].item()),
            }
            rows.append(row)
            for group in ("all", row["level"], time_tag):
                groups[group].append(row)

    ordered_groups = ("all", "light", "dense", "very_dense", "day", "night")
    aggregates = {group: summarize(groups[group]) for group in ordered_groups}
    result = {
        "experiment_id": args.experiment_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sample_count": len(rows),
        "size": args.size,
        "seed": args.seed,
        "aggregates": aggregates,
    }

    result_path = output_dir / f"{args.experiment_id}_stratified_metrics.json"
    sample_path = output_dir / f"{args.experiment_id}_per_sample.jsonl"
    with result_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with sample_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps(aggregates, ensure_ascii=False, indent=2))
    print(f"metrics: {result_path}")
    print(f"per-sample: {sample_path}")


if __name__ == "__main__":
    main()
