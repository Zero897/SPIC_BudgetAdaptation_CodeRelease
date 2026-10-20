"""Frozen zero-shot RRSHID evaluator for the Protocol V2 comparison set.

This evaluator accepts only the fixed 304-pair RRSHID test split and the
already selected B0/P0/FT checkpoints. It performs no training, adaptation,
calibration, checkpoint selection, or output scaling.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DEHAZEFORMER_ROOT = SCRIPT_DIR / "DehazeFormer-main"
if LOCAL_DEHAZEFORMER_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_DEHAZEFORMER_ROOT))

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from crossmodal_dehazeformer_residual_v2 import (
    CrossModalDehazeFormerResidualV2,
    load_dehazeformer_base_weights,
    load_v2_checkpoint,
)


EXPECTED_MANIFEST_SHA256 = (
    "3a0a349a3a92eff0e7467def332c2dda4e1ec38047855cb485762db894221053"
)
EXPECTED_SAMPLE_COUNT = 304
EXPECTED_DENSITY_COUNTS = {"thin": 76, "moderate": 152, "thick": 76}
LOCKED_SIZE = 256
LOCKED_BATCH = 16
ALLOWED_SEEDS = {3407, 2026, 8888}

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


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen Protocol V2 checkpoint zero-shot on RRSHID."
    )
    parser.add_argument("--experiment_id", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True, choices=sorted(ALLOWED_SEEDS))
    parser.add_argument("--preflight_only", action="store_true")
    return parser.parse_args()


def require_test_path(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    if "train" in parts or "val" in parts or "validation" in parts:
        raise ValueError(f"Refusing non-test RRSHID path: {path}")
    if not parts or parts[-1] != "test":
        raise ValueError(f"RRSHID root must be the fixed test directory: {path}")


def validate_run_identity(experiment_id: str, seed: int) -> None:
    if experiment_id == "B0" and seed != 3407:
        raise ValueError("B0 is evaluated once with the fixed evaluator seed 3407")


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"RRSHID manifest hash mismatch: {actual_hash} != "
            f"{EXPECTED_MANIFEST_SHA256}"
        )

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            required = {
                "index",
                "filename",
                "density",
                "lq_relpath",
                "gt_relpath",
                "lq_sha256",
                "gt_sha256",
                "lq_bytes",
                "gt_bytes",
                "width",
                "height",
                "mode",
            }
            missing = sorted(required - set(record))
            if missing:
                raise ValueError(f"Manifest row {line_number} missing fields: {missing}")
            if record["index"] != line_number - 1:
                raise ValueError(f"Non-sequential index at manifest row {line_number}")
            if record["density"] not in EXPECTED_DENSITY_COUNTS:
                raise ValueError(f"Invalid density at manifest row {line_number}")
            filename = record["filename"]
            if record["lq_relpath"] != f"lq/{filename}":
                raise ValueError(f"Invalid LQ path at manifest row {line_number}")
            if record["gt_relpath"] != f"gt/{filename}":
                raise ValueError(f"Invalid GT path at manifest row {line_number}")
            if (
                record["width"] != LOCKED_SIZE
                or record["height"] != LOCKED_SIZE
                or record["mode"] != "RGB"
            ):
                raise ValueError(f"Invalid image identity at manifest row {line_number}")
            records.append(record)

    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} manifest rows, found {len(records)}"
        )
    names = [record["filename"] for record in records]
    if len(set(names)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("RRSHID manifest contains duplicate filenames")
    counts = Counter(record["density"] for record in records)
    if dict(counts) != EXPECTED_DENSITY_COUNTS:
        raise ValueError(
            f"RRSHID density counts mismatch: {dict(counts)} != "
            f"{EXPECTED_DENSITY_COUNTS}"
        )
    return records


class RRSHIDTestDataset(Dataset):
    def __init__(self, test_root: Path, records: List[Dict[str, Any]]) -> None:
        self.test_root = test_root
        self.records = records
        for record in records:
            for role in ("lq", "gt"):
                path = test_root / record[f"{role}_relpath"]
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path.stat().st_size != record[f"{role}_bytes"]:
                    raise ValueError(f"File-size mismatch: {path}")
                actual_hash = sha256_file(path)
                if actual_hash != record[f"{role}_sha256"]:
                    raise ValueError(f"SHA256 mismatch: {path}")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read_rgb(path: Path) -> torch.Tensor:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        if image.shape != (LOCKED_SIZE, LOCKED_SIZE, 3):
            raise ValueError(f"Unexpected RRSHID image shape {image.shape}: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        return {
            "lq": self._read_rgb(self.test_root / record["lq_relpath"]),
            "gt": self._read_rgb(self.test_root / record["gt_relpath"]),
            "name": Path(record["filename"]).stem,
            "filename": record["filename"],
            "density": record["density"],
        }


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
        "size": LOCKED_SIZE,
        "seed": expected_seed,
        **{key: expected[key] for key in keys if key in expected},
    }
    mismatches = {
        key: {"expected": value, "actual": saved_args.get(key)}
        for key, value in required.items()
        if saved_args.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Checkpoint config mismatch: {json.dumps(mismatches, sort_keys=True)}"
        )


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


def ssim_per_sample(
    x: torch.Tensor, y: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, 1, window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, window_size // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, 1, window_size // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, window_size // 2) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1)
        * (sigma_x + sigma_y + c2)
        + 1e-8
    )
    return score.mean(dim=(1, 2, 3))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("A predefined RRSHID stratum is empty")
    return {
        "count": len(rows),
        "psnr": sum(row["psnr"] for row in rows) / len(rows),
        "ssim": sum(row["ssim"] for row in rows) / len(rows),
        "lpips": sum(row["lpips"] for row in rows) / len(rows),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    validate_run_identity(args.experiment_id, args.seed)
    seed_everything(args.seed)

    checkpoint = Path(args.checkpoint).resolve()
    test_root = Path(args.test_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    require_test_path(test_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    records = load_manifest(manifest_path)
    dataset = RRSHIDTestDataset(test_root, records)
    device = torch.device("cpu" if args.preflight_only else "cuda")
    if not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen full evaluation")
    model = build_model(args.experiment_id, checkpoint, device, args.seed)

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "RRSHID_ZERO_SHOT_PREFLIGHT_PASS",
                    "experiment_id": args.experiment_id,
                    "seed": args.seed,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "manifest_sha256": sha256_file(manifest_path),
                    "sample_count": len(dataset),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result_path = output_dir / "summary.json"
    sample_path = output_dir / "per_sample.jsonl"
    config_path = output_dir / "run_config.json"
    for path in (result_path, sample_path, config_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite RRSHID output: {path}")

    generator = torch.Generator()
    generator.manual_seed(args.seed + 1)
    loader = DataLoader(
        dataset,
        batch_size=LOCKED_BATCH,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    rows: List[Dict[str, Any]] = []
    groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for batch in tqdm(
        loader,
        desc=f"RRSHID zero-shot {args.experiment_id} seed {args.seed}",
    ):
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        ir_zero = torch.zeros(
            (lq.shape[0], 1, LOCKED_SIZE, LOCKED_SIZE),
            dtype=lq.dtype,
            device=device,
        )
        pred = model(lq, ir_zero).float().clamp(0, 1)

        mse = F.mse_loss(pred, gt, reduction="none").mean(dim=(1, 2, 3))
        psnr_values = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        ssim_values = ssim_per_sample(pred, gt)
        lpips_values = lpips_fn(pred * 2 - 1, gt * 2 - 1).view(-1)

        for index, (stem, filename, density) in enumerate(
            zip(batch["name"], batch["filename"], batch["density"])
        ):
            row = {
                "name": stem,
                "filename": filename,
                "density": density,
                "psnr": float(psnr_values[index].item()),
                "ssim": float(ssim_values[index].item()),
                "lpips": float(lpips_values[index].item()),
            }
            rows.append(row)
            groups["all"].append(row)
            groups[density].append(row)

    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} evaluated samples, found {len(rows)}"
        )
    if len({row["filename"] for row in rows}) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("Evaluation produced duplicate RRSHID filenames")

    ordered_groups = ("all", "thin", "moderate", "thick")
    aggregates = {group: summarize(groups[group]) for group in ordered_groups}
    source_paths = (
        SCRIPT_DIR / "crossmodal_dehazeformer_residual_v2.py",
        SCRIPT_DIR / "dataset_crossmodal.py",
        SCRIPT_DIR / "train_cm_dehazeformer_v2_residual.py",
    )
    config = {
        "protocol": "RRSHID Protocol V1 strict zero-shot",
        "command": sys.argv,
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "test_root": str(test_root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "source_sha256": {
            path.name: sha256_file(path) for path in source_paths
        },
        "sample_count": len(rows),
        "size": LOCKED_SIZE,
        "batch": LOCKED_BATCH,
        "workers": args.workers,
        "metrics": ["psnr", "ssim", "lpips_alex"],
        "strata": list(ordered_groups),
        "input_range": "RGB float [0,1] from uint8 without resizing",
        "ir_mode": "unused zero map",
        "adaptation": "none",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }
    result = {**config, "aggregates": aggregates}

    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with result_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with sample_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps(aggregates, ensure_ascii=False, indent=2))
    print(f"summary: {result_path}")
    print(f"per-sample: {sample_path}")
    print(f"config: {config_path}")


if __name__ == "__main__":
    main()
