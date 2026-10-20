"""Evaluate frozen FFA-Net RGB-only residual checkpoints on the locked test."""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from post_ft_rgb_residual_v1 import build_model


EXPECTED_MANIFEST_SHA256 = "4f1782e5d4c6e946d7819c5bb647d7f1a5f49355414b54a37cd935d5e4a3d0e2"
EXPECTED_SAMPLE_COUNT = 768
EXPECTED_BASE_PARAMETERS = 4_455_913
EXPECTED_ADAPTER_PARAMETERS = 46_254
EXPECTED_SELECTED_EPOCH = {3407: 12, 2026: 15, 8888: 9}
EXPECTED_ARCHITECTURE_SHA256 = "8e2e7122cb510f157ccf7e1ae09b8885d63634f5e572d12c43ef06e95e3bc199"
EXPECTED_BASE_SHA256 = "113587bb1c656fd77b1c0999f4f0d8e60e55b5f351fa2bd53f39444d8bd1d206"
EXPECTED_CHECKPOINT_SHA256 = {
    3407: "a852a3b3bfb1b5eb65ba1d36cad2de50cd5bee863611a406775d6015b607a725",
    2026: "b303fb9b69f6b8264135a5b6b803ca9e3d9c0bd5851080047ab4776585cdedc0",
    8888: "ef92d0621f3f6b04c52944b44064e11983136bb8e10d13d327d32056429268b6",
}
LOCKED_SIZE = 256
LOCKED_BATCH = 2
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class LockedRGBDataset(Dataset):
    def __init__(self, lq_dir, gt_dir):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.names = sorted(
            path.name
            for path in lq_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(self.names) != EXPECTED_SAMPLE_COUNT:
            raise RuntimeError("Expected {} test images, found {}".format(EXPECTED_SAMPLE_COUNT, len(self.names)))
        self.gt_paths = {}
        for name in self.names:
            stem = Path(name).stem
            matches = [path for path in gt_dir.glob(stem + ".*") if path.suffix.lower() in IMAGE_EXTENSIONS]
            if len(matches) != 1:
                raise FileNotFoundError("Expected one GT for {} in {}".format(stem, gt_dir))
            self.gt_paths[stem] = matches[0]

    def __len__(self):
        return len(self.names)

    @staticmethod
    def load_rgb(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = cv2.resize(image, (LOCKED_SIZE, LOCKED_SIZE), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image).permute(2, 0, 1).float()

    def __getitem__(self, index):
        name = self.names[index]
        stem = Path(name).stem
        return {
            "lq": self.load_rgb(self.lq_dir / name),
            "gt": self.load_rgb(self.gt_paths[stem]),
            "name": stem,
        }


def load_manifest(path):
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Final-test manifest hash mismatch: {}".format(actual_hash))
    records = {}
    source_stems = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            required = {"output_stem", "source_stem", "severity", "is_night"}
            missing = sorted(required - set(record))
            if missing:
                raise ValueError("Manifest row {} missing {}".format(line_number, missing))
            stem = record["output_stem"]
            source_stem = record["source_stem"]
            if stem in records or source_stem in source_stems:
                raise ValueError("Duplicate output/source identity at row {}".format(line_number))
            if record["severity"] not in {"light", "dense", "very_dense"}:
                raise ValueError("Invalid severity at row {}".format(line_number))
            if not isinstance(record["is_night"], bool):
                raise ValueError("Invalid is_night at row {}".format(line_number))
            records[stem] = record
            source_stems.add(source_stem)
    if len(records) != EXPECTED_SAMPLE_COUNT or len(source_stems) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("Final test must contain 768 unique outputs and sources")
    return records


def ssim_per_sample(x, y, window_size=11):
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


def validate_residual_checkpoint(saved, seed):
    if not isinstance(saved, dict) or "model" not in saved or "args" not in saved:
        raise ValueError("Residual checkpoint is missing model/args")
    expected = {
        "backbone": "ffa",
        "seed": seed,
        "epochs": 15,
        "eval_every": 3,
        "size": 256,
        "init_gamma": 0.01,
        "batch": 8,
        "grad_accum_steps": 2,
        "workers": 8,
        "lr": 1e-4,
        "deterministic": True,
        "smoke_test": False,
    }
    mismatches = {
        key: {"expected": value, "actual": saved["args"].get(key)}
        for key, value in expected.items()
        if saved["args"].get(key) != value
    }
    expected_epoch = EXPECTED_SELECTED_EPOCH[seed]
    if saved.get("epoch") != expected_epoch:
        mismatches["selected_epoch"] = {"expected": expected_epoch, "actual": saved.get("epoch")}
    if mismatches:
        raise ValueError("Residual checkpoint identity mismatch: {}".format(json.dumps(mismatches, sort_keys=True)))


def verify_base_identity(model, reference):
    model_state = model.base.state_dict()
    reference_state = reference.base.state_dict()
    if list(model_state) != list(reference_state):
        raise RuntimeError("Residual/base state keys differ")
    for key in model_state:
        if not torch.equal(model_state[key].cpu(), reference_state[key].cpu()):
            raise RuntimeError("Residual checkpoint embeds a different FT base: {}".format(key))


def build_eval_model(args, device):
    if sha256_file(args.architecture) != EXPECTED_ARCHITECTURE_SHA256:
        raise ValueError("FFA-Net architecture hash mismatch")
    if sha256_file(args.base_checkpoint) != EXPECTED_BASE_SHA256:
        raise ValueError("Corrected FT FFA-Net checkpoint hash mismatch")
    reference = build_model("ffa", args.architecture, 0.01)
    reference.load_base_weights_strict(args.base_checkpoint)
    if args.mode == "base":
        model = reference
        adapter_enabled = False
        checkpoint_epoch = None
    else:
        actual_checkpoint_hash = sha256_file(args.checkpoint)
        if actual_checkpoint_hash != EXPECTED_CHECKPOINT_SHA256[args.seed]:
            raise ValueError("Residual checkpoint hash mismatch: {}".format(actual_checkpoint_hash))
        saved = torch.load(str(args.checkpoint), map_location="cpu")
        validate_residual_checkpoint(saved, args.seed)
        model = build_model("ffa", args.architecture, 0.01)
        incompatible = model.load_state_dict(saved["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Strict residual checkpoint load failed")
        verify_base_identity(model, reference)
        adapter_enabled = True
        checkpoint_epoch = saved["epoch"]
    counts = model.parameter_counts()
    if counts["base_total"] != EXPECTED_BASE_PARAMETERS or counts["base_trainable"] != 0:
        raise RuntimeError("Frozen FFA-Net parameter identity mismatch")
    if counts["adapter_total"] != EXPECTED_ADAPTER_PARAMETERS or model.refiner.gate is None:
        raise RuntimeError("RGB-only gated adapter identity mismatch")
    model = model.to(device).eval()
    return model, adapter_enabled, counts, checkpoint_epoch


def summarize(rows):
    if not rows:
        raise ValueError("A frozen final-test stratum is empty")
    return {
        "count": len(rows),
        "psnr": sum(row["psnr"] for row in rows) / len(rows),
        "ssim": sum(row["ssim"] for row in rows) / len(rows),
        "lpips": sum(row["lpips"] for row in rows) / len(rows),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("base", "residual"))
    parser.add_argument("--seed", type=int, required=True, choices=sorted(EXPECTED_SELECTED_EPOCH))
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--test_lq", type=Path, required=True)
    parser.add_argument("--test_gt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.mode == "base" and args.seed != 3407:
        raise ValueError("The shared FT base is evaluated once with seed 3407")
    if args.mode == "base" and args.checkpoint is not None:
        raise ValueError("Base mode must not receive a residual checkpoint")
    if args.mode == "residual" and args.checkpoint is None:
        raise ValueError("Residual mode requires --checkpoint")
    for key in ("test_lq", "test_gt", "manifest"):
        normalized = str(getattr(args, key).resolve()).replace("\\", "/").lower()
        if "synthetic_smoke_test" not in normalized and "/test/" not in normalized:
            raise ValueError("Refusing non-test path for {}".format(key))
    if args.output_dir.exists():
        raise FileExistsError("Refusing existing output directory: {}".format(args.output_dir))

    seed_everything(args.seed)
    records = load_manifest(args.manifest)
    dataset = LockedRGBDataset(args.test_lq, args.test_gt)
    stems = {Path(name).stem for name in dataset.names}
    if stems != set(records):
        raise ValueError("Dataset and manifest stems differ")
    generator = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(
        dataset,
        batch_size=LOCKED_BATCH,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, adapter_enabled, counts, checkpoint_epoch = build_eval_model(args, device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    groups = defaultdict(list)
    for batch in tqdm(loader, desc="Locked FFA-Net final test {} seed {}".format(args.mode, args.seed)):
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        prediction = model(lq * 2.0 - 1.0, None, adapter_enabled=adapter_enabled).clamp(0, 1)
        mse = F.mse_loss(prediction, gt, reduction="none").mean(dim=(1, 2, 3))
        psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        ssim = ssim_per_sample(prediction, gt)
        perceptual = lpips_model(prediction * 2.0 - 1.0, gt * 2.0 - 1.0).view(-1)
        for index, stem in enumerate(batch["name"]):
            metadata = records[stem]
            row = {
                "name": stem,
                "source_stem": metadata["source_stem"],
                "severity": metadata["severity"],
                "time": "night" if metadata["is_night"] else "day",
                "psnr": float(psnr[index].item()),
                "ssim": float(ssim[index].item()),
                "lpips": float(perceptual[index].item()),
            }
            rows.append(row)
            for group in ("all", row["severity"], row["time"]):
                groups[group].append(row)
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("Final test produced {} rows".format(len(rows)))

    ordered_groups = ("all", "light", "dense", "very_dense", "day", "night")
    result = {
        "protocol": "post_ft_rgb_ffa_final_v1",
        "mode": args.mode,
        "seed": args.seed,
        "selected_epoch": checkpoint_epoch,
        "adapter_enabled": adapter_enabled,
        "rgb_only": True,
        "ir_used": False,
        "physics_used": False,
        "spatial_gate": True,
        "init_gamma": 0.01,
        "gamma_clip": [0.0, 0.12],
        "sample_count": len(rows),
        "size": LOCKED_SIZE,
        "batch": LOCKED_BATCH,
        "parameter_counts": counts,
        "architecture_sha256": sha256_file(args.architecture),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
        "manifest_sha256": sha256_file(args.manifest),
        "aggregates": {group: summarize(groups[group]) for group in ordered_groups},
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_sample.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
    print("FINAL_EVAL_PASS mode={} seed={}".format(args.mode, args.seed))


if __name__ == "__main__":
    main()
