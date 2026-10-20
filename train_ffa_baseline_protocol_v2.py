"""Train a clean FFA-Net baseline on Protocol-V2 train/fixed-validation data."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


FFA_MEAN = (0.64, 0.60, 0.58)
FFA_STD = (0.14, 0.15, 0.152)
EXPECTED_PARAMETERS = 4_455_913
FULL_MAX_STEPS = 200_000
EARLY_STOP_MIN_STEPS = 20_000
EARLY_STOP_PATIENCE = 6
EARLY_STOP_MIN_DELTA = 0.01
SAMPLE_LEVELS = ("light", "dense", "very_dense")
SAMPLES_PER_LEVEL = 5
SAMPLE_SEED_OFFSET = 1701
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load module from {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def to_01(value: torch.Tensor) -> torch.Tensor:
    if value.detach().min().item() < 0:
        return ((value + 1.0) / 2.0).clamp(0, 1)
    return value.clamp(0, 1)


def normalize_ffa(value: torch.Tensor) -> torch.Tensor:
    mean = value.new_tensor(FFA_MEAN).view(1, 3, 1, 1)
    std = value.new_tensor(FFA_STD).view(1, 3, 1, 1)
    return (value - mean) / std


def clean_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "params", "net", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def build_model(architecture: Path):
    module = load_module("ffa_protocol_v2_architecture", architecture)
    model = module.FFA(gps=3, blocks=19)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            "FFA parameter mismatch: expected={}, actual={}".format(
                EXPECTED_PARAMETERS, parameter_count
            )
        )
    return model


def load_pretrained_strict(model, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(str(path), map_location="cpu")
    incompatible = model.load_state_dict(clean_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict pretrained loading failed")


class FixedValidationDataset(Dataset):
    def __init__(self, lq_dir: Path, gt_dir: Path, size: int):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.size = size
        for path in (lq_dir, gt_dir):
            if not path.is_dir():
                raise FileNotFoundError(path)
        self.names = sorted(
            path.name
            for path in lq_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.names:
            raise RuntimeError("No validation images found in {}".format(lq_dir))
        self.stem_to_index = {
            Path(name).stem: index for index, name in enumerate(self.names)
        }
        self.gt_paths = []
        for name in self.names:
            stem = Path(name).stem
            matches = [
                path
                for path in gt_dir.glob(stem + ".*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    "Expected one validation GT for {}, found {}".format(
                        stem, len(matches)
                    )
                )
            self.gt_paths.append(matches[0])
        print("[FixedValidation] images={} lq={}".format(len(self.names), lq_dir))

    def __len__(self):
        return len(self.names)

    def _rgb(self, path: Path) -> torch.Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = cv2.resize(
            image, (self.size, self.size), interpolation=cv2.INTER_AREA
        )
        return torch.from_numpy(image).permute(2, 0, 1).float()

    def __getitem__(self, index):
        return {
            "lq": self._rgb(self.lq_dir / self.names[index]),
            "gt": self._rgb(self.gt_paths[index]),
            "name": Path(self.names[index]).stem,
        }


def simple_ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    window = 11
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_pred = F.avg_pool2d(pred, window, 1, window // 2)
    mu_gt = F.avg_pool2d(gt, window, 1, window // 2)
    var_pred = F.avg_pool2d(pred * pred, window, 1, window // 2) - mu_pred ** 2
    var_gt = F.avg_pool2d(gt * gt, window, 1, window // 2) - mu_gt ** 2
    covariance = F.avg_pool2d(pred * gt, window, 1, window // 2) - mu_pred * mu_gt
    value = ((2 * mu_pred * mu_gt + c1) * (2 * covariance + c2)) / (
        (mu_pred ** 2 + mu_gt ** 2 + c1) * (var_pred + var_gt + c2) + 1e-8
    )
    return value.mean(dim=(1, 2, 3))


@torch.no_grad()
def evaluate(model, loader, lpips_model, device, max_batches):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    sample_count = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="Eval", leave=False)):
        if max_batches >= 0 and batch_index >= max_batches:
            break
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        pred = model(normalize_ffa(lq)).float().clamp(0, 1)
        mse = F.mse_loss(pred, gt, reduction="none").mean(dim=(1, 2, 3))
        psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        ssim = simple_ssim(pred, gt)
        lpips_value = lpips_model(pred * 2.0 - 1.0, gt * 2.0 - 1.0).view(-1)
        psnr_sum += psnr.sum().item()
        ssim_sum += ssim.sum().item()
        lpips_sum += lpips_value.sum().item()
        sample_count += lq.shape[0]
    if sample_count == 0:
        raise RuntimeError("Validation produced zero samples")
    return {
        "count": sample_count,
        "psnr": psnr_sum / sample_count,
        "ssim": ssim_sum / sample_count,
        "lpips": lpips_sum / sample_count,
    }


def select_fixed_samples(
    manifest_path: Path, dataset: FixedValidationDataset, seed: int
):
    grouped = {level: [] for level in SAMPLE_LEVELS}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            level = record.get("level")
            stem = record.get("output_stem")
            if level not in grouped:
                raise ValueError(
                    "Invalid level at manifest line {}: {}".format(
                        line_number, level
                    )
                )
            if not isinstance(stem, str) or stem not in dataset.stem_to_index:
                raise ValueError(
                    "Manifest sample is missing from validation data at line {}: {}".format(
                        line_number, stem
                    )
                )
            grouped[level].append(stem)

    selection = {}
    rng = random.Random(seed)
    for level in SAMPLE_LEVELS:
        candidates = sorted(set(grouped[level]))
        if len(candidates) < SAMPLES_PER_LEVEL:
            raise RuntimeError(
                "Need {} {} samples, found {}".format(
                    SAMPLES_PER_LEVEL, level, len(candidates)
                )
            )
        selection[level] = sorted(rng.sample(candidates, SAMPLES_PER_LEVEL))
    return selection


def save_rgb_tensor(path: Path, image: torch.Tensor) -> None:
    array = (
        image.detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Failed to write sample image: {}".format(path))


@torch.no_grad()
def log_fixed_samples(
    model,
    dataset,
    selection,
    device,
    writer,
    global_step,
    output_dir,
):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for level in SAMPLE_LEVELS:
        rows = []
        for stem in selection[level]:
            sample = dataset[dataset.stem_to_index[stem]]
            lq = sample["lq"].unsqueeze(0).to(device)
            gt = sample["gt"].unsqueeze(0).to(device)
            pred = model(normalize_ffa(lq)).float().clamp(0, 1)
            rows.append(torch.cat((lq[0], pred[0], gt[0]), dim=2).cpu())
        panel = torch.cat(rows, dim=1)
        writer.add_image(
            "samples/{}/LQ_Restored_GT".format(level), panel, global_step
        )
        save_rgb_tensor(
            output_dir / "step_{:06d}_{}.png".format(global_step, level), panel
        )
    writer.flush()
    print(
        "[Samples] global_step={} levels={} per_level={} order=LQ|Restored|GT".format(
            global_step, ",".join(SAMPLE_LEVELS), SAMPLES_PER_LEVEL
        )
    )


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    global_step,
    attempted_windows,
    amp_skips,
    best_psnr,
    args,
    hashes,
    early_stop_state,
):
    torch.save(
        {
            "global_step": global_step,
            "attempted_optimizer_windows": attempted_windows,
            "amp_skips": amp_skips,
            "best_psnr": best_psnr,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "hashes": hashes,
            "early_stopping": dict(early_stop_state),
        },
        path,
    )


def strict_reload(path: Path, architecture: Path) -> None:
    checkpoint = torch.load(str(path), map_location="cpu")
    model = build_model(architecture)
    incompatible = model.load_state_dict(clean_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict reload failed for {}".format(path))
    for value in model.state_dict().values():
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite model state in {}".format(path))
    print("[Strict reload] PASS path={}".format(path))


def reject_test_path(path: Path, label: str) -> None:
    lowered = [part.lower() for part in path.resolve().parts]
    if any(part == "test" or part.startswith("test_") for part in lowered):
        raise ValueError("{} must not reference a test path: {}".format(label, path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--dataset_source", type=Path, required=True)
    parser.add_argument("--train_root", type=Path, required=True)
    parser.add_argument("--val_lq", type=Path, required=True)
    parser.add_argument("--val_gt", type=Path, required=True)
    parser.add_argument("--val_manifest", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=FULL_MAX_STEPS)
    parser.add_argument("--eval_every_steps", type=int, default=2000)
    parser.add_argument("--checkpoint_every_steps", type=int, default=2000)
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval_max_batches", type=int, default=-1)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    if args.size != 256:
        raise ValueError("Protocol requires size=256")
    if args.batch * args.grad_accum_steps != 16:
        raise ValueError("Protocol requires effective batch size 16")
    if args.max_steps <= 0 or args.eval_every_steps <= 0:
        raise ValueError("Step counts must be positive")
    if args.save_dir.exists():
        raise FileExistsError("Refusing existing save_dir: {}".format(args.save_dir))
    for path, label in (
        (args.train_root, "train_root"),
        (args.val_lq, "val_lq"),
        (args.val_gt, "val_gt"),
        (args.val_manifest, "val_manifest"),
    ):
        reject_test_path(path, label)
    if args.smoke_test:
        if args.max_steps != 1 or args.eval_max_batches <= 0:
            raise ValueError("Smoke requires max_steps=1 and bounded validation")
    else:
        expected = {
            "max_steps": FULL_MAX_STEPS,
            "eval_every_steps": 2000,
            "checkpoint_every_steps": 2000,
            "seed": 3407,
            "eval_max_batches": -1,
        }
        for key, value in expected.items():
            if getattr(args, key) != value:
                raise ValueError("Full protocol requires {}={}".format(key, value))
        if not math.isclose(args.lr, 1e-4):
            raise ValueError("Full protocol requires lr=1e-4")


def main():
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed, args.deterministic)
    args.save_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA support is required")

    dataset_module = load_module("ffa_protocol_v2_dataset", args.dataset_source)
    train_dataset = dataset_module.CrossModalUAVDataset(
        str(args.train_root), size=args.size, use_depth=False, augment=True
    )
    val_dataset = FixedValidationDataset(args.val_lq, args.val_gt, args.size)
    sample_seed = args.seed + SAMPLE_SEED_OFFSET
    fixed_samples = select_fixed_samples(
        args.val_manifest, val_dataset, sample_seed
    )
    write_json(
        args.save_dir / "sample_selection.json",
        {
            "seed": sample_seed,
            "samples_per_level": SAMPLES_PER_LEVEL,
            "levels": fixed_samples,
        },
    )
    tensorboard_dir = args.save_dir / "tensorboard"
    sample_output_dir = args.save_dir / "samples"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text(
        "configuration/fixed_samples",
        json.dumps(fixed_samples, sort_keys=True),
        0,
    )

    train_generator = torch.Generator().manual_seed(args.seed)
    eval_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch // 2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=eval_generator,
    )

    model = build_model(args.architecture)
    load_pretrained_strict(model, args.pretrained)
    model = model.to(device)
    print("[Load official FFA] missing=0 unexpected=0 strict=PASS")

    hashes = {
        "trainer": sha256_file(Path(__file__).resolve()),
        "architecture": sha256_file(args.architecture),
        "pretrained": sha256_file(args.pretrained),
        "dataset_source": sha256_file(args.dataset_source),
        "validation_manifest": sha256_file(args.val_manifest),
    }
    config = {
        "experiment_id": "FFA_BASE_V2",
        "training_precision": "bf16_autocast_fp32_optimizer",
        "seed": args.seed,
        "selection_metric": "validation_psnr",
        "test_access": "forbidden",
        "checkpoint_policy": {
            "last": "overwrite_every_{}_steps".format(
                args.checkpoint_every_steps
            ),
            "best": "overwrite_on_validation_psnr_improvement",
        },
        "early_stopping": {
            "metric": "validation_psnr",
            "mode": "max",
            "min_steps": EARLY_STOP_MIN_STEPS,
            "patience_validations": EARLY_STOP_PATIENCE,
            "min_delta_db": EARLY_STOP_MIN_DELTA,
        },
        "parameter_count": EXPECTED_PARAMETERS,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "tensorboard_dir": str(tensorboard_dir),
        "sample_output_dir": str(sample_output_dir),
        "fixed_sample_selection": {
            "seed": sample_seed,
            "samples_per_level": SAMPLES_PER_LEVEL,
            "levels": fixed_samples,
            "panel_order": "LQ|Restored|GT",
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "hashes": hashes,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
        },
    }
    write_json(args.save_dir / "run_config.json", config)
    history_path = args.save_dir / "history.jsonl"
    print("[Config] " + json.dumps(config, sort_keys=True))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=0.0
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    try:
        import lpips
    except Exception as error:
        raise RuntimeError("lpips is required for Protocol-V2 validation") from error
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    initial_metrics = evaluate(
        model, val_loader, lpips_model, device, args.eval_max_batches
    )
    initial_event = {
        "event": "initial_validation",
        "global_step": 0,
        **initial_metrics,
    }
    append_jsonl(history_path, initial_event)
    print("[Eval] " + json.dumps(initial_event, sort_keys=True))
    writer.add_scalar("validation/psnr", initial_metrics["psnr"], 0)
    writer.add_scalar("validation/ssim", initial_metrics["ssim"], 0)
    writer.add_scalar("validation/lpips", initial_metrics["lpips"], 0)
    writer.add_scalar("validation/best_psnr", initial_metrics["psnr"], 0)
    writer.add_scalar("early_stopping/validations_without_improvement", 0, 0)
    log_fixed_samples(
        model,
        val_dataset,
        fixed_samples,
        device,
        writer,
        0,
        sample_output_dir,
    )

    global_step = 0
    attempted_windows = 0
    amp_skips = 0
    best_psnr = initial_metrics["psnr"]
    early_stop_state = {
        "reference_psnr": best_psnr,
        "validations_without_improvement": 0,
        "stopped_early": False,
        "stop_reason": None,
    }
    save_checkpoint(
        args.save_dir / "best_psnr.pth",
        model,
        optimizer,
        scheduler,
        scaler,
        global_step,
        attempted_windows,
        amp_skips,
        best_psnr,
        args,
        hashes,
        early_stop_state,
    )

    train_iterator = iter(train_loader)
    loss_since_log = 0.0
    successful_windows_since_log = 0
    gradient_probe_done = False
    maximum_attempts = args.max_steps + max(50, math.ceil(args.max_steps * 0.01))

    while global_step < args.max_steps:
        model.train()
        if attempted_windows >= maximum_attempts:
            raise RuntimeError("Too many AMP-skipped optimizer windows")
        optimizer.zero_grad(set_to_none=True)
        window_loss = 0.0
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            lq = to_01(batch["lq"].to(device, non_blocking=True))
            gt = to_01(batch["gt"].to(device, non_blocking=True))
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                pred = model(normalize_ffa(lq))
                loss = F.l1_loss(pred, gt)
                scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            window_loss += float(loss.detach().cpu())

        scaler.unscale_(optimizer)
        probe = None
        if not gradient_probe_done:
            gradients = [
                parameter.grad.detach()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            maximum_gradient = max(
                [float(gradient.abs().max().cpu()) for gradient in gradients] or [0.0]
            )
            all_finite = all(torch.isfinite(gradient).all() for gradient in gradients)
            probe = {
                "event": "first_gradient",
                "max_abs": maximum_gradient,
                "all_finite": bool(all_finite),
                "gradient_tensor_count": len(gradients),
            }

        scale_before = scaler.get_scale()
        attempted_windows += 1
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() < scale_before:
            amp_skips += 1
            continue

        if probe is not None:
            append_jsonl(history_path, probe)
            print("[Gradient] " + json.dumps(probe, sort_keys=True))
            if probe["max_abs"] <= 0.0 or not probe["all_finite"]:
                raise RuntimeError("Gradient probe failed")
            gradient_probe_done = True

        scheduler.step()
        global_step += 1
        loss_since_log += window_loss / args.grad_accum_steps
        successful_windows_since_log += 1

        if global_step % args.log_every_steps == 0 or global_step == 1:
            train_event = {
                "event": "train",
                "global_step": global_step,
                "attempted_optimizer_windows": attempted_windows,
                "amp_skips": amp_skips,
                "mean_l1_since_log": loss_since_log
                / max(successful_windows_since_log, 1),
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_jsonl(history_path, train_event)
            print("[Train] " + json.dumps(train_event, sort_keys=True))
            writer.add_scalar(
                "train/l1", train_event["mean_l1_since_log"], global_step
            )
            writer.add_scalar("train/lr", train_event["lr"], global_step)
            writer.add_scalar(
                "train/amp_skips_cumulative", amp_skips, global_step
            )
            loss_since_log = 0.0
            successful_windows_since_log = 0

        if global_step % args.checkpoint_every_steps == 0:
            save_checkpoint(
                args.save_dir / "last.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                global_step,
                attempted_windows,
                amp_skips,
                best_psnr,
                args,
                hashes,
                early_stop_state,
            )

        should_evaluate = (
            global_step % args.eval_every_steps == 0
            or global_step == args.max_steps
        )
        if should_evaluate:
            metrics = evaluate(
                model, val_loader, lpips_model, device, args.eval_max_batches
            )
            if metrics["psnr"] >= (
                early_stop_state["reference_psnr"] + EARLY_STOP_MIN_DELTA
            ):
                early_stop_state["reference_psnr"] = metrics["psnr"]
                early_stop_state["validations_without_improvement"] = 0
            elif global_step >= EARLY_STOP_MIN_STEPS:
                early_stop_state["validations_without_improvement"] += 1
            else:
                early_stop_state["validations_without_improvement"] = 0

            if (
                not args.smoke_test
                and global_step >= EARLY_STOP_MIN_STEPS
                and early_stop_state["validations_without_improvement"]
                >= EARLY_STOP_PATIENCE
            ):
                early_stop_state["stopped_early"] = True
                early_stop_state["stop_reason"] = "validation_psnr_plateau"

            validation_event = {
                "event": "validation",
                "global_step": global_step,
                **metrics,
                "early_stopping": dict(early_stop_state),
            }
            append_jsonl(history_path, validation_event)
            print("[Eval] " + json.dumps(validation_event, sort_keys=True))
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                save_checkpoint(
                    args.save_dir / "best_psnr.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    global_step,
                    attempted_windows,
                    amp_skips,
                    best_psnr,
                    args,
                    hashes,
                    early_stop_state,
                )
            writer.add_scalar("validation/psnr", metrics["psnr"], global_step)
            writer.add_scalar("validation/ssim", metrics["ssim"], global_step)
            writer.add_scalar("validation/lpips", metrics["lpips"], global_step)
            writer.add_scalar("validation/best_psnr", best_psnr, global_step)
            writer.add_scalar(
                "early_stopping/validations_without_improvement",
                early_stop_state["validations_without_improvement"],
                global_step,
            )
            log_fixed_samples(
                model,
                val_dataset,
                fixed_samples,
                device,
                writer,
                global_step,
                sample_output_dir,
            )
            if early_stop_state["stopped_early"]:
                print("[Early stop] " + json.dumps(early_stop_state, sort_keys=True))
                break

    save_checkpoint(
        args.save_dir / "last.pth",
        model,
        optimizer,
        scheduler,
        scaler,
        global_step,
        attempted_windows,
        amp_skips,
        best_psnr,
        args,
        hashes,
        early_stop_state,
    )
    if not gradient_probe_done:
        raise RuntimeError("Training completed without a gradient probe")
    if not args.smoke_test and amp_skips / max(attempted_windows, 1) > 0.001:
        raise RuntimeError("AMP skip fraction exceeds 0.1%")
    strict_reload(args.save_dir / "best_psnr.pth", args.architecture)
    strict_reload(args.save_dir / "last.pth", args.architecture)
    writer.close()
    print(
        "FFA_BASELINE_PROTOCOL_V2_PASS "
        + json.dumps(
            {
                "global_step": global_step,
                "attempted_optimizer_windows": attempted_windows,
                "amp_skips": amp_skips,
                "best_psnr": best_psnr,
                "early_stopping": early_stop_state,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
