"""Train a clean Restormer baseline on Protocol-V2 train/validation data."""

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


EXPECTED_PARAMETERS = 13_926_281
EXPECTED_ARCHITECTURE_SHA256 = "3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a"
EXPECTED_DATASET_SHA256 = "5e8a7c975a3f90c54c393c53ab7528eacc70e5b95aff4b8d9009107456daa34d"
EXPECTED_MANIFEST_SHA256 = "943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128"
FULL_EPOCHS = 100
EVAL_EVERY = 2
EARLY_STOP_MIN_EPOCH = 50
EARLY_STOP_PATIENCE = 4
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


def model_parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
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


def build_model(architecture: Path):
    module = load_module("restormer_protocol_v2_architecture", architecture)
    model = module.Restormer(
        inp_channels=3,
        out_channels=3,
        dim=40,
        num_blocks=[3, 4, 4, 6],
        num_refinement_blocks=3,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            "Restormer parameter mismatch: expected={}, actual={}".format(
                EXPECTED_PARAMETERS, parameter_count
            )
        )
    return model


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
        pred = model(lq).float().clamp(0, 1)
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
    epoch,
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
            pred = model(lq).float().clamp(0, 1)
            rows.append(torch.cat((lq[0], pred[0], gt[0]), dim=2).cpu())
        panel = torch.cat(rows, dim=1)
        writer.add_image(
            "samples/{}/LQ_Restored_GT".format(level), panel, epoch
        )
        save_rgb_tensor(
            output_dir / "epoch_{:03d}_{}.png".format(epoch, level), panel
        )
    writer.flush()
    print(
        "[Samples] epoch={} levels={} per_level={} order=LQ|Restored|GT".format(
            epoch, ",".join(SAMPLE_LEVELS), SAMPLES_PER_LEVEL
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
    epoch,
    global_step,
    attempted_steps,
    amp_skips,
    best_psnr,
    args,
    hashes,
    early_stop_state,
):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "attempted_optimizer_steps": attempted_steps,
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
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
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
    parser.add_argument("--dataset_source", type=Path, required=True)
    parser.add_argument("--train_root", type=Path, required=True)
    parser.add_argument("--val_lq", type=Path, required=True)
    parser.add_argument("--val_gt", type=Path, required=True)
    parser.add_argument("--val_manifest", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=FULL_EPOCHS)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--eval_max_batches", type=int, default=-1)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    if args.size != 256 or args.batch != 4 or args.grad_accum_steps != 4:
        raise ValueError("Protocol requires size=256, batch=4, grad_accum_steps=4")
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
        if args.epochs != 1 or args.max_train_batches <= 0 or args.eval_max_batches <= 0:
            raise ValueError("Smoke requires one epoch and bounded train/validation")
    else:
        expected = {
            "epochs": FULL_EPOCHS,
            "eval_every": EVAL_EVERY,
            "eval_max_batches": -1,
            "max_train_batches": -1,
            "seed": 3407,
        }
        for key, value in expected.items():
            if getattr(args, key) != value:
                raise ValueError("Full protocol requires {}={}".format(key, value))
        if not math.isclose(args.lr, 2e-4):
            raise ValueError("Full protocol requires lr=2e-4")


def main():
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed, args.deterministic)

    hashes = {
        "trainer": sha256_file(Path(__file__).resolve()),
        "architecture": sha256_file(args.architecture),
        "dataset_source": sha256_file(args.dataset_source),
        "validation_manifest": sha256_file(args.val_manifest),
    }
    expected_hashes = {
        "architecture": EXPECTED_ARCHITECTURE_SHA256,
        "dataset_source": EXPECTED_DATASET_SHA256,
        "validation_manifest": EXPECTED_MANIFEST_SHA256,
    }
    for label, expected in expected_hashes.items():
        if hashes[label] != expected:
            raise RuntimeError(
                "{} hash mismatch: expected={}, actual={}".format(
                    label, expected, hashes[label]
                )
            )

    args.save_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Protocol requires CUDA BF16 support")

    dataset_module = load_module("restormer_protocol_v2_dataset", args.dataset_source)
    train_dataset = dataset_module.CrossModalUAVDataset(
        str(args.train_root), size=args.size, use_depth=False, augment=True
    )
    val_dataset = FixedValidationDataset(args.val_lq, args.val_gt, args.size)
    if len(train_dataset) != 12000 or len(val_dataset) != 1469:
        raise RuntimeError(
            "Protocol V2 requires 12000 train and 1469 validation samples; got {} and {}".format(
                len(train_dataset), len(val_dataset)
            )
        )
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
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=eval_generator,
    )

    model = build_model(args.architecture).to(device)
    initial_parameter_sha256 = model_parameter_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    try:
        import lpips
    except Exception as error:
        raise RuntimeError("lpips is required for Protocol-V2 validation") from error
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    config = {
        "experiment_id": "RESTORMER_BASE_V2",
        "initialization": "random_from_epoch_1",
        "seed": args.seed,
        "selection_metric": "validation_psnr",
        "test_access": "forbidden",
        "training_precision": "bf16_autocast_fp32_optimizer",
        "effective_batch": args.batch * args.grad_accum_steps,
        "initial_parameter_sha256": initial_parameter_sha256,
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
        "early_stopping": {
            "metric": "validation_psnr",
            "mode": "max",
            "min_epoch": EARLY_STOP_MIN_EPOCH,
            "patience_validations": EARLY_STOP_PATIENCE,
            "min_delta_db": EARLY_STOP_MIN_DELTA,
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

    initial_metrics = evaluate(
        model, val_loader, lpips_model, device, args.eval_max_batches
    )
    initial_event = {
        "event": "initial_validation",
        "epoch": 0,
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

    best_psnr = initial_metrics["psnr"]
    global_step = 0
    attempted_steps = 0
    amp_skips = 0
    gradient_probe_done = False
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
        0,
        global_step,
        attempted_steps,
        amp_skips,
        best_psnr,
        args,
        hashes,
        early_stop_state,
    )

    final_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        processed_batches = 0
        successful_steps = 0
        batches_in_window = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(
            tqdm(train_loader, desc="Epoch {}/{}".format(epoch, args.epochs))
        ):
            if args.max_train_batches >= 0 and batch_index >= args.max_train_batches:
                break
            lq = to_01(batch["lq"].to(device, non_blocking=True))
            gt = to_01(batch["gt"].to(device, non_blocking=True))
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                pred = model(lq).float().clamp(0, 1)
                loss = F.l1_loss(pred, gt)
                scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            loss_sum += float(loss.detach().cpu())
            processed_batches += 1
            batches_in_window += 1
            if batches_in_window < args.grad_accum_steps:
                continue

            scaler.unscale_(optimizer)

            probe = None
            if not gradient_probe_done:
                gradients = [
                    parameter.grad.detach()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                probe = {
                    "event": "first_gradient",
                    "max_abs": max(
                        [float(gradient.abs().max().cpu()) for gradient in gradients]
                        or [0.0]
                    ),
                    "all_finite": bool(
                        all(torch.isfinite(gradient).all() for gradient in gradients)
                    ),
                    "gradient_tensor_count": len(gradients),
                }

            scale_before = scaler.get_scale()
            attempted_steps += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            batches_in_window = 0
            if scaler.get_scale() < scale_before:
                amp_skips += 1
                continue

            if probe is not None:
                append_jsonl(history_path, probe)
                print("[Gradient] " + json.dumps(probe, sort_keys=True))
                if probe["max_abs"] <= 0.0 or not probe["all_finite"]:
                    raise RuntimeError("Gradient probe failed")
                gradient_probe_done = True

            global_step += 1
            successful_steps += 1

        if batches_in_window != 0:
            raise RuntimeError(
                "Incomplete gradient accumulation window: batches={}".format(
                    batches_in_window
                )
            )

        if successful_steps == 0:
            raise RuntimeError("Epoch {} produced zero successful steps".format(epoch))
        scheduler.step()
        final_epoch = epoch
        train_event = {
            "event": "train_epoch",
            "epoch": epoch,
            "global_step": global_step,
            "successful_optimizer_steps": successful_steps,
            "attempted_optimizer_steps_cumulative": attempted_steps,
            "amp_skips_cumulative": amp_skips,
            "mean_l1": loss_sum / processed_batches,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_jsonl(history_path, train_event)
        print("[Train] " + json.dumps(train_event, sort_keys=True))
        writer.add_scalar("train/l1", train_event["mean_l1"], epoch)
        writer.add_scalar("train/lr", train_event["lr"], epoch)
        writer.add_scalar("train/global_step", global_step, epoch)
        writer.add_scalar("train/amp_skips_cumulative", amp_skips, epoch)

        should_evaluate = (
            args.smoke_test
            or epoch % args.eval_every == 0
            or epoch == args.epochs
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
            elif epoch >= EARLY_STOP_MIN_EPOCH:
                early_stop_state["validations_without_improvement"] += 1
            else:
                early_stop_state["validations_without_improvement"] = 0

            if (
                not args.smoke_test
                and epoch >= EARLY_STOP_MIN_EPOCH
                and early_stop_state["validations_without_improvement"]
                >= EARLY_STOP_PATIENCE
            ):
                early_stop_state["stopped_early"] = True
                early_stop_state["stop_reason"] = "validation_psnr_plateau"

            validation_event = {
                "event": "validation",
                "epoch": epoch,
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
                    epoch,
                    global_step,
                    attempted_steps,
                    amp_skips,
                    best_psnr,
                    args,
                    hashes,
                    early_stop_state,
                )
            writer.add_scalar("validation/psnr", metrics["psnr"], epoch)
            writer.add_scalar("validation/ssim", metrics["ssim"], epoch)
            writer.add_scalar("validation/lpips", metrics["lpips"], epoch)
            writer.add_scalar("validation/best_psnr", best_psnr, epoch)
            writer.add_scalar(
                "early_stopping/validations_without_improvement",
                early_stop_state["validations_without_improvement"],
                epoch,
            )
            log_fixed_samples(
                model,
                val_dataset,
                fixed_samples,
                device,
                writer,
                epoch,
                sample_output_dir,
            )

        save_checkpoint(
            args.save_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            attempted_steps,
            amp_skips,
            best_psnr,
            args,
            hashes,
            early_stop_state,
        )
        if early_stop_state["stopped_early"]:
            print("[Early stop] " + json.dumps(early_stop_state, sort_keys=True))
            break

    if not gradient_probe_done:
        raise RuntimeError("Training completed without a gradient probe")
    if not args.smoke_test and amp_skips / max(attempted_steps, 1) > 0.001:
        raise RuntimeError("AMP skip fraction exceeds 0.1%")
    final_parameter_sha256 = model_parameter_sha256(model)
    parameter_event = {
        "event": "model_parameter_change",
        "initial_sha256": initial_parameter_sha256,
        "final_sha256": final_parameter_sha256,
        "changed": final_parameter_sha256 != initial_parameter_sha256,
    }
    append_jsonl(history_path, parameter_event)
    print("[Parameter change] " + json.dumps(parameter_event, sort_keys=True))
    if not parameter_event["changed"]:
        raise RuntimeError("Model parameters did not change")
    strict_reload(args.save_dir / "best_psnr.pth", args.architecture)
    strict_reload(args.save_dir / "last.pth", args.architecture)
    writer.close()
    print(
        "RESTORMER_BASELINE_PROTOCOL_V2_PASS "
        + json.dumps(
            {
                "final_epoch": final_epoch,
                "global_step": global_step,
                "attempted_optimizer_steps": attempted_steps,
                "amp_skips": amp_skips,
                "best_psnr": best_psnr,
                "early_stopping": early_stop_state,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
