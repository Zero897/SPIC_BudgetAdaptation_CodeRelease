"""Train an RGB-only residual adapter on a frozen, already fine-tuned backbone."""

import argparse
import hashlib
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
from tqdm import tqdm

from cm_dehazeformer_v2_loss import CrossModalResidualLoss, simple_ssim_loss
from post_ft_rgb_residual_v1 import BACKBONES, build_model
try:
    from data.dataset_crossmodal import CrossModalUAVDataset
except ModuleNotFoundError:
    from dataset_crossmodal import CrossModalUAVDataset


EXPECTED_BASE_PARAMETERS = {"restormer": 13_926_281, "ffa": 4_455_913}
EXPECTED_ADAPTER_PARAMETERS = {"restormer": 43_637, "ffa": 46_254}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed, deterministic):
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


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def to_01(value):
    return ((value + 1.0) / 2.0).clamp(0, 1)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


class FixedValidationDataset(Dataset):
    def __init__(self, lq_dir, gt_dir, size):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.size = size
        self.names = sorted(
            path.name
            for path in lq_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.names:
            raise RuntimeError("No validation images found")
        for name in self.names:
            stem = Path(name).stem
            self._find(gt_dir, stem)

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _find(folder, stem):
        matches = [
            path
            for path in folder.glob(stem + ".*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(matches) != 1:
            raise FileNotFoundError("Expected one pair for {} in {}".format(stem, folder))
        return matches[0]

    def _rgb(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image).permute(2, 0, 1).float()

    def __getitem__(self, index):
        name = self.names[index]
        stem = Path(name).stem
        return {
            "lq": self._rgb(self.lq_dir / name),
            "gt": self._rgb(self._find(self.gt_dir, stem)),
            "name": stem,
        }


@torch.no_grad()
def evaluate(model, loader, lpips_model, device, max_batches, adapter_enabled):
    model.eval()
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    count = 0
    gate_sum = 0.0
    gate_square_sum = 0.0
    gate_count = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="Eval", leave=False)):
        if max_batches >= 0 and batch_index >= max_batches:
            break
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        prediction, aux = model(
            lq * 2.0 - 1.0,
            None,
            return_aux=True,
            adapter_enabled=adapter_enabled,
        )
        mse = F.mse_loss(prediction, gt, reduction="none").mean(dim=(1, 2, 3))
        totals["psnr"] += (20.0 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))).sum().item()
        totals["ssim"] += float(1.0 - simple_ssim_loss(prediction, gt)) * lq.shape[0]
        totals["lpips"] += (
            lpips_model(prediction * 2.0 - 1.0, gt * 2.0 - 1.0).view(-1).sum().item()
        )
        gate = aux.get("gate")
        if gate is not None:
            gate = gate.detach().float()
            gate_sum += gate.sum().item()
            gate_square_sum += gate.square().sum().item()
            gate_count += gate.numel()
        count += lq.shape[0]
    if count == 0:
        raise RuntimeError("Validation produced zero samples")
    result = {"count": count, **{key: value / count for key, value in totals.items()}}
    if gate_count:
        gate_mean = gate_sum / gate_count
        result.update(
            {
                "gate_mean": gate_mean,
                "gate_std": math.sqrt(max(gate_square_sum / gate_count - gate_mean ** 2, 0.0)),
                "gamma": float(aux["gamma"].detach().cpu()),
                "gamma_raw": float(model.gamma.detach().cpu()),
                "delta_scale": float(model.refiner.get_delta_scale().detach().cpu()),
            }
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--train_root", type=Path, required=True)
    parser.add_argument("--val_lq", type=Path, required=True)
    parser.add_argument("--val_gt", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init_gamma", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_max_batches", type=int, default=-1)
    parser.add_argument("--eval_max_batches", type=int, default=-1)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.size != 256 or args.batch * args.grad_accum_steps != 16:
        raise ValueError("Protocol requires size=256 and effective batch 16")
    expected_gamma = 0.08 if args.backbone == "ffa" else 0.01
    if abs(args.lr - 1e-4) > 1e-12 or abs(args.init_gamma - expected_gamma) > 1e-12:
        raise ValueError(
            "Protocol requires lr=1e-4 and init_gamma={} for {}".format(
                expected_gamma, args.backbone
            )
        )
    for path in (args.val_lq, args.val_gt):
        normalized = str(path.resolve()).replace("\\", "/").lower()
        if "/test/" in normalized or "synthetic_smoke_test" in normalized:
            raise ValueError("Test path is forbidden: {}".format(path))
    run_path = str(args.save_dir).lower()
    if args.backbone not in run_path or "seed{}".format(args.seed) not in run_path:
        raise ValueError("save_dir must contain backbone and seed")
    if args.save_dir.exists() and any(args.save_dir.iterdir()):
        raise FileExistsError("Refusing nonempty save_dir: {}".format(args.save_dir))
    if args.smoke_test:
        if args.train_max_batches <= 0 or args.eval_max_batches <= 0 or "smoke" not in run_path:
            raise ValueError("Smoke requires bounded loaders and a smoke save_dir")
    elif (
        args.epochs != 15
        or args.eval_every != 3
        or args.train_max_batches != -1
        or args.eval_max_batches != -1
    ):
        raise ValueError("Pilot requires 15 full epochs and full validation every 3 epochs")


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_psnr, config):
    torch.save(
        {
            "epoch": epoch,
            "global_step": step,
            "best_psnr": best_psnr,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": config["args"],
        },
        path,
    )


def strict_reload(args, checkpoint_path):
    reloaded = build_model(args.backbone, args.architecture, args.init_gamma)
    state = torch.load(str(checkpoint_path), map_location="cpu")["model"]
    incompatible = reloaded.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint reload failed")
    if any(not torch.isfinite(value).all() for value in reloaded.state_dict().values()):
        raise RuntimeError("Non-finite checkpoint tensor")


def main():
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed, args.deterministic)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU is required")

    model = build_model(args.backbone, args.architecture, args.init_gamma)
    missing, unexpected = model.load_base_weights_strict(args.pretrained)
    counts = model.parameter_counts()
    if counts["base_total"] != EXPECTED_BASE_PARAMETERS[args.backbone]:
        raise RuntimeError("Unexpected backbone parameter count")
    expected_adapter = EXPECTED_ADAPTER_PARAMETERS[args.backbone]
    if (
        counts["base_trainable"] != 0
        or counts["adapter_total"] != expected_adapter
        or counts["adapter_trainable"] != expected_adapter
    ):
        raise RuntimeError("The FT backbone must be frozen and the residual framework trainable")
    model = model.to(device)

    train_generator = torch.Generator().manual_seed(args.seed)
    eval_generator = torch.Generator().manual_seed(args.seed + 1)
    train_dataset = CrossModalUAVDataset(
        str(args.train_root), size=args.size, use_depth=False, augment=True
    )
    val_dataset = FixedValidationDataset(args.val_lq, args.val_gt, args.size)
    if len(train_dataset) != 12000 or len(val_dataset) != 1469:
        raise RuntimeError("Protocol requires 12000 train and 1469 validation samples")
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

    script_path = Path(__file__).resolve()
    model_path = script_path.with_name("post_ft_rgb_residual_v1.py")
    loss_path = script_path.with_name("cm_dehazeformer_v2_loss.py")
    dataset_path = script_path.parent / "data" / "dataset_crossmodal.py"
    if not dataset_path.is_file():
        dataset_path = script_path.with_name("dataset_crossmodal.py")
    val_manifest = args.val_lq.parent / "generation_manifest.jsonl"
    config = {
        "protocol": "post_ft_rgb_residual_v1",
        "backbone": args.backbone,
        "base_identity": "validation_selected_already_fine_tuned_checkpoint",
        "base_training": "frozen_all_epochs",
        "residual_framework": "rgb_only_gamma_ffa_gate_restormer_ungated",
        "input_channels": "pred_base_rgb_plus_lq_rgb",
        "ir_used": False,
        "physics_used": False,
        "spatial_gate": args.backbone == "ffa",
        "attnres": False,
        "test_access": "forbidden",
        "selection_metric": "validation_psnr",
        "training_precision": "bf16_autocast_fp32_optimizer",
        "base_load": {"missing": missing, "unexpected": unexpected, "strict": True},
        "parameter_counts": counts,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "hashes": {
            "trainer": sha256_file(script_path),
            "model": sha256_file(model_path),
            "loss": sha256_file(loss_path),
            "dataset": sha256_file(dataset_path),
            "architecture": sha256_file(args.architecture),
            "pretrained": sha256_file(args.pretrained),
            "validation_manifest": sha256_file(val_manifest),
        },
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

    try:
        import lpips
    except Exception as error:
        raise RuntimeError("lpips is required") from error
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    baseline = evaluate(model, val_loader, lpips_model, device, args.eval_max_batches, False)
    baseline_event = {"event": "frozen_ft_baseline", "backbone": args.backbone, **baseline}
    append_jsonl(history_path, baseline_event)
    print("[Baseline] " + json.dumps(baseline_event, sort_keys=True))

    enabled_initial = evaluate(model, val_loader, lpips_model, device, 1, True)
    disabled_initial = evaluate(model, val_loader, lpips_model, device, 1, False)
    if abs(enabled_initial["psnr"] - disabled_initial["psnr"]) > 1e-7:
        raise RuntimeError("Zero-tail residual framework does not start at the exact FT output")

    criterion = CrossModalResidualLoss(
        w_char=1.0,
        w_ssim=0.12,
        w_edge=0.03,
        w_color=0.01,
        w_delta=0.0,
        w_gate=0.0,
        w_not_worse=0.0,
    )
    adapter_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("base.") and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        adapter_parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4
    )
    batches_per_epoch = (
        min(len(train_loader), args.train_max_batches)
        if args.train_max_batches >= 0
        else len(train_loader)
    )
    windows_per_epoch = math.ceil(batches_per_epoch / args.grad_accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * windows_per_epoch, eta_min=1e-7
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    best_psnr = float("-inf")
    global_step = 0
    gradient_probe_done = False
    gate_gradient_seen = False
    gamma_gradient_seen = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.freeze_base()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        processed = 0
        for batch_index, batch in enumerate(tqdm(train_loader, desc="Epoch {}/{}".format(epoch, args.epochs))):
            if args.train_max_batches >= 0 and batch_index >= args.train_max_batches:
                break
            lq = batch["lq"].to(device, non_blocking=True)
            gt = to_01(batch["gt"].to(device, non_blocking=True))
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                prediction, aux = model(lq, None, return_aux=True, adapter_enabled=True)
                loss, _ = criterion(prediction, gt, aux=aux)
                scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            loss_sum += float(loss.detach().cpu())
            processed += 1
            if processed % args.grad_accum_steps and processed != batches_per_epoch:
                continue

            base_grad = max(
                [float(parameter.grad.detach().abs().max().cpu()) for parameter in model.base.parameters() if parameter.grad is not None]
                or [0.0]
            )
            adapter_grad = max(
                [
                    float(parameter.grad.detach().abs().max().cpu())
                    for name, parameter in model.named_parameters()
                    if not name.startswith("base.") and parameter.grad is not None
                ]
                or [0.0]
            )
            if model.refiner.gate is not None:
                gate_gradient_seen = gate_gradient_seen or any(
                    parameter.grad is not None and parameter.grad.detach().abs().max().item() > 0
                    for parameter in model.refiner.gate.parameters()
                )
            gamma_gradient_seen = gamma_gradient_seen or (
                model.gamma.grad is not None and model.gamma.grad.detach().abs().item() > 0
            )
            if not gradient_probe_done:
                if base_grad != 0.0 or adapter_grad <= 0.0 or not math.isfinite(adapter_grad):
                    raise RuntimeError("Frozen-base residual gradient isolation failed")
                print("[Gradient] " + json.dumps({"base_max_abs": base_grad, "adapter_max_abs": adapter_grad}, sort_keys=True))
                gradient_probe_done = True

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter_parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        train_event = {
            "event": "train_epoch",
            "backbone": args.backbone,
            "epoch": epoch,
            "global_step": global_step,
            "processed_batches": processed,
            "mean_loss": loss_sum / max(processed, 1),
            "gate_gradient_seen": gate_gradient_seen,
            "gamma_gradient_seen": gamma_gradient_seen,
        }
        append_jsonl(history_path, train_event)
        print("[Train] " + json.dumps(train_event, sort_keys=True))

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, lpips_model, device, args.eval_max_batches, True)
            event = {
                "event": "validation",
                "backbone": args.backbone,
                "epoch": epoch,
                "global_step": global_step,
                **metrics,
            }
            append_jsonl(history_path, event)
            print("[Eval] " + json.dumps(event, sort_keys=True))
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                save_checkpoint(args.save_dir / "best_psnr.pth", model, optimizer, scheduler, epoch, global_step, best_psnr, config)
        save_checkpoint(args.save_dir / "last.pth", model, optimizer, scheduler, epoch, global_step, best_psnr, config)

    if not gradient_probe_done or not gamma_gradient_seen:
        raise RuntimeError("Residual framework and gamma did not receive gradients")
    if args.backbone == "ffa" and not gate_gradient_seen:
        raise RuntimeError("FFA spatial gate did not receive gradients")
    strict_reload(args, args.save_dir / "best_psnr.pth")
    strict_reload(args, args.save_dir / "last.pth")
    print("POST_FT_RGB_RESIDUAL_V1_PASS backbone={} seed={}".format(args.backbone, args.seed))


if __name__ == "__main__":
    main()
