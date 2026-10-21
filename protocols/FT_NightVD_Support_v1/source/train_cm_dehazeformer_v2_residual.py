"""Protocol-v2 training entry point.

Use only the fixed validation set for checkpoint selection. Commands for B0,
A0, P0, A2, A3, and FT are provided in EXPERIMENT_COMMANDS_V2.md.
"""

# train_cm_dehazeformer_v2_residual.py
# Full training script: imports V2 model/loss and uses fixed validation lq/ir/gt.
# For brevity, this is a compact version adapted from train_cm_dehazeformer_v1.py.

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset_crossmodal import CrossModalUAVDataset
from crossmodal_dehazeformer_residual_v2 import CrossModalDehazeFormerResidualV2, load_dehazeformer_base_weights, load_v2_checkpoint
from cm_dehazeformer_v2_loss import CrossModalResidualLoss, simple_ssim_loss

try:
    import lpips
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False


LOSS_PROFILES = {
    "common": {
        "w_char": 1.0,
        "w_ssim": 0.12,
        "w_edge": 0.03,
        "w_color": 0.01,
        "w_delta": 0.0,
        "w_gate": 0.0,
        "w_not_worse": 0.0,
    },
    "legacy_adapter": {
        "w_char": 1.0,
        "w_ssim": 0.12,
        "w_edge": 0.03,
        "w_color": 0.01,
        "w_delta": 0.001,
        "w_gate": 0.0005,
        "w_not_worse": 0.2,
    },
}


def seed_everything(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def to_01(x): return ((x + 1.0) / 2.0).clamp(0, 1)

class FixedEvalSmokeDataset(Dataset):
    def __init__(self, lq_dir, ir_dir, gt_dir, size=256):
        self.lq_dir, self.ir_dir, self.gt_dir, self.size = lq_dir, ir_dir, gt_dir, size
        self.exts = [".png",".jpg",".jpeg",".bmp",".tif",".tiff",".PNG",".JPG",".JPEG",".BMP",".TIF",".TIFF"]
        self.names = sorted([f for f in os.listdir(lq_dir) if not f.startswith(".") and os.path.splitext(f)[1] in self.exts])
        print(f"[FixedEval] {len(self.names)} images | {lq_dir}")
    def __len__(self): return len(self.names)
    def _find(self, folder, stem):
        for e in self.exts:
            p = os.path.join(folder, stem + e)
            if os.path.exists(p): return p
        return None
    def _rgb(self, p):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (self.size,self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).permute(2,0,1).float() * 2 - 1
    def _ir(self, p):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None: raise FileNotFoundError(p)
        img = np.squeeze(img)
        if img.ndim == 3: img = img.mean(axis=2)
        img = img.astype(np.float32)
        img = (img - img.min())/(img.max()-img.min()+1e-8) if img.max() > 255 else img/255.0
        img = cv2.resize(img, (self.size,self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).float().unsqueeze(0) * 2 - 1
    def __getitem__(self, idx):
        name = self.names[idx]; stem = os.path.splitext(name)[0]
        irp, gtp = self._find(self.ir_dir, stem), self._find(self.gt_dir, stem)
        if irp is None or gtp is None: raise FileNotFoundError(stem)
        return {"lq": self._rgb(os.path.join(self.lq_dir, name)), "ir": self._ir(irp), "gt": self._rgb(gtp), "name": stem, "txt": ""}

@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval() if HAS_LPIPS else None
    psnr_sum=ssim_sum=lpips_sum=0.0; n=0
    for bi,batch in enumerate(tqdm(loader, desc="Eval", leave=False)):
        if max_batches is not None and bi >= max_batches: break
        lq, gt, ir = to_01(batch["lq"].to(device)), to_01(batch["gt"].to(device)), batch["ir"].to(device)
        pred = model(lq, ir).clamp(0,1)
        mse = F.mse_loss(pred, gt, reduction="none").mean(dim=[1,2,3])
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        ssim = 1.0 - simple_ssim_loss(pred, gt)
        bs = lq.size(0); psnr_sum += psnr.sum().item(); ssim_sum += float(ssim.item()) * bs
        if lpips_fn is not None: lpips_sum += lpips_fn(pred*2-1, gt*2-1).view(-1).sum().item()
        n += bs
    model.train()
    return {"psnr": psnr_sum/max(n,1), "ssim": ssim_sum/max(n,1), "lpips": lpips_sum/max(n,1) if lpips_fn is not None else -1}

def build_optimizer(model, lr_base, lr_new):
    base, new = [], []
    for name,p in model.named_parameters():
        if name.startswith("base."):
            if lr_base > 0:
                base.append(p)
        elif p.requires_grad and lr_new > 0:
            new.append(p)
    print(f"[Optimizer] base params={sum(p.numel() for p in base):,}, lr={lr_base}")
    print(f"[Optimizer] new  params={sum(p.numel() for p in new):,}, lr={lr_new}")
    groups = []
    if base:
        groups.append({"params": base, "lr": lr_base})
    if new:
        groups.append({"params": new, "lr": lr_new})
    if not groups:
        raise ValueError("No parameters selected for optimization.")
    return torch.optim.AdamW(groups, betas=(0.9,0.999), weight_decay=1e-4)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_counts(model):
    base_total = sum(p.numel() for p in model.base.parameters())
    adapter_total = sum(
        p.numel() for name, p in model.named_parameters()
        if not name.startswith("base.")
    )
    base_trainable = sum(
        p.numel() for p in model.base.parameters() if p.requires_grad
    )
    adapter_trainable = sum(
        p.numel() for name, p in model.named_parameters()
        if not name.startswith("base.") and p.requires_grad
    )
    return {
        "base_total": base_total,
        "adapter_total_stored": adapter_total,
        "base_trainable": base_trainable,
        "adapter_trainable": adapter_trainable,
        "trainable_total": base_trainable + adapter_trainable,
    }


def validate_protocol(args):
    if args.batch <= 0:
        raise ValueError("batch must be positive.")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive.")

    for key in ("val_lq", "val_ir", "val_gt"):
        normalized = str(Path(getattr(args, key)).resolve()).replace("\\", "/").lower()
        if (
            "/test/" in normalized
            or "test/smoke" in normalized
            or "synthetic_smoke_test" in normalized
        ):
            raise ValueError(
                f"{key} points to a test directory: {getattr(args, key)}"
            )

    run_path = str(Path(args.save_dir).resolve()).replace("\\", "/").lower()
    if args.experiment_id.lower() not in run_path:
        raise ValueError(
            f"save_dir must contain experiment id {args.experiment_id!r}: {args.save_dir}"
        )
    if f"seed{args.seed}" not in run_path:
        raise ValueError(
            f"save_dir must contain seed{args.seed}: {args.save_dir}"
        )

    expected = {
        "B0": ("frozen_baseline", "wo_ir_physics", True, False, "fscatterres"),
        "A0": ("adapter", "wo_ir_physics", True, False, "fscatterres"),
        "P0": ("adapter", "wo_ir_physics", True, False, "plain"),
        "A2": ("adapter", "wo_ir", True, True, "fscatterres"),
        "A3": ("adapter", "full", True, True, "fscatterres"),
        "FT": ("base_only", "wo_ir_physics", True, False, "fscatterres"),
    }
    actual = (
        args.train_mode,
        args.ablation,
        args.wo_gate,
        args.use_attnres,
        args.adapter_variant,
    )
    if actual != expected[args.experiment_id]:
        raise ValueError(
            f"Configuration mismatch for {args.experiment_id}: "
            f"expected {expected[args.experiment_id]}, got {actual}"
        )

    expected_width = 25 if args.experiment_id == "P0" else 24
    if args.refiner_width != expected_width or args.refiner_blocks != 4:
        raise ValueError(
            f"{args.experiment_id} requires refiner_width={expected_width} "
            "and refiner_blocks=4."
        )

    if args.reset_best:
        raise ValueError("Protocol-v2 does not allow --reset_best.")

    if args.smoke_test:
        if (
            not 1 <= args.epochs <= 2
            or args.train_max_batches <= 0
            or args.eval_max_batches <= 0
            or "smoke" not in run_path
        ):
            raise ValueError(
                "Smoke tests require 1-2 epochs, positive train/eval batch limits, "
                "and 'smoke' in save_dir."
            )
        if args.train_max_batches % args.grad_accum_steps != 0:
            raise ValueError(
                "Smoke-test train_max_batches must be divisible by "
                "grad_accum_steps."
            )
    elif args.train_max_batches != -1 or args.eval_max_batches != -1:
        raise ValueError(
            "Full protocol runs require train_max_batches=-1 and eval_max_batches=-1."
        )

    if args.experiment_id in {"A0", "P0", "A2", "A3"}:
        if args.lr_base != 0 or args.lr_new <= 0:
            raise ValueError("Adapter runs require lr_base=0 and lr_new>0.")
        if args.freeze_base_epochs < args.epochs:
            raise ValueError("Adapter runs require the backbone frozen for every epoch.")
    elif args.experiment_id == "FT":
        if args.lr_base <= 0 or args.lr_new != 0 or args.freeze_base_epochs != 0:
            raise ValueError(
                "FT requires lr_base>0, lr_new=0, and freeze_base_epochs=0."
            )
    elif args.experiment_id == "B0":
        if args.lr_base != 0 or args.lr_new != 0:
            raise ValueError("B0 requires lr_base=0 and lr_new=0.")

    if args.experiment_id != "B0":
        effective_batch = args.batch * args.grad_accum_steps
        if effective_batch != 16:
            raise ValueError(
                "Protocol-v2 trainable runs require effective batch 16; "
                f"got batch={args.batch} * grad_accum_steps="
                f"{args.grad_accum_steps} = {effective_batch}."
            )

    save_dir = Path(args.save_dir)
    if save_dir.exists() and any(save_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Non-empty save_dir requires --resume: {save_dir}"
        )


def validate_resume_checkpoint(args):
    if not args.resume:
        return
    checkpoint = torch.load(args.resume, map_location="cpu")
    previous = checkpoint.get("args", {})
    keys = [
        "experiment_id", "train_mode", "train_root", "val_lq", "val_ir",
        "val_gt", "backbone", "ablation", "wo_gate", "use_attnres",
        "adapter_variant", "refiner_width", "refiner_blocks", "batch",
        "grad_accum_steps", "seed",
        "loss_profile", "smoke_test", "train_max_batches", "eval_max_batches",
    ]
    mismatches = []
    current = vars(args)
    for key in keys:
        if key not in previous or previous[key] != current[key]:
            mismatches.append((key, previous.get(key), current[key]))
    if mismatches:
        raise ValueError(f"Resume configuration mismatch: {mismatches}")


def write_run_metadata(args, counts, loss_config):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    val_manifest = Path(args.val_lq).resolve().parent / "generation_manifest.jsonl"
    source_files = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "crossmodal_dehazeformer_residual_v2.py",
        Path(__file__).resolve().parent / "cm_dehazeformer_v2_loss.py",
        Path(__file__).resolve().parent / "data/dataset_crossmodal.py",
    ]
    metadata = {
        "args": vars(args),
        "command": [sys.executable] + sys.argv,
        "loss_config": loss_config,
        "effective_batch_size": args.batch * args.grad_accum_steps,
        "parameter_counts": counts,
        "pretrained_sha256": sha256_file(args.pretrained),
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parent)): sha256_file(path)
            for path in source_files if path.is_file()
        },
        "validation_manifest_sha256": (
            sha256_file(val_manifest) if val_manifest.is_file() else None
        ),
    }
    output = save_dir / ("resume_config.json" if args.resume else "run_config.json")
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def append_history(save_dir, record):
    with open(Path(save_dir) / "history.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", required=True); p.add_argument("--val_lq", required=True);
    p.add_argument("--val_ir", required=True); p.add_argument("--val_gt", required=True)
    p.add_argument("--save_dir", required=True); p.add_argument("--backbone", default="b", choices=["t","s","b"])
    p.add_argument("--experiment_id", required=True, choices=["B0", "A0", "P0", "A2", "A3", "FT"])
    p.add_argument(
        "--train_mode",
        required=True,
        choices=["frozen_baseline", "adapter", "base_only"],
    )
    p.add_argument("--pretrained", required=True); p.add_argument("--resume", default="")
    p.add_argument("--size", type=int, default=256); p.add_argument("--epochs", type=int, default=80);
    p.add_argument("--batch", type=int, default=8); p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Accumulate this many micro-batches before each optimizer step.",
    )
    p.add_argument("--lr_base", type=float, default=1e-6); p.add_argument("--lr_new", type=float, default=1e-4);
    p.add_argument("--freeze_base_epochs", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=5); p.add_argument("--eval_max_batches", type=int, default=-1); 
    p.add_argument("--train_max_batches", type=int, default=-1)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument(
        "--loss_profile",
        default="common",
        choices=sorted(LOSS_PROFILES),
        help=(
            "Loss weights used by every trainable experiment. Protocol-v2 main "
            "comparisons use 'common'; 'legacy_adapter' reproduces the previous "
            "adapter-specific regularizers."
        ),
    )
    p.add_argument("--init_gamma", type=float, default=0.01)
    p.add_argument(
        "--ablation",
        default="full",
        choices=["full", "wo_ir", "wo_physics", "wo_ir_physics", "wo_gate"],
        help="Component ablation. Legacy value 'wo_gate' is mapped to ablation='full' with --wo_gate."
    )
    p.add_argument(
        "--wo_gate",
        action="store_true",
        help="Disable final spatial gate and use ungated residual output."
    )
    p.add_argument(
        "--use_attnres",
        action="store_true",
        help="Use AttnRes-style depth-wise residual aggregation in residual refiner."
    )
    p.add_argument(
        "--adapter_variant",
        default="fscatterres",
        choices=["fscatterres", "plain"],
    )
    p.add_argument("--reset_best", action="store_true")
    p.add_argument("--refiner_width", type=int, default=24)
    p.add_argument("--refiner_blocks", type=int, default=4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic PyTorch algorithms when available."
    )
    args = p.parse_args()
    if args.ablation == "wo_gate":
        args.ablation = "full"
        args.wo_gate = True

    validate_protocol(args)
    validate_resume_checkpoint(args)

    seed_everything(args.seed, deterministic=args.deterministic)
    print(f"[Seed] seed={args.seed}, deterministic={args.deterministic}")

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        CrossModalUAVDataset(args.train_root, size=args.size, use_depth=False, augment=True),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    if (
        args.train_mode != "frozen_baseline"
        and not args.smoke_test
        and len(train_loader) % args.grad_accum_steps != 0
    ):
        raise ValueError(
            f"Training loader has {len(train_loader)} micro-batches, which is "
            f"not divisible by grad_accum_steps={args.grad_accum_steps}."
        )
    val_loader = DataLoader(
        FixedEvalSmokeDataset(args.val_lq,args.val_ir,args.val_gt,args.size),
        batch_size=max(1,args.batch//2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    model = CrossModalDehazeFormerResidualV2(
        backbone=args.backbone,
        init_gamma=args.init_gamma,
        ablation=args.ablation,
        wo_gate=args.wo_gate,
        use_attnres=args.use_attnres,
        refiner_width=args.refiner_width,
        refiner_blocks=args.refiner_blocks,
        adapter_enabled=args.train_mode == "adapter",
        adapter_variant=args.adapter_variant,
    ).to(device)
    print(
        f"[Model Config] backbone={args.backbone}, "
        f"ablation={args.ablation}, "
        f"wo_gate={args.wo_gate}, "
        f"use_attnres={args.use_attnres}, "
        f"refiner_width={args.refiner_width}, "
        f"refiner_blocks={args.refiner_blocks}, "
        f"adapter_variant={args.adapter_variant}, "
        f"init_gamma={args.init_gamma}, "
        f"train_mode={args.train_mode}"
    )
    if args.resume: load_v2_checkpoint(model, args.resume, strict=True)
    else: load_dehazeformer_base_weights(model, args.pretrained, strict=False)

    if args.train_mode == "adapter":
        model.freeze_base()
    elif args.train_mode == "base_only":
        model.set_adapter_enabled(False)
        model.unfreeze_base()
    else:
        model.set_adapter_enabled(False)
        model.freeze_base()

    loss_weights = dict(LOSS_PROFILES[args.loss_profile])
    criterion = CrossModalResidualLoss(**loss_weights)
    loss_config = {
        "profile": args.loss_profile,
        "weights": loss_weights,
    }
    print(f"[Loss Config] {json.dumps(loss_config, sort_keys=True)}")
    counts = parameter_counts(model)
    print(f"[Parameters] {json.dumps(counts, sort_keys=True)}")
    print(
        f"[Batch Config] micro_batch={args.batch}, "
        f"grad_accum_steps={args.grad_accum_steps}, "
        f"effective_batch={args.batch * args.grad_accum_steps}"
    )
    write_run_metadata(args, counts, loss_config)

    if args.train_mode == "frozen_baseline":
        metrics = evaluate(model, val_loader, device, None)
        record = {
            "event": "frozen_baseline",
            "experiment_id": args.experiment_id,
            "seed": args.seed,
            **metrics,
        }
        append_history(args.save_dir, record)
        with open(Path(args.save_dir) / "baseline_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[B0] {json.dumps(record, sort_keys=True)}")
        writer.close()
        return

    optimizer = build_optimizer(model, args.lr_base, args.lr_new)
    optimizer_steps_per_epoch = len(train_loader) // args.grad_accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * optimizer_steps_per_epoch,
        eta_min=1e-7,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    best_psnr, best_lpips, global_step, start_epoch = 0.0, 999.0, 0, 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
        if "train_generator_state" in ckpt:
            train_generator.set_state(ckpt["train_generator_state"])
        start_epoch = int(ckpt.get("epoch",0))+1; global_step = int(ckpt.get("global_step",0))
        best_psnr = float(ckpt.get("best_psnr",0.0)); best_lpips = float(ckpt.get("best_lpips",999.0))
        if args.reset_best: best_psnr, best_lpips = 0.0, 999.0
    for epoch in range(start_epoch, args.epochs+1):
        detach_base = args.train_mode == "adapter" and epoch <= args.freeze_base_epochs
        if args.train_mode == "adapter":
            if detach_base: model.freeze_base()
            else: model.unfreeze_base()
        else:
            model.unfreeze_base()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")):
            if args.train_max_batches >= 0 and batch_index >= args.train_max_batches:
                break
            lq, gt, ir = to_01(batch["lq"].to(device)), to_01(batch["gt"].to(device)), batch["ir"].to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                if args.train_mode == "adapter":
                    pred, aux = model(lq, ir, return_aux=True, detach_base=detach_base)
                else:
                    pred = model(lq, ir, return_aux=False, detach_base=False)
                    aux = None
                loss, loss_dict = criterion(pred, gt, aux=aux)
            scaler.scale(loss / args.grad_accum_steps).backward()

            if (batch_index + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before_step:
                    scheduler.step()
                    global_step += 1
                    if global_step == 1 or global_step % 50 == 0:
                        writer.add_scalar("train/loss", loss.item(), global_step)
                        if args.train_mode == "adapter":
                            writer.add_scalar(
                                "train/gamma",
                                float(model.gamma.detach().cpu()),
                                global_step,
                            )

                        if (
                            args.train_mode == "adapter"
                            and hasattr(model.refiner, "get_delta_scale")
                        ):
                            writer.add_scalar(
                                "train/delta_scale",
                                float(model.refiner.get_delta_scale().detach().cpu()),
                                global_step,
                            )
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "best_psnr": best_psnr,
            "best_lpips": best_lpips,
            "args": vars(args),
            "loss_config": loss_config,
            "train_generator_state": train_generator.get_state(),
        }
        if epoch % args.eval_every == 0:
            mb = None if args.eval_max_batches < 0 else args.eval_max_batches
            metrics = evaluate(model, val_loader, device, mb)
        
            delta_scale = (
                float(model.refiner.get_delta_scale().detach().cpu())
                if args.train_mode == "adapter" and hasattr(model.refiner, "get_delta_scale")
                else -1
            )
        
            print(
                f"[Eval] epoch={epoch} "
                f"PSNR={metrics['psnr']:.4f} "
                f"SSIM={metrics['ssim']:.4f} "
                f"LPIPS={metrics['lpips']:.4f} "
                f"gamma={float(model.gamma.detach().cpu()):.5f} "
                f"delta_scale={delta_scale:.5f}"
            )

            append_history(
                args.save_dir,
                {
                    "event": "validation",
                    "epoch": epoch,
                    "experiment_id": args.experiment_id,
                    "seed": args.seed,
                    "global_step": global_step,
                    "gamma": float(model.gamma.detach().cpu()),
                    "delta_scale": delta_scale,
                    **metrics,
                },
            )
        
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                ckpt["best_psnr"] = best_psnr
                ckpt["best_lpips"] = best_lpips
                torch.save(ckpt, os.path.join(args.save_dir, "best_psnr.pth"))
        
            if metrics["lpips"] >= 0 and metrics["lpips"] < best_lpips:
                best_lpips = metrics["lpips"]
                ckpt["best_psnr"] = best_psnr
                ckpt["best_lpips"] = best_lpips
                torch.save(ckpt, os.path.join(args.save_dir, "best_lpips.pth"))
        torch.save(ckpt, os.path.join(args.save_dir, f"epoch_{epoch}.pth"))
        torch.save(ckpt, os.path.join(args.save_dir, "last.pth"))
    writer.close()
if __name__ == "__main__": main()
