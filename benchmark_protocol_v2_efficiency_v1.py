"""Compute-only efficiency benchmark for the frozen Protocol V2 B0/P0/FT set.

The benchmark uses deterministic synthetic tensors so disk I/O and data-loader
speed do not affect the comparison. It never saves modified model weights.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DEHAZEFORMER_ROOT = SCRIPT_DIR / "DehazeFormer-main"
if LOCAL_DEHAZEFORMER_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_DEHAZEFORMER_ROOT))

import numpy as np
import torch

from cm_dehazeformer_v2_loss import CrossModalResidualLoss
from crossmodal_dehazeformer_residual_v2 import (
    CrossModalDehazeFormerResidualV2,
    load_dehazeformer_base_weights,
    load_v2_checkpoint,
)


LOCKED_SEED = 3407
LOCKED_SIZE = 256
REPEATS = 3
INFERENCE_WARMUP = 20
LATENCY_ITERATIONS = 100
THROUGHPUT_ITERATIONS = 50
TRAIN_WARMUP_STEPS = 3
TRAIN_MEASURED_STEPS = 10

EXPERIMENTS = {
    "B0": {
        "train_mode": "frozen_baseline",
        "adapter_enabled": False,
        "adapter_variant": "fscatterres",
        "refiner_width": 24,
        "refiner_blocks": 4,
        "checkpoint_type": "base",
        "expected_trainable": 0,
        "train_micro_batch": None,
        "grad_accum_steps": None,
        "lr_base": 0.0,
        "lr_new": 0.0,
    },
    "P0": {
        "train_mode": "adapter",
        "adapter_enabled": True,
        "adapter_variant": "plain",
        "refiner_width": 25,
        "refiner_blocks": 4,
        "checkpoint_type": "v2",
        "expected_trainable": 47255,
        "train_micro_batch": 16,
        "grad_accum_steps": 1,
        "lr_base": 0.0,
        "lr_new": 1e-4,
    },
    "FT": {
        "train_mode": "base_only",
        "adapter_enabled": False,
        "adapter_variant": "fscatterres",
        "refiner_width": 24,
        "refiner_blocks": 4,
        "checkpoint_type": "v2",
        "expected_trainable": 2517612,
        "train_micro_batch": 8,
        "grad_accum_steps": 2,
        "lr_base": 1e-6,
        "lr_new": 0.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark one frozen Protocol V2 B0/P0/FT checkpoint."
    )
    parser.add_argument("--experiment_id", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected_checkpoint_sha256", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--preflight_only", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_checkpoint_args(checkpoint, experiment_id):
    if experiment_id == "B0":
        return
    config = EXPERIMENTS[experiment_id]
    payload = torch.load(checkpoint, map_location="cpu")
    saved = payload.get("args", {}) if isinstance(payload, dict) else {}
    required = {
        "experiment_id": experiment_id,
        "train_mode": config["train_mode"],
        "ablation": "wo_ir_physics",
        "wo_gate": True,
        "use_attnres": False,
        "adapter_variant": config["adapter_variant"],
        "refiner_width": config["refiner_width"],
        "refiner_blocks": config["refiner_blocks"],
        "size": LOCKED_SIZE,
        "seed": LOCKED_SEED,
    }
    mismatches = {
        key: {"expected": value, "actual": saved.get(key)}
        for key, value in required.items()
        if saved.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Checkpoint config mismatch: {json.dumps(mismatches, sort_keys=True)}"
        )


def build_model(experiment_id, checkpoint, device):
    config = EXPERIMENTS[experiment_id]
    model = CrossModalDehazeFormerResidualV2(
        backbone="b",
        init_gamma=0.01,
        ablation="wo_ir_physics",
        wo_gate=True,
        use_attnres=False,
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
        validate_checkpoint_args(checkpoint, experiment_id)
        missing, unexpected = load_v2_checkpoint(model, str(checkpoint), strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Non-exact checkpoint load: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    if experiment_id == "P0":
        model.set_adapter_enabled(True)
        model.freeze_base()
    elif experiment_id == "FT":
        model.set_adapter_enabled(False)
        model.unfreeze_base()
    else:
        model.set_adapter_enabled(False)
        model.freeze_base()
    return model


def parameter_counts(model):
    base_total = sum(parameter.numel() for parameter in model.base.parameters())
    adapter_stored = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("base.")
    )
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "base_total": base_total,
        "adapter_total_stored": adapter_stored,
        "wrapper_total": base_total + adapter_stored,
        "trainable_total": trainable,
        "active_inference_total": base_total + (trainable if trainable < base_total else 0),
    }


def bytes_to_mib(value):
    return float(value) / (1024.0 ** 2)


def summarize(values):
    values = [float(value) for value in values]
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }


def model_forward(model, experiment_id, lq, ir, training=False):
    if training and experiment_id == "P0":
        return model(lq, ir, return_aux=True, detach_base=True)
    prediction = model(lq, ir, return_aux=False, detach_base=False)
    return prediction, None


def make_inputs(batch, device):
    generator = torch.Generator(device="cpu").manual_seed(LOCKED_SEED + batch)
    lq = torch.rand(
        (batch, 3, LOCKED_SIZE, LOCKED_SIZE), generator=generator, dtype=torch.float32
    ).to(device)
    gt = torch.rand(
        (batch, 3, LOCKED_SIZE, LOCKED_SIZE), generator=generator, dtype=torch.float32
    ).to(device)
    ir = torch.zeros((batch, 1, LOCKED_SIZE, LOCKED_SIZE), device=device)
    return lq, gt, ir


def benchmark_inference_case(model, experiment_id, batch, iterations, device):
    lq, _, ir = make_inputs(batch, device)
    model.eval()
    with torch.inference_mode():
        for _ in range(INFERENCE_WARMUP):
            with torch.cuda.amp.autocast(enabled=True):
                model_forward(model, experiment_id, lq, ir, training=False)
        torch.cuda.synchronize()

        times_ms = []
        peaks_allocated = []
        peaks_reserved = []
        for _ in range(REPEATS):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(iterations):
                with torch.cuda.amp.autocast(enabled=True):
                    prediction, _ = model_forward(
                        model, experiment_id, lq, ir, training=False
                    )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0 / iterations
            if not torch.isfinite(prediction).all():
                raise RuntimeError("Non-finite inference output")
            times_ms.append(elapsed_ms)
            peaks_allocated.append(bytes_to_mib(torch.cuda.max_memory_allocated(device)))
            peaks_reserved.append(bytes_to_mib(torch.cuda.max_memory_reserved(device)))

    output = {
        "batch": batch,
        "iterations_per_repeat": iterations,
        "milliseconds_per_batch": summarize(times_ms),
        "milliseconds_per_image": summarize([value / batch for value in times_ms]),
        "images_per_second": summarize([batch * 1000.0 / value for value in times_ms]),
        "peak_allocated_mib": summarize(peaks_allocated),
        "peak_reserved_mib": summarize(peaks_reserved),
    }
    del lq, ir, prediction
    torch.cuda.empty_cache()
    return output


def build_optimizer(model, experiment_id):
    config = EXPERIMENTS[experiment_id]
    base_parameters = []
    new_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith("base."):
            if config["lr_base"] > 0 and parameter.requires_grad:
                base_parameters.append(parameter)
        elif config["lr_new"] > 0 and parameter.requires_grad:
            new_parameters.append(parameter)
    groups = []
    if base_parameters:
        groups.append({"params": base_parameters, "lr": config["lr_base"]})
    if new_parameters:
        groups.append({"params": new_parameters, "lr": config["lr_new"]})
    if not groups:
        raise RuntimeError(f"No optimizer parameters for {experiment_id}")
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=1e-4)


def optimizer_state_bytes(optimizer):
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def one_optimizer_step(
    model,
    experiment_id,
    criterion,
    optimizer,
    scaler,
    lq,
    gt,
    ir,
    accumulation,
    allow_amp_skip=False,
):
    optimizer.zero_grad(set_to_none=True)
    loss_value = None
    for _ in range(accumulation):
        with torch.cuda.amp.autocast(enabled=True):
            prediction, aux = model_forward(
                model, experiment_id, lq, ir, training=True
            )
            loss, _ = criterion(prediction, gt, aux=aux)
            scaled_loss = loss / accumulation
        scaler.scale(scaled_loss).backward()
        loss_value = float(loss.detach())
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    step_applied = scaler.get_scale() >= scale_before
    if not step_applied and not allow_amp_skip:
        raise RuntimeError("AMP skipped an optimizer step during efficiency benchmark")
    if loss_value is None or not math.isfinite(loss_value):
        raise RuntimeError("Non-finite training loss")
    return loss_value, step_applied


def benchmark_training(model, experiment_id, device):
    config = EXPERIMENTS[experiment_id]
    micro_batch = config["train_micro_batch"]
    accumulation = config["grad_accum_steps"]
    lq, gt, ir = make_inputs(micro_batch, device)
    criterion = CrossModalResidualLoss(
        w_char=1.0,
        w_ssim=0.12,
        w_edge=0.03,
        w_color=0.01,
        w_delta=0.0,
        w_gate=0.0,
        w_not_worse=0.0,
    ).to(device)
    optimizer = build_optimizer(model, experiment_id)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model.train()

    warmup_attempts = 0
    warmup_completed = 0
    while warmup_completed < TRAIN_WARMUP_STEPS:
        _, step_applied = one_optimizer_step(
            model,
            experiment_id,
            criterion,
            optimizer,
            scaler,
            lq,
            gt,
            ir,
            accumulation,
            allow_amp_skip=True,
        )
        warmup_attempts += 1
        warmup_completed += int(step_applied)
        if warmup_attempts > 20:
            raise RuntimeError("AMP did not stabilize within 20 warmup attempts")
    torch.cuda.synchronize()

    times_ms = []
    peaks_allocated = []
    peaks_reserved = []
    final_losses = []
    for _ in range(REPEATS):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        final_loss = None
        for _ in range(TRAIN_MEASURED_STEPS):
            final_loss, _ = one_optimizer_step(
                model,
                experiment_id,
                criterion,
                optimizer,
                scaler,
                lq,
                gt,
                ir,
                accumulation,
            )
        torch.cuda.synchronize()
        times_ms.append(
            (time.perf_counter() - started) * 1000.0 / TRAIN_MEASURED_STEPS
        )
        peaks_allocated.append(bytes_to_mib(torch.cuda.max_memory_allocated(device)))
        peaks_reserved.append(bytes_to_mib(torch.cuda.max_memory_reserved(device)))
        final_losses.append(final_loss)

    output = {
        "micro_batch": micro_batch,
        "grad_accum_steps": accumulation,
        "effective_batch": micro_batch * accumulation,
        "warmup_optimizer_steps": TRAIN_WARMUP_STEPS,
        "warmup_attempts": warmup_attempts,
        "measured_optimizer_steps_per_repeat": TRAIN_MEASURED_STEPS,
        "milliseconds_per_optimizer_step": summarize(times_ms),
        "effective_images_per_second": summarize(
            [micro_batch * accumulation * 1000.0 / value for value in times_ms]
        ),
        "peak_allocated_mib": summarize(peaks_allocated),
        "peak_reserved_mib": summarize(peaks_reserved),
        "optimizer_state_mib": bytes_to_mib(optimizer_state_bytes(optimizer)),
        "final_loss_per_repeat": final_losses,
        "excludes": ["data_loading", "validation", "checkpoint_io"],
    }
    del lq, gt, ir, criterion, optimizer, scaler
    torch.cuda.empty_cache()
    return output


def environment_record(device):
    properties = torch.cuda.get_device_properties(device)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": properties.name,
        "gpu_total_memory_mib": bytes_to_mib(properties.total_memory),
        "gpu_compute_capability": [properties.major, properties.minor],
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "amp": "float16",
    }


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_checkpoint_sha256 = sha256_file(args.checkpoint)
    if actual_checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError(
            f"Checkpoint SHA256 mismatch: {actual_checkpoint_sha256} != "
            f"{args.expected_checkpoint_sha256}"
        )
    seed_everything(LOCKED_SEED)

    preflight_device = torch.device("cpu")
    model = build_model(args.experiment_id, args.checkpoint, preflight_device)
    counts = parameter_counts(model)
    expected_trainable = EXPERIMENTS[args.experiment_id]["expected_trainable"]
    if counts["trainable_total"] != expected_trainable:
        raise RuntimeError(
            f"Trainable parameter mismatch: {counts['trainable_total']} != "
            f"{expected_trainable}"
        )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PROTOCOL_V2_EFFICIENCY_V1_PREFLIGHT_PASS",
                    "experiment_id": args.experiment_id,
                    "checkpoint_sha256": actual_checkpoint_sha256,
                    "parameter_counts": counts,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full efficiency benchmark")
    del model
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    model = build_model(args.experiment_id, args.checkpoint, device)
    counts = parameter_counts(model)

    inference = {
        "batch1_latency": benchmark_inference_case(
            model, args.experiment_id, 1, LATENCY_ITERATIONS, device
        ),
        "batch16_throughput": benchmark_inference_case(
            model, args.experiment_id, 16, THROUGHPUT_ITERATIONS, device
        ),
    }
    training = None
    if args.experiment_id in {"P0", "FT"}:
        training = benchmark_training(model, args.experiment_id, device)

    output = {
        "protocol": "Protocol V2 efficiency V1",
        "experiment_id": args.experiment_id,
        "seed": LOCKED_SEED,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "model_config": EXPERIMENTS[args.experiment_id],
        "parameter_counts": counts,
        "environment": environment_record(device),
        "inference": inference,
        "training": training,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "efficiency.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"result: {output_path}")


if __name__ == "__main__":
    main()
