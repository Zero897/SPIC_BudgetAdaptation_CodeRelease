"""Frozen preflight, technical smoke, and full A2I2 downstream detection V1.

No training, adaptation, or threshold selection is implemented here. The smoke
set is fixed to the first, middle, and last record of the frozen manifest; full
mode uses all 177 frozen records without result-dependent filtering.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DEHAZEFORMER_ROOT = SCRIPT_DIR / "DehazeFormer-main"
if LOCAL_DEHAZEFORMER_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_DEHAZEFORMER_ROOT))

import cv2
import numpy as np
import torch

from crossmodal_dehazeformer_residual_v2 import (
    CrossModalDehazeFormerResidualV2,
    load_dehazeformer_base_weights,
    load_v2_checkpoint,
)


EXPECTED_MANIFEST_SHA256 = (
    "4a6eaa9816ed442af54af1ed4ccabe2297fca4c97f9401913e8bdf7e9e6bb64f"
)
EXPECTED_SAMPLE_COUNT = 177
SMOKE_IDS = ("000", "092", "192")
WIDTH = 1845
HEIGHT = 750
COMPOSITE_HEIGHT = 1500
TILE = 256
OVERLAP = 64
YOLO_CONF_FOR_AP = 0.001
YOLO_FIXED_PR_CONF = 0.25
YOLO_NMS_IOU = 0.7
FASTER_SCORE_FOR_AP = 0.001
FASTER_FIXED_PR_CONF = 0.70195472240448
MAX_DETECTIONS = 300


RESTORATION_CONFIGS = {
    "B0": {
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
        "ablation": "wo_ir_physics",
        "wo_gate": True,
        "use_attnres": False,
        "adapter_variant": "plain",
        "refiner_width": 25,
        "refiner_blocks": 4,
        "adapter_enabled": True,
        "checkpoint_type": "v2",
    },
    "MFT": {
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


def seed_everything(seed: int = 20261013) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def write_png(path: Path, image_bgr: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Manifest SHA256 mismatch: {actual_hash}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"Expected {EXPECTED_SAMPLE_COUNT} records, found {len(records)}")
    if [records[0]["id"], records[len(records) // 2]["id"], records[-1]["id"]] != list(SMOKE_IDS):
        raise ValueError("Frozen smoke IDs no longer match first/middle/last records")
    return records


def validate_dataset(dataset_root: Path, records: Sequence[Dict[str, Any]]) -> None:
    for record in records:
        image_path = dataset_root / record["image_relpath"]
        label_path = dataset_root / record["label_relpath"]
        for path, bytes_key, hash_key in (
            (image_path, "image_bytes", "image_sha256"),
            (label_path, "label_bytes", "label_sha256"),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != record[bytes_key]:
                raise ValueError(f"File size mismatch: {path}")
            if sha256_file(path) != record[hash_key]:
                raise ValueError(f"SHA256 mismatch: {path}")
        image = read_bgr(image_path)
        if image.shape != (COMPOSITE_HEIGHT, WIDTH, 3):
            raise ValueError(f"Unexpected composite shape {image.shape}: {image_path}")


def validate_checkpoints(launch: Dict[str, Any]) -> None:
    for item in launch["restoration_arms"] + list(launch["detectors"].values()):
        path = Path(item["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != item["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {path}: {actual}")


def validate_checkpoint_args(path: Path, experiment_id: str, seed: int) -> None:
    saved = torch.load(path, map_location="cpu")
    args = saved.get("args", {}) if isinstance(saved, dict) else {}
    expected = {"experiment_id": experiment_id, "seed": seed, "size": 256}
    mismatches = {
        key: {"expected": value, "actual": args.get(key)}
        for key, value in expected.items()
        if args.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint args mismatch: {path}: {mismatches}")


def build_restoration(item: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    experiment_id = item["experiment_id"]
    config = RESTORATION_CONFIGS[experiment_id]
    checkpoint = Path(item["checkpoint"])
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
        missing, unexpected = load_dehazeformer_base_weights(model, str(checkpoint), strict=True)
    else:
        validate_checkpoint_args(checkpoint, experiment_id, int(item["seed"]))
        missing, unexpected = load_v2_checkpoint(model, str(checkpoint), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Non-exact restoration load: {item['arm_id']}")
    model.set_adapter_enabled(config["adapter_enabled"])
    model.eval()
    return model


def build_faster(item: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        num_classes=2,
        min_size=512,
        max_size=640,
        box_detections_per_img=MAX_DETECTIONS,
        box_score_thresh=FASTER_SCORE_FOR_AP,
    )
    saved = torch.load(item["checkpoint"], map_location="cpu")
    state = saved.get("model", saved.get("state_dict", saved))
    state = {str(key).replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def preflight(launch: Dict[str, Any], records: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    validate_checkpoints(launch)
    imports = {}
    import torchvision
    from ultralytics import YOLO
    import ultralytics

    imports.update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    strict_loads = []
    for item in launch["restoration_arms"]:
        model = build_restoration(item, torch.device("cpu"))
        strict_loads.append(item["arm_id"])
        del model
        gc.collect()
    faster = build_faster(launch["detectors"]["faster"], torch.device("cpu"))
    del faster
    yolo = YOLO(launch["detectors"]["yolo"]["checkpoint"])
    raw_names = yolo.names
    if isinstance(raw_names, dict):
        yolo_names = {int(key): str(value) for key, value in raw_names.items()}
    else:
        yolo_names = {index: str(value) for index, value in enumerate(raw_names)}
    del yolo
    gc.collect()

    summary = {
        "status": "A2I2_DOWNSTREAM_V1_PREFLIGHT_PASS",
        "sample_count": len(records),
        "smoke_ids": list(SMOKE_IDS),
        "strict_restoration_loads": strict_loads,
        "faster_strict_load": True,
        "faster_num_classes_including_background": 2,
        "yolo_names": yolo_names,
        "environment": imports,
        "inference_started": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def tile_positions(length: int) -> List[int]:
    stride = TILE - OVERLAP
    positions = list(range(0, max(1, length - TILE + 1), stride))
    last = length - TILE
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.inference_mode()
def restore_tiled(model: torch.nn.Module, image_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    ys = tile_positions(tensor.shape[1])
    xs = tile_positions(tensor.shape[2])
    window_1d = torch.hann_window(TILE, periodic=False).clamp_min(0.05)
    window = torch.outer(window_1d, window_1d).view(1, TILE, TILE).to(device)
    accum = torch.zeros((3, tensor.shape[1], tensor.shape[2]), device=device)
    weights = torch.zeros((1, tensor.shape[1], tensor.shape[2]), device=device)
    locations = [(y, x) for y in ys for x in xs]
    for start in range(0, len(locations), 4):
        batch_locations = locations[start : start + 4]
        batch = torch.stack(
            [tensor[:, y : y + TILE, x : x + TILE] for y, x in batch_locations]
        ).to(device)
        ir_zero = torch.zeros((batch.shape[0], 1, TILE, TILE), device=device)
        predictions = model(batch, ir_zero).float().clamp(0, 1)
        for prediction, (y, x) in zip(predictions, batch_locations):
            accum[:, y : y + TILE, x : x + TILE] += prediction * window
            weights[:, y : y + TILE, x : x + TILE] += window
    output = (accum / weights.clamp_min(1e-8)).clamp(0, 1)
    return (output.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def write_arm_inputs(
    arm_dir: Path,
    records: Sequence[Dict[str, Any]],
    dataset_root: Path,
    model: torch.nn.Module = None,
    device: torch.device = None,
    progress_label: str = None,
) -> List[Dict[str, Any]]:
    rows = []
    image_dir = arm_dir / "images"
    label_dir = arm_dir / "labels"
    for record_index, record in enumerate(records, start=1):
        composite = read_bgr(dataset_root / record["image_relpath"])
        upper_bgr = composite[:HEIGHT]
        if model is None:
            output_bgr = upper_bgr
        else:
            rgb = cv2.cvtColor(upper_bgr, cv2.COLOR_BGR2RGB)
            restored_rgb = restore_tiled(model, rgb, device)
            output_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        image_path = image_dir / f"{record['id']}.png"
        label_path = label_dir / f"{record['id']}.txt"
        write_png(image_path, output_bgr)
        labels = []
        for x1, y1, x2, y2 in record["boxes"]:
            xc = (x1 + x2) / (2 * WIDTH)
            yc = (y1 + y2) / (2 * HEIGHT)
            width = (x2 - x1) / WIDTH
            height = (y2 - y1) / HEIGHT
            labels.append(f"0 {xc:.10f} {yc:.10f} {width:.10f} {height:.10f}")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="ascii")
        rows.append(
            {
                "id": record["id"],
                "image": str(image_path),
                "image_sha256": sha256_file(image_path),
                "label": str(label_path),
                "label_sha256": sha256_file(label_path),
                "width": WIDTH,
                "height": HEIGHT,
            }
        )
        if progress_label and (
            record_index == 1 or record_index % 10 == 0 or record_index == len(records)
        ):
            print(
                f"[{progress_label}] images={record_index}/{len(records)}",
                flush=True,
            )
    return rows


def make_coco_ground_truth(
    records: Sequence[Dict[str, Any]], description: str
) -> Dict[str, Any]:
    images = []
    annotations = []
    annotation_id = 1
    for record in records:
        image_id = int(record["index"]) + 1
        images.append({"id": image_id, "file_name": f"{record['id']}.png", "width": WIDTH, "height": HEIGHT})
        for x1, y1, x2, y2 in record["boxes"]:
            width = x2 - x1
            height = y2 - y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {"description": description},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "vehicle", "supercategory": "vehicle"}],
    }


def yolo_predictions(checkpoint: str, image_dir: Path, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    paths = [str(image_dir / f"{record['id']}.png") for record in records]
    results = model.predict(
        source=paths,
        imgsz=640,
        conf=YOLO_CONF_FOR_AP,
        iou=YOLO_NMS_IOU,
        max_det=MAX_DETECTIONS,
        device=0,
        verbose=False,
        stream=False,
    )
    predictions = []
    for record, result in zip(records, results):
        if result.boxes is None:
            continue
        for box, score, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            if int(class_id) != 0:
                continue
            x1, y1, x2, y2 = map(float, box)
            predictions.append(
                {
                    "image_id": int(record["index"]) + 1,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


@torch.inference_mode()
def faster_predictions(model: torch.nn.Module, image_dir: Path, records: Sequence[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    predictions = []
    for record in records:
        bgr = read_bgr(image_dir / f"{record['id']}.png")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        output = model([tensor])[0]
        for box, score, label in zip(
            output["boxes"].cpu().numpy(),
            output["scores"].cpu().numpy(),
            output["labels"].cpu().numpy(),
        ):
            if int(label) != 1:
                continue
            x1, y1, x2, y2 = map(float, box)
            predictions.append(
                {
                    "image_id": int(record["index"]) + 1,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return predictions


def box_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def fixed_precision_recall(gt: Dict[str, Any], predictions: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, float]:
    gt_by_image: Dict[int, List[Sequence[float]]] = {}
    pred_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for annotation in gt["annotations"]:
        gt_by_image.setdefault(annotation["image_id"], []).append(annotation["bbox"])
    for prediction in predictions:
        if prediction["score"] >= threshold:
            pred_by_image.setdefault(prediction["image_id"], []).append(prediction)
    tp = fp = fn = 0
    for image in gt["images"]:
        image_id = image["id"]
        targets = gt_by_image.get(image_id, [])
        matched = set()
        candidates = sorted(pred_by_image.get(image_id, []), key=lambda row: row["score"], reverse=True)
        for prediction in candidates:
            options = [
                (box_iou_xywh(prediction["bbox"], target), index)
                for index, target in enumerate(targets)
                if index not in matched
            ]
            best_iou, best_index = max(options, default=(0.0, -1))
            if best_iou >= 0.5:
                tp += 1
                matched.add(best_index)
            else:
                fp += 1
        fn += len(targets) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision_iou50": precision, "recall_iou50": recall}


def coco_metrics(gt: Dict[str, Any], predictions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Single-class COCO-style 101-point AP for non-crowd A2I2 boxes."""
    gt_by_image: Dict[int, List[Sequence[float]]] = {}
    for annotation in gt["annotations"]:
        gt_by_image.setdefault(annotation["image_id"], []).append(annotation["bbox"])
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        raise ValueError("A2I2 evaluation has no ground-truth boxes")

    pred_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for prediction in predictions:
        pred_by_image.setdefault(prediction["image_id"], []).append(prediction)
    capped_predictions = []
    for image_predictions in pred_by_image.values():
        capped_predictions.extend(
            sorted(image_predictions, key=lambda row: row["score"], reverse=True)[
                :MAX_DETECTIONS
            ]
        )
    capped_predictions.sort(key=lambda row: row["score"], reverse=True)

    thresholds = [round(0.50 + 0.05 * index, 2) for index in range(10)]
    average_precisions = []
    recalls = []
    for iou_threshold in thresholds:
        matched: Dict[int, set] = {}
        true_positive = []
        false_positive = []
        for prediction in capped_predictions:
            image_id = prediction["image_id"]
            targets = gt_by_image.get(image_id, [])
            used = matched.setdefault(image_id, set())
            options = [
                (box_iou_xywh(prediction["bbox"], target), index)
                for index, target in enumerate(targets)
                if index not in used
            ]
            best_iou, best_index = max(options, default=(0.0, -1))
            is_match = best_iou >= iou_threshold
            true_positive.append(1.0 if is_match else 0.0)
            false_positive.append(0.0 if is_match else 1.0)
            if is_match:
                used.add(best_index)

        if true_positive:
            tp_cumulative = np.cumsum(np.asarray(true_positive, dtype=np.float64))
            fp_cumulative = np.cumsum(np.asarray(false_positive, dtype=np.float64))
            recall_curve = tp_cumulative / total_gt
            precision_curve = tp_cumulative / np.maximum(
                tp_cumulative + fp_cumulative, 1e-12
            )
            interpolated = []
            for recall_level in np.linspace(0.0, 1.0, 101):
                eligible = precision_curve[recall_curve >= recall_level]
                interpolated.append(float(eligible.max()) if eligible.size else 0.0)
            average_precisions.append(float(np.mean(interpolated)))
            recalls.append(float(recall_curve[-1]))
        else:
            average_precisions.append(0.0)
            recalls.append(0.0)

    return {
        "ap50_95": float(np.mean(average_precisions)),
        "ap50": average_precisions[0],
        "ap75": average_precisions[5],
        "ar300": float(np.mean(recalls)),
    }


def run_evaluation(
    launch: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    mode: str,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for the A2I2 {mode} evaluation")
    validate_checkpoints(launch)
    output_dir.mkdir(parents=True, exist_ok=False)
    arms_root = output_dir / "arms"
    arm_manifests = {
        "raw": write_arm_inputs(
            arms_root / "raw",
            records,
            dataset_root,
            progress_label="raw",
        )
    }
    device = torch.device("cuda")
    for item in launch["restoration_arms"]:
        print(f"[{item['arm_id']}] loading restoration model", flush=True)
        model = build_restoration(item, device)
        arm_manifests[item["arm_id"]] = write_arm_inputs(
            arms_root / item["arm_id"],
            records,
            dataset_root,
            model,
            device,
            progress_label=item["arm_id"],
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    gt = make_coco_ground_truth(
        records,
        "A2I2 frozen technical smoke"
        if mode == "smoke"
        else "A2I2 frozen full 177-image evaluation",
    )
    (output_dir / "ground_truth_coco.json").write_text(
        json.dumps(gt, indent=2) + "\n", encoding="utf-8"
    )
    metrics: Dict[str, Dict[str, Any]] = {"yolo": {}, "faster": {}}
    predictions_root = output_dir / "predictions"
    predictions_root.mkdir()
    for arm_id in arm_manifests:
        print(f"[yolo/{arm_id}] inference start", flush=True)
        predictions = yolo_predictions(
            launch["detectors"]["yolo"]["checkpoint"],
            arms_root / arm_id / "images",
            records,
        )
        (predictions_root / f"yolo_{arm_id}.json").write_text(
            json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
        )
        metrics["yolo"][arm_id] = {
            **coco_metrics(gt, predictions),
            **fixed_precision_recall(gt, predictions, YOLO_FIXED_PR_CONF),
        }
        print(f"[yolo/{arm_id}] inference complete", flush=True)

    print("[faster] loading detector", flush=True)
    faster = build_faster(launch["detectors"]["faster"], device)
    for arm_id in arm_manifests:
        print(f"[faster/{arm_id}] inference start", flush=True)
        predictions = faster_predictions(faster, arms_root / arm_id / "images", records, device)
        (predictions_root / f"faster_{arm_id}.json").write_text(
            json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
        )
        metrics["faster"][arm_id] = {
            **coco_metrics(gt, predictions),
            **fixed_precision_recall(gt, predictions, FASTER_FIXED_PR_CONF),
        }
        print(f"[faster/{arm_id}] inference complete", flush=True)
    del faster
    torch.cuda.empty_cache()

    for arm_id, rows in arm_manifests.items():
        path = output_dir / f"{arm_id}_manifest.json"
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    is_smoke = mode == "smoke"
    summary = {
        "status": (
            "A2I2_DOWNSTREAM_V1_TECHNICAL_SMOKE_PASS"
            if is_smoke
            else "A2I2_DOWNSTREAM_V1_FULL_PASS"
        ),
        "scope": (
            "technical only; metrics must not select or alter any method, checkpoint, threshold, subset, or inference setting"
            if is_smoke
            else "frozen full evaluation; metrics must not select or alter any method, checkpoint, threshold, subset, or inference setting"
        ),
        "mode": mode,
        "sample_count": len(records),
        "smoke_ids": list(SMOKE_IDS) if is_smoke else None,
        "arm_count": len(arm_manifests),
        "detectors": ["yolo", "faster"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "metrics": metrics,
    }
    summary_name = "smoke_summary.json" if is_smoke else "full_summary.json"
    (output_dir / summary_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=("preflight", "smoke", "full")
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--launch_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    seed_everything()
    records = load_manifest(args.manifest)
    validate_dataset(args.dataset_root.resolve(), records)
    launch = load_json(args.launch_manifest)
    if args.mode == "preflight":
        preflight(launch, records, args.output_dir.resolve())
    else:
        if args.mode == "smoke":
            records_by_id = {record["id"]: record for record in records}
            selected_records = [records_by_id[item] for item in SMOKE_IDS]
        else:
            selected_records = records
        run_evaluation(
            launch,
            selected_records,
            args.dataset_root.resolve(),
            args.output_dir.resolve(),
            args.mode,
        )


if __name__ == "__main__":
    main()
