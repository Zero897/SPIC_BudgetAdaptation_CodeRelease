"""Evaluate the project's RGB Restormer on fixed Protocol-V2 validation data."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from restormer_arch_protocol_v2 import Restormer


MANIFEST_SHA256 = "943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128"
ARCHITECTURE_SHA256 = "3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a"
LEVELS = ("light", "dense", "very_dense")
GROUPS = ("all", "day_light", "night_light", "day_dense", "night_dense", "day_very_dense", "night_very_dense")
LABELS = ("全部", "白天淡烟", "夜间淡烟", "白天浓烟", "夜间浓烟", "白天超浓烟", "夜间超浓烟")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path):
    if sha256_file(path) != MANIFEST_SHA256:
        raise ValueError("Expected the frozen 1469-image validation manifest")
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        stem = record["output_stem"]
        if stem in records:
            raise ValueError("Duplicate output_stem: {}".format(stem))
        if record["level"] not in LEVELS or not isinstance(record["is_night"], bool):
            raise ValueError("Invalid severity/day-night metadata: {}".format(stem))
        records[stem] = record
    if len(records) != 1469:
        raise ValueError("Expected 1469 validation samples")
    return records


def image_map(directory):
    paths = {}
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in paths:
                raise ValueError("Duplicate image stem: {}".format(path.stem))
            paths[path.stem] = path
    return paths


class ValidationDataset(Dataset):
    def __init__(self, root, records):
        self.lq = image_map(root / "lq")
        self.gt = image_map(root / "gt")
        if set(self.lq) != set(records) or set(self.gt) != set(records):
            raise ValueError("LQ/GT sample identities do not match the validation manifest")
        self.names = sorted(records)
        for stem in self.names:
            if sha256_file(self.lq[stem]) != records[stem]["lq_sha256"]:
                raise ValueError("Validation LQ hash mismatch: {}".format(stem))

    def __len__(self):
        return len(self.names)

    @staticmethod
    def read_rgb(path):
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image: {}".format(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image).permute(2, 0, 1).float()

    def __getitem__(self, index):
        stem = self.names[index]
        return {"name": stem, "lq": self.read_rgb(self.lq[stem]), "gt": self.read_rgb(self.gt[stem])}


def build_model(checkpoint):
    model = Restormer(
        inp_channels=3, out_channels=3, dim=40, num_blocks=[3, 4, 4, 6],
        num_refinement_blocks=3, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="WithBias", dual_pixel_task=False,
    )
    if sum(p.numel() for p in model.parameters()) != 13_926_281:
        raise ValueError("Restormer architecture parameter count mismatch")
    saved = torch.load(str(checkpoint), map_location="cpu")
    state = saved
    if isinstance(saved, dict):
        for key in ("model", "state_dict", "params", "net", "model_state_dict"):
            if isinstance(saved.get(key), dict):
                state = saved[key]
                break
    # DataParallel is only a naming wrapper; all model keys still load strictly.
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite tensors")
    return model.eval()


def ssim_per_sample(x, y):
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    var_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x.square()
    var_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y.square()
    cov = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / (
        (mu_x.square() + mu_y.square() + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2) + 1e-8
    )
    return score.mean(dim=(1, 2, 3))


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        for group in ("all", "{}_{}".format(row["time"], row["level"])):
            groups[group].append(row)
    return {
        group: {"count": len(groups[group]), **{
            metric: sum(row[metric] for row in groups[group]) / len(groups[group])
            for metric in ("psnr", "ssim", "lpips")
        }} for group in GROUPS
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint_sha256", default=None)
    parser.add_argument("--val_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight", action="store_true", help="Check data/weights without model inference")
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0:
        raise ValueError("batch must be positive and workers non-negative")
    root = args.val_root.resolve()
    if any("test" in part.lower() for part in root.parts):
        raise ValueError("This evaluator accepts validation data only")
    architecture = Path(__file__).with_name("restormer_arch_protocol_v2.py")
    if sha256_file(architecture) != ARCHITECTURE_SHA256:
        raise ValueError("Restormer architecture hash mismatch")
    checkpoint_hash = sha256_file(args.checkpoint)
    if args.checkpoint_sha256 and checkpoint_hash != args.checkpoint_sha256.lower():
        raise ValueError("Checkpoint SHA256 mismatch")
    records = load_manifest(root / "generation_manifest.jsonl")
    dataset = ValidationDataset(root, records)
    model = build_model(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu explicitly if needed")
    print("checkpoint_sha256={}".format(checkpoint_hash))
    print("samples={} strict_load=PASS".format(len(dataset)))
    if args.preflight:
        print("RESTORMER_VALIDATION_PREFLIGHT_PASS")
        return
    import lpips

    torch.manual_seed(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    model = model.to(device)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=args.device == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Restormer validation"):
            lq, gt = batch["lq"].to(device), batch["gt"].to(device)
            pred = model(lq).float().clamp(0, 1)
            mse = (pred - gt).square().mean(dim=(1, 2, 3))
            values = {
                "psnr": 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8)),
                "ssim": ssim_per_sample(pred, gt),
                "lpips": lpips_fn(pred * 2 - 1, gt * 2 - 1).view(-1),
            }
            if not all(torch.isfinite(value).all() for value in values.values()):
                raise ValueError("Non-finite validation metric")
            for index, stem in enumerate(batch["name"]):
                meta = records[stem]
                rows.append({
                    "name": stem, "source_stem": meta["source_stem"],
                    "level": meta["level"], "time": "night" if meta["is_night"] else "day",
                    **{key: value[index].item() for key, value in values.items()},
                })
    aggregates = aggregate(rows)
    result = {
        "model": "Restormer RGB baseline", "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash, "manifest_sha256": MANIFEST_SHA256,
        "architecture_sha256": ARCHITECTURE_SHA256, "evaluator_sha256": sha256_file(__file__),
        "sample_count": len(rows), "size": 256, "input_range": [0, 1],
        "batch": args.batch, "device": str(device), "torch": torch.__version__,
        "val_root": str(root), "aggregates": aggregates,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "per_sample.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("group", "label", "count", "psnr", "ssim", "lpips"))
        writer.writeheader()
        for group, label in zip(GROUPS, LABELS):
            writer.writerow({"group": group, "label": label, **aggregates[group]})
            score = aggregates[group]
            print("{} ({}) n={} PSNR={:.6f} SSIM={:.6f} LPIPS={:.6f}".format(
                label, group, score["count"], score["psnr"], score["ssim"], score["lpips"]))
    print("RESTORMER_VALIDATION_COMPLETE output={}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
