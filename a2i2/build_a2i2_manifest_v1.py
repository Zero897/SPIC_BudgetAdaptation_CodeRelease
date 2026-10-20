import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    images = sorted((root / "dehazing" / "images").glob("*.jpg"))
    if len(images) != 177:
        raise ValueError(f"Expected 177 images, found {len(images)}")

    rows = []
    for index, image_path in enumerate(images):
        label_path = root / "dehazing" / "labels" / f"{image_path.stem}.txt"
        boxes = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5 or fields[0] != "vehicle":
                raise ValueError(f"Invalid label at {label_path}:{line_number}")
            x1, y1, x2, y2 = map(float, fields[1:])
            if not (0 <= x1 < x2 <= 1845 and 0 <= y1 < y2 <= 750):
                raise ValueError(f"Out-of-bounds label at {label_path}:{line_number}")
            boxes.append([x1, y1, x2, y2])
        rows.append(
            {
                "index": index,
                "id": image_path.stem,
                "image_relpath": f"dehazing/images/{image_path.name}",
                "label_relpath": f"dehazing/labels/{label_path.name}",
                "image_sha256": sha256_file(image_path),
                "label_sha256": sha256_file(label_path),
                "image_bytes": image_path.stat().st_size,
                "label_bytes": label_path.stat().st_size,
                "composite_width": 1845,
                "composite_height": 1500,
                "evaluation_height": 750,
                "boxes": boxes,
            }
        )

    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "boxes": sum(len(row["boxes"]) for row in rows),
                "smoke_ids": [rows[0]["id"], rows[len(rows) // 2]["id"], rows[-1]["id"]],
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
