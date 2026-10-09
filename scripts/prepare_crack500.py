"""Pair CRACK500 images/masks and create reproducible manifests."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "data/raw/sources/crack500_mirror/"
            "crack-detection-unet-master/CRACK500/CRACK500"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/crack500"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    image_dir = source_root / "JPEGImages" / "images"
    mask_dir = source_root / "Annotations" / "masks"
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        candidates = [
            mask_dir / f"{image_path.stem}_mask.png",
            mask_dir / f"{image_path.stem}.png",
        ]
        mask_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if mask_path is None:
            missing.append(image_path.name)
            continue
        rows.append(
            {
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "severity": "",
                "group": f"crack500_{image_path.stem}",
                "domain": "road",
                "material": "asphalt",
                "device": "camera",
            }
        )
    if not rows:
        raise SystemExit("No CRACK500 image-mask pairs were found.")

    random.Random(args.seed).shuffle(rows)
    train_end = int(len(rows) * args.train_ratio)
    val_end = train_end + int(len(rows) * args.val_ratio)
    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]
    if not val_rows:
        val_rows.append(train_rows.pop())
    if not test_rows:
        test_rows.append(train_rows.pop())

    output_dir = args.output_dir.resolve()
    _write(output_dir / "train.csv", train_rows)
    _write(output_dir / "val.csv", val_rows)
    _write(output_dir / "test.csv", test_rows)
    _write(output_dir / "manifest.csv", rows)
    print(
        f"CRACK500: train={len(train_rows)} val={len(val_rows)} "
        f"test={len(test_rows)} unpaired={len(missing)} -> {output_dir}"
    )
    if missing:
        print("Unpaired images: " + ", ".join(missing))


if __name__ == "__main__":
    main()
