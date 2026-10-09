"""Generate deterministic synthetic wall-crack data for smoke testing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import ROOT
from walltopo_uq.data import SyntheticWallDataset
from walltopo_uq.targets import geometry_severity
from walltopo_uq.utils import ensure_dir, imwrite_unicode, write_json


def generate_split(
    output_root: Path,
    split: str,
    count: int,
    size: int,
    seed: int,
    num_severity_classes: int,
) -> Path:
    split_root = ensure_dir(output_root / split)
    image_dir = ensure_dir(split_root / "images")
    mask_dir = ensure_dir(split_root / "masks")
    dataset = SyntheticWallDataset(
        size=size,
        length=count,
        seed=seed,
        num_severity_classes=num_severity_classes,
    )
    rows: list[dict[str, str | int]] = []
    for index in range(count):
        sample = dataset[index]
        image = sample["image"].numpy().transpose(1, 2, 0)
        mask = sample["mask"].numpy()[0]
        image_path = image_dir / f"{split}_{index:05d}.png"
        mask_path = mask_dir / f"{split}_{index:05d}.png"
        image_written = imwrite_unicode(
            image_path,
            cv2.cvtColor((image.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        mask_written = imwrite_unicode(
            mask_path,
            (mask.clip(0, 1) * 255).astype(np.uint8),
        )
        if not image_written or not mask_written:
            raise RuntimeError(f"Failed to write synthetic sample {index}")
        severity = geometry_severity(mask, sample["skeleton"].numpy()[0], num_severity_classes)
        rows.append(
            {
                "image": str(image_path.relative_to(split_root)),
                "mask": str(mask_path.relative_to(split_root)),
                "severity": severity,
                "group": f"synthetic_{split}",
                "domain": "synthetic",
                "material": "masonry",
                "device": "simulated",
            }
        )
    manifest = split_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--val", type=int, default=16)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severity-classes", type=int, default=5)
    args = parser.parse_args()

    output_root = args.output.resolve()
    manifests = {
        "train": generate_split(
            output_root,
            "train",
            args.train,
            args.size,
            args.seed,
            args.severity_classes,
        ),
        "val": generate_split(
            output_root,
            "val",
            args.val,
            args.size,
            args.seed + 10000,
            args.severity_classes,
        ),
        "test": generate_split(
            output_root,
            "test",
            args.test,
            args.size,
            args.seed + 20000,
            args.severity_classes,
        ),
    }
    write_json(
        output_root / "dataset_summary.json",
        {
            "purpose": "synthetic smoke testing only",
            "image_size": args.size,
            "splits": {
                split: {"count": count, "manifest": str(manifests[split])}
                for split, count in (
                    ("train", args.train),
                    ("val", args.val),
                    ("test", args.test),
                )
            },
        },
    )
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        print(f"{split}: {count} samples -> {manifests[split]}")


if __name__ == "__main__":
    main()
