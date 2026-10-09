"""Convert CrackForest MATLAB masks and build a WallTopo-UQ manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.utils import ensure_dir, imwrite_unicode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "data" / "raw" / "sources" / "crackforest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "crackforest",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    image_dir = source_root / "image"
    truth_dir = source_root / "groundTruth"
    mask_dir = ensure_dir(args.output_dir / "masks")
    rows: list[dict[str, str | int]] = []
    for truth_path in sorted(truth_dir.glob("*.mat")):
        image_path = next(
            (
                image_dir / f"{truth_path.stem}{extension}"
                for extension in (".jpg", ".jpeg", ".png", ".bmp")
                if (image_dir / f"{truth_path.stem}{extension}").exists()
            ),
            None,
        )
        if image_path is None:
            continue
        payload = loadmat(truth_path)
        segmentation = np.asarray(
            payload["groundTruth"][0, 0]["Segmentation"]
        )
        mask = np.where(segmentation > 1, 255, 0).astype(np.uint8)
        mask_path = mask_dir / f"{truth_path.stem}.png"
        if not imwrite_unicode(mask_path, mask):
            raise RuntimeError(f"Unable to write mask: {mask_path}")
        rows.append(
            {
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "severity": "",
                "group": f"crackforest_{truth_path.stem}",
                "domain": "road",
                "material": "asphalt",
                "device": "camera",
            }
        )
    if not rows:
        raise SystemExit("No CrackForest image-mask pairs were found.")
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} pairs -> {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
