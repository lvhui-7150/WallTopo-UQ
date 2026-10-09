"""Print basic dataset and target statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from _bootstrap import ROOT
from walltopo_uq.config import load_config
from walltopo_uq.data import ManifestCrackDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.yaml")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = ManifestCrackDataset(
        config["data"][f"{args.split}_manifest"],
        config["image_size"],
        training=False,
        config=config["data"],
    )
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=False)
    batch = next(iter(loader))
    report = {
        "samples": len(dataset),
        "image_shape": tuple(batch["image"].shape),
        "mean_mask_ratio": float(batch["mask"].mean()),
        "skeleton_pixels": float(batch["skeleton"].sum()),
        "endpoint_pixels": float(batch["endpoint"].sum()),
        "junction_pixels": float(batch["junction"].sum()),
        "mean_width_on_skeleton": float(
            (batch["width"] * batch["skeleton"]).sum()
            / batch["skeleton"].sum().clamp_min(1.0)
        ),
        "severity_valid": float(batch["severity_valid"].mean()),
        "severity_values": np.bincount(
            batch["severity"].numpy()[batch["severity_valid"].numpy() > 0.5],
            minlength=int(config["num_severity_classes"]),
        ).tolist(),
    }
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
