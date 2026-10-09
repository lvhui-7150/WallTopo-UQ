"""Run prediction and export optional qualitative panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.trainer import Trainer, build_dataloader
from walltopo_uq.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    device = resolve_device(config["device"])
    trainer = Trainer(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model"])
    if args.threshold is not None:
        trainer.mask_threshold = float(args.threshold)
    manifest = config["data"].get(f"{args.split}_manifest")
    if not manifest:
        raise SystemExit(f"No {args.split}_manifest configured.")
    loader = build_dataloader(manifest, config, training=False)
    records = trainer.predict_loader(loader, save_panels=args.save_panels)
    print(f"Predicted {len(records)} images. Results: {trainer.prediction_dir}")


if __name__ == "__main__":
    main()
