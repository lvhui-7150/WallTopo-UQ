"""Train WallTopo-UQ."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.trainer import Trainer
from walltopo_uq.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "smoke.yaml",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    device = resolve_device(config["device"])
    print(f"Device: {device}")
    trainer = Trainer(config, device)
    best = trainer.fit()
    print(f"Best validation metrics: {best}")


if __name__ == "__main__":
    main()
