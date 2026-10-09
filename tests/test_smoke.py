from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from walltopo_uq.data import ManifestCrackDataset, SyntheticWallDataset
from walltopo_uq.losses import WallTopoLoss
from walltopo_uq.models import UNetBaseline, WallTopoUQ
from walltopo_uq.targets import derive_targets
from walltopo_uq.uncertainty import (
    geometry_dispersion,
    selective_metrics,
    topology_instability,
)
from walltopo_uq.utils import imwrite_unicode


def test_derived_targets_have_expected_shapes() -> None:
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[8:56, 30:33] = 1.0
    targets = derive_targets(mask)
    assert targets["mask"].shape == (64, 64)
    assert targets["skeleton"].shape == (64, 64)
    assert targets["endpoint"].shape == (64, 64)
    assert targets["junction"].shape == (64, 64)
    assert targets["orientation"].shape == (2, 64, 64)
    assert targets["width"].shape == (64, 64)


def test_synthetic_model_forward_backward() -> None:
    model = WallTopoUQ(
        channels=(8, 16, 24, 32),
        mdsm_stages=(3,),
        max_attention_tokens=64,
    )
    image = torch.randn(2, 3, 64, 64)
    outputs = model(image)
    assert outputs["mask_logits"].shape == (2, 1, 64, 64)
    assert outputs["skeleton_logits"].shape == (2, 1, 64, 64)
    assert outputs["orientation"].shape == (2, 2, 64, 64)
    assert outputs["evidence"].shape == (2, 2, 64, 64)
    assert outputs["severity_logits"].shape == (2, 4)
    loss = outputs["mask_logits"].mean() + outputs["skeleton_logits"].mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_segmentation_shortcut_forward_backward() -> None:
    model = WallTopoUQ(
        channels=(8, 16, 24, 32),
        mdsm_stages=(3,),
        max_attention_tokens=64,
        use_segmentation_shortcut=True,
        shortcut_gate_bias=-2.0,
    )
    image = torch.randn(2, 3, 64, 64)
    outputs = model(image)
    assert outputs["mask_logits"].shape == (2, 1, 64, 64)
    loss = outputs["mask_logits"].mean() + outputs["skeleton_logits"].mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_loss_step_on_synthetic_sample() -> None:
    dataset = SyntheticWallDataset(size=64, length=1, seed=7)
    sample = dataset[0]
    batch = {
        key: value[None] if torch.is_tensor(value) and key != "meta" else value
        for key, value in sample.items()
    }
    model = WallTopoUQ(
        channels=(8, 16, 24, 32),
        mdsm_stages=(3,),
        max_attention_tokens=64,
    )
    outputs = model(batch["image"])
    criterion = WallTopoLoss(
        {
            "seg": 1.0,
            "topo": 0.5,
            "geo": 0.5,
            "ordinal": 0.2,
            "evidence": 0.1,
        }
    )
    loss, components = criterion(outputs, batch, epoch_fraction=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss_topo" in components


def test_manifest_dataset_reads_utf8_paths(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    image = np.full((32, 32, 3), 180, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:28, 15:17] = 255
    image_path = image_dir / "wall_001.png"
    mask_path = mask_dir / "wall_001.png"
    assert imwrite_unicode(image_path, image)
    assert imwrite_unicode(mask_path, mask)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "mask", "severity", "group"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image": "images/wall_001.png",
                "mask": "masks/wall_001.png",
                "severity": 2,
                "group": "wall_001",
            }
        )
    dataset = ManifestCrackDataset(
        manifest,
        image_size=32,
        training=False,
        config={"cache_derived": True, "use_severity": True},
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)
    assert sample["severity"].item() == 2


def test_smoke_config_can_be_loaded() -> None:
    pytest.importorskip("yaml")
    from walltopo_uq.config import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    assert config["image_size"] == 128
    assert len(config["channels"]) == 4


def test_unet_baseline_forward() -> None:
    model = UNetBaseline(channels=(8, 16, 24, 32))
    outputs = model(torch.randn(2, 3, 64, 64))
    assert outputs["mask_logits"].shape == (2, 1, 64, 64)
    assert outputs["topology_tokens"].shape == (2, 4, 8)


def test_uncertainty_helpers_detect_view_variation() -> None:
    stable = np.zeros((32, 32), dtype=np.float32)
    stable[4:28, 15:17] = 1.0
    changed = stable.copy()
    changed[4:12, 15:17] = 0.0
    assert topology_instability([stable, stable]) == pytest.approx(0.0)
    assert topology_instability([stable, changed]) > 0.0
    assert geometry_dispersion([10.0, 10.0], [100.0, 100.0]) == pytest.approx(0.0)
    assert geometry_dispersion([8.0, 12.0], [90.0, 110.0]) > 0.0


def test_selective_metrics_reward_useful_uncertainty() -> None:
    errors = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    useful = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    metrics = selective_metrics(errors, useful, coverage=0.5)
    assert metrics["selective_error"] < errors.mean()
    assert metrics["error_detection_auroc"] == pytest.approx(1.0)
