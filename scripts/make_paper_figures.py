"""Generate training-curve and qualitative figures for the manuscript."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import ROOT  # noqa: F401
from walltopo_uq.config import load_config
from walltopo_uq.data import ManifestCrackDataset
from walltopo_uq.metrics import binary_scores
from walltopo_uq.models import build_model
from walltopo_uq.utils import resolve_device


RUNS = {
    "Dais": ROOT / "runs" / "dais",
    "DeepCrack": ROOT / "runs" / "deepcrack",
}
COLORS = {"Dais": "#D55E00", "DeepCrack": "#0072B2"}


def _history(run_dir: Path) -> list[dict[str, float]]:
    with (run_dir / "history.csv").open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.25,
            "savefig.bbox": "tight",
        }
    )


def _training_curve(figures_dir: Path) -> None:
    metrics = [
        ("dice", "Dice"),
        ("cl_dice", "clDice"),
        ("width_mae_px", "Width MAE (px)"),
        ("composite", "Composite"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 1.85))
    for axis, (metric, title) in zip(axes, metrics):
        for name, run_dir in RUNS.items():
            rows = _history(run_dir)
            epochs = np.asarray([row["epoch"] for row in rows])
            values = np.asarray([row[metric] for row in rows])
            axis.plot(epochs, values, color=COLORS[name], label=name)
            best_index = int(np.argmin(values) if metric == "width_mae_px" else np.argmax(values))
            axis.scatter(
                epochs[best_index],
                values[best_index],
                color=COLORS[name],
                edgecolor="white",
                linewidth=0.35,
                s=12,
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.18, linewidth=0.4)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Value")
    axes[1].set_ylim(0.55, 0.92)
    axes[2].set_ylim(0.0, 14.0)
    axes[3].set_ylim(0.50, 0.84)
    axes[-1].legend(frameon=False, loc="lower right")
    figure.savefig(figures_dir / "fig_training_curves.pdf")
    figure.savefig(figures_dir / "fig_training_curves.png", dpi=300)
    plt.close(figure)


def _load_selected_samples(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], torch.nn.Module]:
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = ManifestCrackDataset(
        manifest=config["data"]["test_manifest"],
        image_size=int(config["image_size"]),
        training=False,
        config=config["data"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    scored: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            image = batch["image"].to(device)
            outputs = model(image)
            probability = torch.sigmoid(outputs["mask_logits"])[0, 0].cpu().numpy()
            target = batch["mask"][0, 0].numpy()
            score = binary_scores(probability >= 0.5, target >= 0.5)
            scored.append(
                {
                    "index": index,
                    "dice": score["dice"],
                    "image": batch["image"],
                    "mask": batch["mask"],
                    "skeleton": batch["skeleton"],
                    "meta": batch["meta"],
                }
            )
    scored.sort(key=lambda item: item["dice"])
    median_index = len(scored) // 2
    return [scored[median_index]], config, model


def _error_panel(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    panel = image * 0.42
    true_positive = prediction & target
    false_positive = prediction & ~target
    false_negative = ~prediction & target
    panel[true_positive] = np.asarray([0.0, 0.78, 0.18])
    panel[false_positive] = np.asarray([0.92, 0.08, 0.08])
    panel[false_negative] = np.asarray([0.10, 0.32, 1.0])
    return np.clip(panel, 0.0, 1.0)


def _qualitative(figures_dir: Path, device: torch.device) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 4.0))
    settings = [
        ("Dais", ROOT / "configs" / "dais.yaml", ROOT / "runs" / "dais" / "checkpoints" / "best.pt"),
        (
            "DeepCrack",
            ROOT / "configs" / "deepcrack.yaml",
            ROOT / "runs" / "deepcrack" / "checkpoints" / "best.pt",
        ),
    ]
    for row_index, (name, config_path, checkpoint_path) in enumerate(settings):
        samples, config, model = _load_selected_samples(config_path, checkpoint_path, device)
        sample = samples[0]
        with torch.inference_mode():
            outputs = model(sample["image"].to(device))
        probability = torch.sigmoid(outputs["mask_logits"])[0, 0].cpu().numpy()
        prediction = probability >= 0.5
        target = sample["mask"][0, 0].numpy() >= 0.5
        image = sample["image"][0].permute(1, 2, 0).numpy()
        error = _error_panel(image, target, prediction)
        panels = [image, target.astype(np.float32), prediction.astype(np.float32), error]
        colormaps = [None, "gray", "gray", None]
        for offset, (panel, colormap) in enumerate(zip(panels, colormaps)):
            axis = axes[row_index, offset]
            axis.imshow(panel, cmap=colormap, vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.35)
        score = binary_scores(prediction, target)["dice"]
        axes[row_index, 0].set_ylabel(
            f"{name}\nDice={score:.3f}",
            fontsize=7,
        )
    for axis, label in zip(
        axes[0],
        ["Input", "Ground truth", "Prediction", "Error map"],
    ):
        axis.set_title(label, pad=3)
    figure.subplots_adjust(wspace=0.04, hspace=0.04)
    figure.savefig(figures_dir / "fig_qualitative_results.pdf")
    figure.savefig(figures_dir / "fig_qualitative_results.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT.parent / "figures",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    figures_dir = args.figures_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    _style()
    _training_curve(figures_dir)
    _qualitative(figures_dir, resolve_device(args.device))
    print(f"Saved manuscript figures to {figures_dir}")


if __name__ == "__main__":
    main()
