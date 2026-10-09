"""Compute physical-width agreement statistics and a Bland-Altman plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _coefficient_of_determination(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.sum((target - prediction) ** 2))
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--measured-column", default="measured_mm")
    parser.add_argument("--predicted-column", default="predicted_mm")
    parser.add_argument("--predicted-pixel-column", default="predicted_width_px")
    parser.add_argument("--pixels-per-mm-column", default="pixels_per_mm")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.predicted_column in frame.columns:
        measured = frame[args.measured_column].to_numpy(dtype=np.float64)
        predicted = frame[args.predicted_column].to_numpy(dtype=np.float64)
    else:
        measured = frame[args.measured_column].to_numpy(dtype=np.float64)
        pixels_per_mm = frame[args.pixels_per_mm_column].to_numpy(dtype=np.float64)
        predicted = (
            frame[args.predicted_pixel_column].to_numpy(dtype=np.float64)
            / np.clip(pixels_per_mm, 1.0e-9, None)
        )

    valid = np.isfinite(measured) & np.isfinite(predicted)
    measured = measured[valid]
    predicted = predicted[valid]
    if measured.size < 3:
        raise SystemExit("At least three valid paired measurements are required.")

    error = predicted - measured
    bias = float(error.mean())
    standard_deviation = float(error.std(ddof=1))
    limits = (bias - 1.96 * standard_deviation, bias + 1.96 * standard_deviation)
    metrics = {
        "n": int(measured.size),
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "bias_mm": bias,
        "loa_lower_mm": limits[0],
        "loa_upper_mm": limits[1],
        "r2": _coefficient_of_determination(measured, predicted),
        "pearson_r": float(stats.pearsonr(measured, predicted).statistic),
        "spearman_r": float(stats.spearmanr(measured, predicted).statistic),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "physical_geometry_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    axes[0].scatter(measured, predicted, s=18, alpha=0.75, edgecolors="none")
    lower = min(float(measured.min()), float(predicted.min()))
    upper = max(float(measured.max()), float(predicted.max()))
    axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1.0)
    axes[0].set_xlabel("Measured width (mm)")
    axes[0].set_ylabel("Predicted width (mm)")
    axes[0].set_title("Prediction agreement")
    axes[0].grid(alpha=0.25)

    average = 0.5 * (measured + predicted)
    axes[1].scatter(average, error, s=18, alpha=0.75, edgecolors="none")
    axes[1].axhline(bias, color="black", linewidth=1.0)
    axes[1].axhline(limits[0], color="#B22222", linestyle="--", linewidth=1.0)
    axes[1].axhline(limits[1], color="#B22222", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Average width (mm)")
    axes[1].set_ylabel("Predicted - measured (mm)")
    axes[1].set_title("Bland-Altman")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "physical_geometry.png", dpi=600)
    figure.savefig(args.output_dir / "physical_geometry.pdf")
    plt.close(figure)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
