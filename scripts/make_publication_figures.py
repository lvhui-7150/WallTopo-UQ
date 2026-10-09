"""Create publication-style figures from collected benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional visual dependency
    sns = None

from _bootstrap import ROOT  # noqa: F401


DISPLAY_NAMES = {
    "walltopo_uq": "WallTopo-UQ",
    "walltopo_uq_v2": "WallTopo-UQ-v2",
    "unet": "U-Net",
    "deeplabv3plus": "DeepLabV3+",
    "smp_unetplusplus": "U-Net++",
    "ablation_no_mdsm": "w/o MDSM",
    "ablation_single_delay": "Single delay",
    "ablation_no_morphology": "w/o morphology",
    "ablation_no_topology_tokens": "w/o topology tokens",
    "ablation_no_dual_cross": "w/o dual cross",
    "ablation_no_topology_loss": "w/o topology loss",
    "ablation_no_geometry_consistency": "w/o geometry consistency",
    "ablation_no_uncertainty": "w/o uncertainty",
}

PALETTE = {
    "walltopo_uq": "#C43C39",
    "walltopo_uq_v2": "#8E44AD",
    "unet": "#2A6FBB",
    "deeplabv3plus": "#35A27A",
    "smp_unetplusplus": "#D28E2C",
}


def _style() -> None:
    if sns is not None:
        sns.set_theme(
            context="paper",
            style="whitegrid",
            rc={
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif"],
                "axes.edgecolor": "#333333",
                "axes.linewidth": 0.65,
                "grid.color": "#D9D9D9",
                "grid.linewidth": 0.45,
                "axes.titlesize": 9,
                "axes.labelsize": 8,
                "legend.fontsize": 7,
            },
        )
    plt.rcParams.update(
        {
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.5,
        }
    )


def _save(figure: plt.Figure, output: Path, stem: str) -> list[str]:
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    figure.savefig(png, dpi=600)
    figure.savefig(pdf)
    plt.close(figure)
    return [str(png), str(pdf)]


def _summary_rows(
    summary: pd.DataFrame,
    variants: list[str],
    metric: str,
) -> pd.DataFrame:
    return summary[
        summary["variant"].isin(variants) & (summary["metric"] == metric)
    ].copy()


def _plot_main_performance(
    summary: pd.DataFrame,
    pairwise_path: Path,
    output: Path,
) -> list[str]:
    preferred = [
        variant
        for variant in [
            "walltopo_uq",
            "walltopo_uq_v2",
            "unet",
            "deeplabv3plus",
            "smp_unetplusplus",
        ]
        if variant in set(summary["variant"])
    ]
    if not preferred:
        return []
    datasets = sorted(summary["dataset"].unique())
    pairwise = pd.read_csv(pairwise_path) if pairwise_path.exists() else pd.DataFrame()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.8),
        constrained_layout=True,
    )
    for axis, metric, label in zip(
        axes,
        ["dice", "cl_dice"],
        ["Dice", "clDice"],
    ):
        selected = _summary_rows(summary, preferred, metric)
        width = 0.78 / max(len(preferred), 1)
        x = np.arange(len(datasets), dtype=np.float64)
        for index, variant in enumerate(preferred):
            values = []
            errors = []
            for dataset in datasets:
                row = selected[
                    (selected["dataset"] == dataset)
                    & (selected["variant"] == variant)
                ]
                values.append(
                    float(row.iloc[0]["mean"]) if not row.empty else np.nan
                )
                errors.append(
                    float(row.iloc[0]["ci95"]) if not row.empty else 0.0
                )
            offset = (index - (len(preferred) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                values,
                width=width * 0.92,
                yerr=errors,
                capsize=1.4,
                color=PALETTE.get(variant, "#777777"),
                label=DISPLAY_NAMES.get(variant, variant),
                error_kw={"elinewidth": 0.7, "capthick": 0.7},
            )
        axis.set_xticks(x)
        axis.set_xticklabels([item.title() for item in datasets])
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel(label)
        axis.grid(axis="x", visible=False)
        if not pairwise.empty:
            for dataset_index, dataset in enumerate(datasets):
                comparisons = pairwise[
                    (pairwise["dataset"] == dataset)
                    & (pairwise["metric"] == metric)
                    & (pairwise["baseline"].astype(str).str.startswith("ablation_") == False)
                ]
                p_values = comparisons["p_value"].dropna()
                if p_values.empty:
                    continue
                minimum = float(p_values.min())
                stars = (
                    "***"
                    if minimum < 0.001
                    else "**"
                    if minimum < 0.01
                    else "*"
                    if minimum < 0.05
                    else ""
                )
                if stars:
                    axis.text(
                        dataset_index,
                        0.97,
                        stars,
                        ha="center",
                        va="top",
                        fontsize=9,
                    )
    axes[0].legend(
        frameon=False,
        ncol=min(2, len(preferred)),
        loc="upper left",
    )
    figure.suptitle("Main crack-segmentation and topology benchmark", fontsize=10)
    return _save(figure, output, "fig_main_performance")


def _plot_ablation(pairwise_path: Path, output: Path) -> list[str]:
    if not pairwise_path.exists():
        return []
    pairwise = pd.read_csv(pairwise_path)
    selected = pairwise[
        pairwise["baseline"].astype(str).str.startswith("ablation_")
        & (pairwise["metric"] == "dice")
    ].copy()
    if selected.empty:
        return []
    datasets = sorted(selected["dataset"].unique())
    variants = sorted(
        selected["baseline"].unique(),
        key=lambda item: (
            selected[selected["baseline"] == item]["improvement"].mean()
        ),
    )
    figure, axis = plt.subplots(figsize=(7.0, max(2.8, 0.36 * len(variants) + 1.2)))
    y = np.arange(len(variants), dtype=np.float64)
    height = 0.76 / max(len(datasets), 1)
    for index, dataset in enumerate(datasets):
        values = []
        errors = []
        for variant in variants:
            row = selected[
                (selected["dataset"] == dataset)
                & (selected["baseline"] == variant)
            ]
            values.append(
                float(row.iloc[0]["improvement"]) if not row.empty else np.nan
            )
            errors.append(
                float(row.iloc[0]["improvement_percent"]) if not row.empty else 0.0
            )
        offset = (index - (len(datasets) - 1) / 2.0) * height
        bars = axis.barh(
            y + offset,
            values,
            height=height * 0.9,
            color=plt.cm.viridis(index / max(len(datasets) - 1, 1)),
            label=dataset.title(),
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(
                    value + (0.0004 if value >= 0 else -0.0004),
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:+.3f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=6.5,
                )
    axis.axvline(0.0, color="#222222", linewidth=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels(
        [DISPLAY_NAMES.get(item, item) for item in variants]
    )
    axis.set_xlabel(r"$\Delta$ Dice (full model $-$ ablated)")
    axis.set_title("Component ablation effect")
    axis.legend(frameon=False)
    axis.grid(axis="y", visible=False)
    return _save(figure, output, "fig_ablation_effects")


def _plot_reliability(summary: pd.DataFrame, output: Path) -> list[str]:
    variants = [
        variant
        for variant in [
            "walltopo_uq",
            "walltopo_uq_v2",
            "unet",
            "deeplabv3plus",
            "smp_unetplusplus",
        ]
        if variant in set(summary["variant"])
    ]
    metrics = [
        ("ece", "ECE", True),
        ("aurc", "AURC", True),
        ("error_detection_auroc", "Error-detection AUROC", False),
        ("selective_error", "Selective Dice error", True),
    ]
    if not variants:
        return []
    datasets = sorted(summary["dataset"].unique())
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.3),
        constrained_layout=True,
    )
    for axis, (metric, title, lower_better) in zip(axes.flat, metrics):
        selected = _summary_rows(summary, variants, metric)
        width = 0.78 / max(len(variants), 1)
        x = np.arange(len(datasets), dtype=np.float64)
        for index, variant in enumerate(variants):
            values = []
            errors = []
            for dataset in datasets:
                row = selected[
                    (selected["dataset"] == dataset)
                    & (selected["variant"] == variant)
                ]
                values.append(
                    float(row.iloc[0]["mean"]) if not row.empty else np.nan
                )
                errors.append(
                    float(row.iloc[0]["ci95"]) if not row.empty else 0.0
                )
            offset = (index - (len(variants) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                values,
                width=width * 0.92,
                yerr=errors,
                capsize=1.4,
                color=PALETTE.get(variant, "#777777"),
                error_kw={"elinewidth": 0.7, "capthick": 0.7},
            )
        axis.set_xticks(x)
        axis.set_xticklabels([item.title() for item in datasets])
        axis.set_title(title)
        if metric in {"ece", "aurc", "selective_error"}:
            axis.set_ylim(bottom=0.0)
        axis.grid(axis="x", visible=False)
    handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE.get(variant, "#777777"),
            lw=5,
            label=DISPLAY_NAMES.get(variant, variant),
        )
        for variant in variants
    ]
    figure.legend(handles=handles, frameon=False, ncol=min(4, len(handles)), loc="upper center")
    return _save(figure, output, "fig_reliability")


def _plot_efficiency(efficiency_path: Path, output: Path) -> list[str]:
    if not efficiency_path.exists():
        return []
    efficiency = pd.read_csv(efficiency_path)
    efficiency = efficiency[np.isfinite(efficiency["seconds_per_iteration"])]
    if efficiency.empty:
        return []
    summary_path = output / "summary.csv"
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    dice = summary[
        (summary["metric"] == "dice")
        & (summary["dataset"] == summary["dataset"].iloc[0])
    ][["variant", "mean"]].rename(columns={"variant": "model", "mean": "dice"})
    merged = efficiency.merge(dice, on="model", how="inner")
    if merged.empty:
        return []
    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    for row in merged.itertuples():
        axis.scatter(
            row.parameters / 1.0e6,
            row.dice,
            s=max(20.0, row.peak_cuda_memory_gb * 70.0),
            color=PALETTE.get(row.model, "#777777"),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.annotate(
            DISPLAY_NAMES.get(row.model, row.model),
            (row.parameters / 1.0e6, row.dice),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Parameters (M)")
    axis.set_ylabel("Dice")
    axis.set_title("Accuracy--efficiency trade-off")
    axis.grid(alpha=0.35)
    return _save(figure, output, "fig_efficiency")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "figures" / "publication",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _style()

    summary_path = args.results / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")
    summary = pd.read_csv(summary_path)
    outputs: list[str] = []
    outputs.extend(
        _plot_main_performance(
            summary,
            args.results / "pairwise_vs_proposed.csv",
            args.output,
        )
    )
    outputs.extend(
        _plot_ablation(
            args.results / "pairwise_vs_proposed.csv",
            args.output,
        )
    )
    outputs.extend(_plot_reliability(summary, args.output))
    outputs.extend(
        _plot_efficiency(
            args.results / "efficiency.csv",
            args.output,
        )
    )
    with (args.output / "figures_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(outputs, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(outputs)} publication files to {args.output.resolve()}")


if __name__ == "__main__":
    main()
