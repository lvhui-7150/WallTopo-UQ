"""Collect benchmark metrics, confidence intervals, tests, and LaTeX tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DISPLAY_NAMES = {
    "walltopo_uq": "WallTopo-UQ",
    "walltopo_uq_v2": "WallTopo-UQ-v2",
    "unet": "U-Net",
    "unet_cldice": "U-Net + clDice",
    "unet_connectivity": "U-Net + connectivity",
    "deeplabv3plus": "DeepLabV3+",
    "smp_unetplusplus": "U-Net++",
    "ablation_no_mdsm": "w/o MDSM",
    "ablation_single_delay": "Single delay",
    "ablation_no_morphology": "w/o morphology gate",
    "ablation_no_topology_tokens": "w/o topology tokens",
    "ablation_no_dual_cross": "w/o dual cross",
    "ablation_no_topology_loss": "w/o topology loss",
    "ablation_no_geometry_consistency": "w/o geometry consistency",
    "ablation_no_uncertainty": "w/o uncertainty",
}

IMPORTANT_METRICS = [
    "dice",
    "iou",
    "boundary_f1",
    "cl_dice",
    "endpoint_f1",
    "junction_f1",
    "break_rate",
    "false_merge_error",
    "length_mae_px",
    "width_mae_px",
    "orientation_mae_deg",
    "ece",
    "aurc",
    "error_detection_auroc",
    "selective_error",
]

LOWER_IS_BETTER = {
    "break_rate",
    "false_merge_error",
    "length_mae_px",
    "width_mae_px",
    "orientation_mae_deg",
    "ece",
    "aurc",
    "selective_error",
}


def _load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.rglob("test_metrics.json")):
        run_dir = metrics_path.parent
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                rows.append(
                    {
                        "dataset": str(metadata["dataset"]),
                        "variant": str(metadata["variant"]),
                        "seed": int(metadata["seed"]),
                        "metric": str(metric),
                        "value": float(value),
                        "run_dir": str(run_dir),
                    }
                )
    return rows


def _summary(raw: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = raw.groupby(["dataset", "variant", "metric"], sort=True)
    for (dataset, variant, metric), frame in grouped:
        values = frame["value"].to_numpy(dtype=np.float64)
        count = int(values.size)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if count > 1 else 0.0
        if count > 1:
            critical = float(stats.t.ppf(0.975, count - 1))
            ci95 = critical * std / math.sqrt(count)
        else:
            ci95 = 0.0
        records.append(
            {
                "dataset": dataset,
                "variant": variant,
                "metric": metric,
                "mean": mean,
                "std": std,
                "ci95": ci95,
                "count": count,
            }
        )
    return pd.DataFrame.from_records(records)


def _pairwise(raw: pd.DataFrame, proposed: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset in sorted(raw["dataset"].unique()):
        dataset_frame = raw[raw["dataset"] == dataset]
        baselines = sorted(set(dataset_frame["variant"]) - {proposed})
        for baseline in baselines:
            for metric in IMPORTANT_METRICS:
                left = (
                    dataset_frame[
                        (dataset_frame["variant"] == proposed)
                        & (dataset_frame["metric"] == metric)
                    ][["seed", "value"]]
                    .rename(columns={"value": "proposed"})
                )
                right = (
                    dataset_frame[
                        (dataset_frame["variant"] == baseline)
                        & (dataset_frame["metric"] == metric)
                    ][["seed", "value"]]
                    .rename(columns={"value": "baseline"})
                )
                paired = left.merge(right, on="seed", how="inner")
                if paired.empty:
                    continue
                difference = (
                    paired["baseline"].to_numpy() - paired["proposed"].to_numpy()
                    if metric in LOWER_IS_BETTER
                    else paired["proposed"].to_numpy() - paired["baseline"].to_numpy()
                )
                sample_count = int(difference.size)
                if sample_count > 1 and np.any(np.abs(difference) > 1.0e-12):
                    if sample_count >= 6:
                        test_name = "wilcoxon"
                        p_value = float(
                            stats.wilcoxon(
                                difference,
                                zero_method="wilcox",
                                alternative="two-sided",
                            ).pvalue
                        )
                    else:
                        test_name = "paired_t"
                        p_value = float(
                            stats.ttest_rel(
                                paired["proposed"],
                                paired["baseline"],
                            ).pvalue
                        )
                else:
                    test_name = "not_available"
                    p_value = float("nan")
                mean_baseline = float(paired["baseline"].mean())
                mean_improvement = float(difference.mean())
                percent = (
                    100.0 * mean_improvement / abs(mean_baseline)
                    if abs(mean_baseline) > 1.0e-12
                    else float("nan")
                )
                records.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "metric": metric,
                        "direction": (
                            "lower_is_better"
                            if metric in LOWER_IS_BETTER
                            else "higher_is_better"
                        ),
                        "proposed_mean": float(paired["proposed"].mean()),
                        "baseline_mean": mean_baseline,
                        "improvement": mean_improvement,
                        "improvement_percent": percent,
                        "count": sample_count,
                        "test": test_name,
                        "p_value": p_value,
                    }
                )
    return pd.DataFrame.from_records(records)


def _format_mean_std(
    summary: pd.DataFrame,
    dataset: str,
    variant: str,
    metric: str,
) -> str:
    selected = summary[
        (summary["dataset"] == dataset)
        & (summary["variant"] == variant)
        & (summary["metric"] == metric)
    ]
    if selected.empty:
        return "--"
    row = selected.iloc[0]
    if int(row["count"]) <= 1:
        return f"{float(row['mean']):.3f}"
    return f"{float(row['mean']):.3f}$\\pm${float(row['std']):.3f}"


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def _write_main_table(summary: pd.DataFrame, output: Path) -> None:
    core_variants = [
        item
        for item in [
            "walltopo_uq",
            "walltopo_uq_v2",
            "unet",
            "deeplabv3plus",
            "smp_unetplusplus",
        ]
        if item in set(summary["variant"])
    ]
    datasets = sorted(summary["dataset"].unique())
    if not core_variants or not datasets:
        return
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Main segmentation and topology comparison. Values are mean"
        " $\\pm$ standard deviation over the available seeds.}",
        "\\label{tab:benchmark_main}",
        "\\small",
        "\\begin{tabular}{l" + "cc" * len(datasets) + "}",
        "\\toprule",
        "& "
        + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{_latex_escape(dataset.title())}}}"
            for dataset in datasets
        )
        + " \\\\",
    ]
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 2 * index}-{3 + 2 * index}}}"
        for index in range(len(datasets))
    )
    lines.extend(
        [
            cmidrules,
            "\\textbf{Method} & "
            + " & ".join("Dice & clDice" for _ in datasets)
            + " \\\\",
            "\\midrule",
        ]
    )
    for variant in core_variants:
        values: list[str] = []
        for dataset in datasets:
            values.append(_format_mean_std(summary, dataset, variant, "dice"))
            values.append(_format_mean_std(summary, dataset, variant, "cl_dice"))
        name = DISPLAY_NAMES.get(variant, variant)
        if variant in {"walltopo_uq", "walltopo_uq_v2"}:
            name = f"\\textbf{{{name}}}"
        lines.append(name + " & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (output / "table_benchmark_main.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _write_ablation_table(pairwise: pd.DataFrame, output: Path) -> None:
    if pairwise.empty:
        return
    ablation = pairwise[
        pairwise["baseline"].astype(str).str.startswith("ablation_")
        & (pairwise["metric"] == "dice")
    ]
    if ablation.empty:
        return
    datasets = sorted(ablation["dataset"].unique())
    variants = sorted(ablation["baseline"].unique())
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation effect on Dice. Positive values indicate that the"
        " full model is better than the ablated variant.}",
        "\\label{tab:benchmark_ablation}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(datasets) + "}",
        "\\toprule",
        "\\textbf{Variant} & "
        + " & ".join(_latex_escape(dataset.title()) for dataset in datasets)
        + " \\\\",
        "\\midrule",
    ]
    for variant in variants:
        values: list[str] = []
        for dataset in datasets:
            row = ablation[
                (ablation["dataset"] == dataset)
                & (ablation["baseline"] == variant)
            ]
            values.append(
                "--"
                if row.empty
                else f"{float(row.iloc[0]['improvement']):+.4f}"
            )
        lines.append(
            _latex_escape(DISPLAY_NAMES.get(variant, variant))
            + " & "
            + " & ".join(values)
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (output / "table_benchmark_ablation.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposed", default="walltopo_uq")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.root.resolve())
    if not rows:
        raise SystemExit(
            f"No test_metrics.json files with run_metadata.json found under {args.root}"
        )
    raw = pd.DataFrame.from_records(rows)
    summary = _summary(raw)
    pairwise = _pairwise(raw, proposed=str(args.proposed))

    raw.to_csv(args.output / "raw_metrics.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    if not pairwise.empty:
        pairwise.to_csv(args.output / "pairwise_vs_proposed.csv", index=False)
    _write_main_table(summary, args.output)
    _write_ablation_table(pairwise, args.output)
    with (args.output / "collection_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "runs": int(raw[["dataset", "variant", "seed"]].drop_duplicates().shape[0]),
                "datasets": sorted(raw["dataset"].unique().tolist()),
                "variants": sorted(raw["variant"].unique().tolist()),
                "metrics": sorted(raw["metric"].unique().tolist()),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote benchmark results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
