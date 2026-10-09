"""Collect leave-one-domain-out and corruption-uncertainty results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401


METRICS = [
    "dice",
    "iou",
    "boundary_f1",
    "cl_dice",
    "junction_f1",
    "break_rate",
    "width_mae_px",
    "ece",
    "brier",
    "aurc",
    "error_detection_auroc",
    "selective_error",
]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_lodo(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("cross_domain_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenario = str(payload["scenario"])
        model = str(payload["model"])
        seed = int(payload["seed"])
        for target, metrics in payload.get("targets", {}).items():
            for metric in METRICS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    rows.append(
                        {
                            "scenario": scenario,
                            "target": target,
                            "model": model,
                            "seed": seed,
                            "metric": metric,
                            "value": float(value),
                        }
                    )
        for target, metrics in payload.get("targets", {}).items():
            for metric in METRICS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    summary_rows.append(
                        {
                            "dataset": target,
                            "model": model,
                            "metric": metric,
                            "value": float(value),
                        }
                    )
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in summary_rows:
        grouped[
            (
                str(row["dataset"]),
                str(row["model"]),
                str(row["metric"]),
            )
        ].append(float(row["value"]))
    summary = [
        {
            "dataset": dataset,
            "model": model,
            "metric": metric,
            "mean": _mean(values),
            "count": len(values),
        }
        for (dataset, model, metric), values in sorted(grouped.items())
    ]
    return rows, summary


def _collect_corruptions(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("uncertainty_corruptions_test.json")):
        try:
            relative = path.relative_to(root)
            dataset = relative.parts[0]
        except (ValueError, IndexError):
            dataset = "unknown"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for corruption_key, methods in payload.items():
            if not isinstance(methods, dict):
                continue
            for method, metrics in methods.items():
                if not isinstance(metrics, dict):
                    continue
                for metric in METRICS + ["composite", "seconds_per_image"]:
                    value = metrics.get(metric)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        rows.append(
                            {
                                "dataset": dataset,
                                "corruption": corruption_key,
                                "method": method,
                                "metric": metric,
                                "value": float(value),
                            }
                        )
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["dataset"]),
                str(row["corruption"]),
                str(row["method"]),
                str(row["metric"]),
            )
        ].append(float(row["value"]))
    summary = [
        {
            "dataset": dataset,
            "corruption": corruption,
            "method": method,
            "metric": metric,
            "mean": _mean(values),
            "count": len(values),
        }
        for (dataset, corruption, method, metric), values in sorted(grouped.items())
    ]
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / "results" / "round2"
    )
    runs_root = project_root / "runs"
    lodo_rows, lodo_summary = _collect_lodo(runs_root / "reviewer_lodo")
    corruption_rows, corruption_summary = _collect_corruptions(
        runs_root / "benchmark"
    )
    _write_csv(output_root / "lodo_raw.csv", lodo_rows)
    _write_csv(output_root / "lodo_summary.csv", lodo_summary)
    _write_csv(output_root / "corruption_raw.csv", corruption_rows)
    _write_csv(output_root / "corruption_summary.csv", corruption_summary)
    metadata = {
        "project_root": str(project_root),
        "lodo_runs": len(lodo_rows),
        "corruption_runs": len(corruption_rows),
        "output_root": str(output_root),
    }
    (output_root / "collection_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
