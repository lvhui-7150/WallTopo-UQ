"""Collect ablation, topology, cross-domain, and uncertainty results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT  # noqa: F401


def _collect_benchmark(prefix: str, output: Path) -> None:
    run_root = ROOT / "runs" / prefix
    if not run_root.exists():
        print(f"skip missing benchmark runs: {run_root}")
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "collect_benchmark_results.py"),
            "--root",
            str(run_root),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )


def _write_cross_domain_table(output: Path) -> Path | None:
    rows: list[dict] = []
    root = ROOT / "runs" / "reviewer_crossdomain"
    for summary_path in sorted(root.glob("*/*/seed_*/cross_domain_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for dataset, metrics in payload["targets"].items():
            rows.append(
                {
                    "scenario": payload["scenario"],
                    "model": payload["model"],
                    "seed": payload["seed"],
                    "target": dataset,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    },
                }
            )
    if not rows:
        return None
    path = output / "cross_domain_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_uncertainty_table(output: Path) -> Path | None:
    rows: list[dict] = []
    root = ROOT / "runs" / "benchmark"
    for result_path in sorted(root.glob("*/*/uncertainty_baselines_test.json")):
        dataset = result_path.parents[1].name
        model = result_path.parent.name
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for method, result in payload.items():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    **result["metrics"],
                }
            )
    if not rows:
        return None
    path = output / "uncertainty_baselines.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "reviewer_summary",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    _collect_benchmark("reviewer_ablation", args.output / "ablation")
    _collect_benchmark("reviewer_topology", args.output / "topology")
    cross_domain = _write_cross_domain_table(args.output)
    uncertainty = _write_uncertainty_table(args.output)

    summary = {
        "ablation": str(args.output / "ablation"),
        "topology": str(args.output / "topology"),
        "cross_domain": str(cross_domain) if cross_domain else None,
        "uncertainty": str(uncertainty) if uncertainty else None,
    }
    with (args.output / "reviewer_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
