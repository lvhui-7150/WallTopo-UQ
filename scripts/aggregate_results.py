"""Aggregate test metrics from multiple runs into mean and standard deviation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--pattern", default="*metrics.json")
    parser.add_argument("--output", type=Path, default=Path("runs/aggregated_metrics.csv"))
    args = parser.parse_args()

    values: dict[str, list[float]] = defaultdict(list)
    run_names: dict[str, list[str]] = defaultdict(list)
    for path in sorted(args.root.rglob(args.pattern)):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                values[key].append(float(value))
                run_names[key].append(str(path.parent))
    if not values:
        raise SystemExit("No numeric metrics were found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "mean", "std", "count", "runs"],
        )
        writer.writeheader()
        for key in sorted(values):
            array = np.asarray(values[key], dtype=np.float64)
            writer.writerow(
                {
                    "metric": key,
                    "mean": float(array.mean()),
                    "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
                    "count": int(array.size),
                    "runs": ";".join(run_names[key]),
                }
            )
    print(f"Wrote aggregated results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
