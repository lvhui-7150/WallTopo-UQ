"""Create leakage-aware train/validation/test manifests by group."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-column", default="group")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Input manifest is empty.")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        group = row.get(args.group_column) or Path(row["image"]).stem
        groups.setdefault(group, []).append(row)

    group_names = sorted(groups)
    random.Random(args.seed).shuffle(group_names)
    train_end = int(len(group_names) * args.train_ratio)
    val_end = train_end + int(len(group_names) * args.val_ratio)
    split_groups = {
        "train": group_names[:train_end],
        "val": group_names[train_end:val_end],
        "test": group_names[val_end:],
    }
    if not split_groups["val"] and len(group_names) >= 2:
        split_groups["val"].append(split_groups["train"].pop())
    if not split_groups["test"] and len(group_names) >= 2:
        split_groups["test"].append(split_groups["train"].pop())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for split, selected_groups in split_groups.items():
        selected_rows = [row for group in selected_groups for row in groups[group]]
        path = args.output_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_rows)
        print(
            f"{split}: {len(selected_rows)} samples, {len(selected_groups)} groups -> {path}"
        )


if __name__ == "__main__":
    main()
