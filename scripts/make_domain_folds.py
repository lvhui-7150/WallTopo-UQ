"""Create leave-one-domain-out manifests without group-level leakage."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "domain"


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain-column", default="domain")
    parser.add_argument("--group-column", default="group")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Input manifest is empty.")
    fieldnames = list(rows[0].keys())
    if args.domain_column not in fieldnames:
        raise SystemExit(f"Missing domain column: {args.domain_column}")
    if args.group_column not in fieldnames:
        raise SystemExit(f"Missing group column: {args.group_column}")

    domains = sorted(
        {row[args.domain_column].strip() for row in rows if row[args.domain_column].strip()}
    )
    if len(domains) < 2:
        raise SystemExit("At least two non-empty domains are required.")

    summary: dict[str, dict[str, int | str]] = {}
    for fold_index, held_out in enumerate(domains):
        test_rows = [row for row in rows if row[args.domain_column].strip() == held_out]
        remaining = [row for row in rows if row[args.domain_column].strip() != held_out]
        groups: dict[str, list[dict[str, str]]] = {}
        for row in remaining:
            group = row[args.group_column].strip() or Path(row["image"]).stem
            groups.setdefault(group, []).append(row)
        group_names = sorted(groups)
        random.Random(args.seed + fold_index).shuffle(group_names)
        val_count = max(1, int(round(len(group_names) * args.val_ratio)))
        val_count = min(val_count, max(1, len(group_names) - 1))
        val_groups = set(group_names[:val_count])
        train_rows = [row for group in group_names if group not in val_groups for row in groups[group]]
        val_rows = [row for group in group_names if group in val_groups for row in groups[group]]

        fold_dir = args.output_dir / f"fold_{_slug(held_out)}"
        _write_rows(fold_dir / "train.csv", fieldnames, train_rows)
        _write_rows(fold_dir / "val.csv", fieldnames, val_rows)
        _write_rows(fold_dir / "test.csv", fieldnames, test_rows)
        summary[held_out] = {
            "manifest": str(fold_dir.resolve()),
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
            "test_domain": held_out,
        }
        print(
            f"fold={held_out}: train={len(train_rows)} val={len(val_rows)} "
            f"test={len(test_rows)} -> {fold_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "folds.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
