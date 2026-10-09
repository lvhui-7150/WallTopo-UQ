"""Merge multiple dataset manifests into one CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED = ["image", "mask"]
OPTIONAL = ["severity", "group", "domain", "material", "device"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fieldnames = REQUIRED + OPTIONAL
    rows: list[dict[str, str]] = []
    for manifest in args.manifests:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("image") or not row.get("mask"):
                    continue
                rows.append({key: row.get(key, "") for key in fieldnames})
    if not rows:
        raise SystemExit("No valid rows were found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} rows from {len(args.manifests)} manifests -> {args.output}")


if __name__ == "__main__":
    main()
