"""Add dataset/domain metadata to a manifest without changing image paths."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--domain")
    parser.add_argument("--material")
    parser.add_argument("--device")
    parser.add_argument(
        "--group-prefix",
        default="",
        help="Prefix existing group IDs to avoid collisions across source datasets.",
    )
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Input manifest is empty.")
    fieldnames = list(rows[0].keys())
    for key in ("group", "domain", "material", "device"):
        if key not in fieldnames:
            fieldnames.append(key)
    for row in rows:
        if args.group_prefix:
            row["group"] = f"{args.group_prefix}_{row.get('group', '')}".rstrip("_")
        if args.domain is not None:
            row["domain"] = args.domain
        if args.material is not None:
            row["material"] = args.material
        if args.device is not None:
            row["device"] = args.device
    output = args.output or args.manifest
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Annotated {len(rows)} rows -> {output.resolve()}")


if __name__ == "__main__":
    main()
