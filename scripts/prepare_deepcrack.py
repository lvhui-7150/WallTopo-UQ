"""Build official-split manifests for the DeepCrack dataset."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def _rows(
    image_dir: Path,
    mask_dir: Path,
    prefix: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        rows.append(
            {
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "severity": "",
                "group": f"deepcrack_{prefix}_{image_path.stem}",
                "domain": "road",
                "material": "asphalt",
                "device": "camera",
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/raw/sources/deepcrack_dataset_extracted"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/deepcrack"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    official_train = _rows(
        source_root / "train_img",
        source_root / "train_lab",
        "train",
    )
    test_rows = _rows(
        source_root / "test_img",
        source_root / "test_lab",
        "test",
    )
    if not official_train or not test_rows:
        raise SystemExit("DeepCrack image-mask pairs were not found.")

    groups: dict[str, list[dict[str, str]]] = {}
    for row in official_train:
        groups.setdefault(row["group"], []).append(row)
    group_names = sorted(groups)
    random.Random(args.seed).shuffle(group_names)
    val_count = min(
        max(1, int(round(len(group_names) * args.val_ratio))),
        len(group_names) - 1,
    )
    val_groups = set(group_names[:val_count])
    train_rows = [
        row
        for group in group_names
        if group not in val_groups
        for row in groups[group]
    ]
    val_rows = [
        row
        for group in group_names
        if group in val_groups
        for row in groups[group]
    ]

    output_dir = args.output_dir.resolve()
    _write(output_dir / "train.csv", train_rows)
    _write(output_dir / "val.csv", val_rows)
    _write(output_dir / "test.csv", test_rows)
    _write(output_dir / "manifest.csv", train_rows + val_rows + test_rows)
    print(
        f"DeepCrack official split: train={len(train_rows)} "
        f"val={len(val_rows)} test={len(test_rows)} -> {output_dir}"
    )


if __name__ == "__main__":
    main()
