"""Build a WallTopo-UQ manifest from image and mask directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_images(directory: Path, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_mask(
    image_path: Path,
    image_root: Path,
    mask_root: Path,
    suffix: str,
    recursive: bool,
) -> Path | None:
    relative = image_path.relative_to(image_root)
    candidate = mask_root / relative.with_name(f"{relative.stem}{suffix}{relative.suffix}")
    if candidate.exists():
        return candidate
    for extension in IMAGE_EXTENSIONS:
        candidate = mask_root / relative.with_name(f"{relative.stem}{suffix}{extension}")
        if candidate.exists():
            return candidate
    if recursive:
        matches = list(mask_root.rglob(f"{image_path.stem}{suffix}.*"))
        if len(matches) == 1:
            return matches[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-suffix", default="")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--severity-csv", type=Path)
    parser.add_argument("--severity-column", default="severity")
    args = parser.parse_args()

    image_root = args.images.resolve()
    mask_root = args.masks.resolve()
    images = find_images(image_root, args.recursive)
    severity_map: dict[str, str] = {}
    if args.severity_csv:
        with args.severity_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = row.get("image") or row.get("filename") or row.get("stem")
                if key:
                    severity_map[Path(key).stem] = row[args.severity_column]

    rows: list[dict[str, str | int]] = []
    for image_path in images:
        mask_path = find_mask(
            image_path,
            image_root,
            mask_root,
            args.mask_suffix,
            args.recursive,
        )
        if mask_path is None:
            continue
        relative = image_path.relative_to(image_root)
        group = relative.parts[0] if len(relative.parts) > 1 else image_path.stem
        rows.append(
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "severity": severity_map.get(image_path.stem, ""),
                "group": group,
                "domain": "",
                "material": "",
                "device": "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "image",
            "mask",
            "severity",
            "group",
            "domain",
            "material",
            "device",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} matched image-mask pairs to {args.output.resolve()}")
    if not rows:
        raise SystemExit("No image-mask pairs were found. Check --mask-suffix and folder layout.")


if __name__ == "__main__":
    main()
