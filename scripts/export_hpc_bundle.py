"""Export a portable Linux/HPC bundle without Windows absolute paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401


CODE_DIRECTORIES = [
    "configs",
    "hpc",
    "hpc_jobs",
    "scripts",
    "tests",
    "walltopo_uq",
]

CODE_FILES = [
    "README.md",
    "EXPERIMENT_PROTOCOL_ZH.md",
    "requirements.txt",
    "requirements-benchmark.txt",
    "requirements-hpc.txt",
    "pytest.ini",
]

DATASETS = ["dais", "deepcrack", "crack500", "crackforest"]
SPLIT_FILES = ["manifest.csv", "train.csv", "val.csv", "test.csv"]


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            "generated",
        ),
    )


def _portable_name(path: Path, kind: str) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{digest}_{path.name}"


def _export_dataset(source_root: Path, bundle_root: Path, dataset: str) -> None:
    source_dir = source_root / "data" / "raw" / dataset
    if not source_dir.exists():
        return
    output_dir = bundle_root / "data" / "raw" / dataset
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for csv_name in SPLIT_FILES:
        source_csv = source_dir / csv_name
        if not source_csv.exists():
            continue
        with source_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        for row in rows:
            for key, destination_dir in (
                ("image", image_dir),
                ("mask", mask_dir),
            ):
                value = row.get(key, "")
                if not value:
                    continue
                source_path = Path(value)
                if not source_path.is_absolute():
                    source_path = (source_csv.parent / source_path).resolve()
                target_name = _portable_name(source_path, key)
                target_path = destination_dir / target_name
                if not target_path.exists():
                    shutil.copy2(source_path, target_path)
                row[key] = str(
                    Path("images" if key == "image" else "masks") / target_name
                ).replace("\\", "/")
        with (output_dir / csv_name).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"exported {dataset}/{csv_name}: {len(rows)} rows")


def _write_readme(
    bundle_root: Path,
    zip_path: Path | None,
    include_weights: bool,
) -> None:
    weight_note = (
        "- cached ResNet-50 weights used by the DeepLabV3+ baseline;\n"
        if include_weights
        else "- no torch weight cache; DeepLabV3+ runs without ImageNet initialization;\n"
    )
    text = f"""# WallTopo-UQ HPC Upload Bundle

This bundle contains:

- Linux-compatible WallTopo-UQ code;
- relative-path manifests and the four prepared public datasets;
- JHINNO/JSUB job templates for the CDUT "Lizhiyun" platform;
{weight_note.rstrip()}
- no Windows absolute paths and no training runs.

Recommended upload target:

```text
$HOME/projects/WallTopo-UQ-HPC
```

Check the existing environment:

```bash
cd "$HOME/projects/WallTopo-UQ-HPC"
export PATH="$HOME/miniconda3/bin:$PATH"
unset PYTHONPATH
python hpc/check_env.py
```

If anything is missing, install only the missing packages or optionally run
`bash hpc/setup_hpc_env.sh` to create a dedicated environment.

Run a small smoke test on gpu4:

```bash
mkdir -p logs
jsub < hpc/smoke_gpu.sub
```

Submit the decision suite:

```bash
bash hpc_jobs/decision/submit_decision.sh
```

Submit the full core suite:

```bash
bash hpc_jobs/core/submit_core.sh
```

Submit ablations after the main comparison:

```bash
bash hpc_jobs/ablations/submit_ablations.sh
```

Submit the v2 segmentation-stability model on the decision subset:

```bash
bash hpc_jobs/decision_v2/submit_decision_v2.sh
```

Submit the full v2 comparison (the completed U-Net/DeepLab runs are reused):

```bash
bash hpc_jobs/v2/submit_v2.sh
```

After all jobs finish:

```bash
jsub < hpc/finalize_hpc.sub
```

Finalize with v2 as the proposed model:

```bash
jsub < hpc/finalize_v2.sub
```

The generated scripts use `#JSUB`, not Slurm `#SBATCH`.
"""
    (bundle_root / "README_FIRST_HPC.md").write_text(text, encoding="utf-8")
    if zip_path is not None:
        print(f"zip archive: {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "dist",
    )
    parser.add_argument(
        "--name",
        default=f"WallTopo-UQ-HPC-{datetime.now():%Y%m%d_%H%M}",
    )
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Export only code, configs, and HPC jobs; omit all datasets.",
    )
    parser.add_argument(
        "--omit-torch-weights",
        action="store_true",
        help="Do not include the 102 MB ResNet-50 cache.",
    )
    args = parser.parse_args()

    bundle_root = args.output.resolve() / args.name
    bundle_root.mkdir(parents=True, exist_ok=False)
    for directory in CODE_DIRECTORIES:
        _copy_tree(ROOT / directory, bundle_root / directory)
    for filename in CODE_FILES:
        source = ROOT / filename
        if source.exists():
            shutil.copy2(source, bundle_root / filename)

    if not args.code_only:
        for dataset in DATASETS:
            _export_dataset(ROOT, bundle_root, dataset)

        synthetic_source = ROOT / "data" / "synthetic"
        if synthetic_source.exists():
            shutil.copytree(
                synthetic_source,
                bundle_root / "data" / "synthetic",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "derived_cache",
                    "__pycache__",
                    "dataset_summary.json",
                ),
            )

    weight = (
        Path.home()
        / ".cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "resnet50-11ad3fa6.pth"
    )
    if weight.exists() and not args.omit_torch_weights and not args.code_only:
        target = (
            bundle_root
            / "torch_cache"
            / "hub"
            / "checkpoints"
            / weight.name
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weight, target)

    _write_readme(
        bundle_root,
        None,
        include_weights=(
            weight.exists()
            and not args.omit_torch_weights
            and not args.code_only
        ),
    )
    if not args.no_zip:
        zip_path = Path(
            shutil.make_archive(
                str(bundle_root),
                "zip",
                root_dir=bundle_root.parent,
                base_dir=bundle_root.name,
            )
        )
        print(f"zip archive: {zip_path}")
    print(f"bundle root: {bundle_root}")


if __name__ == "__main__":
    main()
