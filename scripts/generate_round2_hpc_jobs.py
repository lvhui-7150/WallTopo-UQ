"""Generate JSUB jobs for the reviewer round-2 experiments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401


DATASETS = ["crack500", "crackforest", "dais", "deepcrack"]
ABLATION_MODELS = [
    "ablation_no_mdsm",
    "ablation_single_delay",
    "ablation_no_morphology",
    "ablation_no_topology_tokens",
    "ablation_no_dual_cross",
    "ablation_no_topology_loss",
    "ablation_no_geometry_consistency",
    "ablation_no_uncertainty",
]
LODO_SCENARIOS = [
    "leave_crack500",
    "leave_crackforest",
    "leave_dais",
    "leave_deepcrack",
]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def _job_body(
    command: list[str],
    name: str,
    walltime: int,
    result_path: str,
) -> str:
    quoted = " \\\n  ".join(command)
    return f"""#!/bin/bash
#JSUB -q gpu4
#JSUB -n 1
#JSUB -gpgpu 1
#JSUB -J {_safe_name(name)}
#JSUB -W {int(walltime)}
#JSUB -o logs/%J.out
#JSUB -e logs/%J.err

#WALLTOPO_RESULT={result_path}

set -euo pipefail
PROJECT_ROOT="${{PROJECT_ROOT:-$HOME/projects/WallTopo-UQ-HPC}}"
cd "$PROJECT_ROOT"

WALLTOPO_ENV_BIN="${{WALLTOPO_ENV_BIN:-$HOME/miniconda3/envs/aenet/bin}}"
export PATH="$WALLTOPO_ENV_BIN:$PATH"
unset PYTHONPATH || true
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export OMP_NUM_THREADS="${{NSLOTS:-1}}"
export MKL_NUM_THREADS="${{NSLOTS:-1}}"
export TORCH_HOME="$HOME/.cache/torch"

mkdir -p logs runs results
python \\
  {quoted}
"""


def _write_job(
    output_dir: Path,
    name: str,
    command: list[str],
    walltime: int,
    result_path: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_name(name)}.sub"
    path.write_text(
        _job_body(command, name, walltime, result_path),
        encoding="utf-8",
    )
    return path


def _result_probe(job_path: Path) -> str:
    text = job_path.read_text(encoding="utf-8")
    match = re.search(r"^#WALLTOPO_RESULT=(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else "../../../runs/.completed"


def _write_submit_script(
    output_dir: Path,
    job_paths: list[Path],
    name: str,
) -> Path:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        "mkdir -p logs",
        "",
    ]
    for path in job_paths:
        result_path = _result_probe(path)
        lines.extend(
            [
                f'result_path="{result_path}"',
                'if [ -f "$result_path" ]; then',
                f'  echo "skip completed {path.name}"',
                "else",
                f'  echo "submit {path.name}"',
                f'  jsub < "{path.name}"',
                "fi",
                "",
            ]
        )
    submit_path = output_dir / f"submit_{name}.sh"
    submit_path.write_text("\n".join(lines), encoding="utf-8")
    submit_path.chmod(0o755)
    return submit_path


def _baseline_jobs(
    output_dir: Path,
    datasets: list[str],
    seeds: list[int],
    profile: str,
    batch_size: int,
) -> list[Path]:
    jobs: list[Path] = []
    for dataset in datasets:
        for seed in seeds:
            name = f"unetplusplus__{dataset}__seed_{seed}"
            command = [
                "scripts/run_benchmark_suite.py",
                "--profile",
                profile,
                "--datasets",
                dataset,
                "--models",
                "smp_unetplusplus",
                "--seeds",
                str(seed),
                "--run-prefix",
                "benchmark",
                "--resume",
                "--no-collect",
                "--no-pretrained-baseline",
                "--num-workers",
                "0",
                "--batch-size",
                str(batch_size),
                "--device",
                "cuda",
            ]
            jobs.append(
                _write_job(
                    output_dir,
                    name,
                    command,
                    720,
                    (
                        "../../../runs/benchmark/"
                        f"{dataset}/smp_unetplusplus/seed_{seed}/"
                        "test_metrics.json"
                    ),
                )
            )
    return jobs


def _ablation_jobs(
    output_dir: Path,
    datasets: list[str],
    seeds: list[int],
    profile: str,
    batch_size: int,
) -> list[Path]:
    jobs: list[Path] = []
    for dataset in datasets:
        for model in ABLATION_MODELS:
            for seed in seeds:
                name = f"ablation__{dataset}__{model}__seed_{seed}"
                command = [
                    "scripts/run_benchmark_suite.py",
                    "--profile",
                    profile,
                    "--datasets",
                    dataset,
                    "--models",
                    model,
                    "--seeds",
                    str(seed),
                    "--run-prefix",
                    "reviewer_ablation",
                    "--resume",
                    "--no-collect",
                    "--no-pretrained-baseline",
                    "--num-workers",
                    "0",
                    "--batch-size",
                    str(batch_size),
                    "--device",
                    "cuda",
                ]
                jobs.append(
                    _write_job(
                        output_dir,
                        name,
                        command,
                        720,
                        (
                            "../../../runs/reviewer_ablation/"
                            f"{dataset}/{model}/seed_{seed}/test_metrics.json"
                        ),
                    )
                )
    return jobs


def _lodo_jobs(
    output_dir: Path,
    models: list[str],
    seeds: list[int],
    batch_size: int,
) -> list[Path]:
    jobs: list[Path] = []
    for scenario in LODO_SCENARIOS:
        for model in models:
            for seed in seeds:
                name = f"lodo__{scenario}__{model}__seed_{seed}"
                command = [
                    "scripts/run_cross_domain_suite.py",
                    "--scenario",
                    scenario,
                    "--model",
                    model,
                    "--seed",
                    str(seed),
                    "--epochs",
                    "120",
                    "--patience",
                    "25",
                    "--num-workers",
                    "0",
                    "--batch-size",
                    str(batch_size),
                    "--device",
                    "cuda",
                    "--run-prefix",
                    "reviewer_lodo",
                    "--resume",
                ]
                jobs.append(
                    _write_job(
                        output_dir,
                        name,
                        command,
                        1440,
                        (
                            "../../../runs/reviewer_lodo/"
                            f"{scenario}/{model}/seed_{seed}/"
                            "cross_domain_summary.json"
                        ),
                    )
                )
    return jobs


def _uncertainty_corruption_jobs(
    output_dir: Path,
    datasets: list[str],
    seeds: list[int],
    passes: int,
    batch_size: int,
) -> list[Path]:
    jobs: list[Path] = []
    for dataset in datasets:
        config = (
            f"configs/generated/benchmark/{dataset}/walltopo_uq/"
            f"seed_{seeds[0]}.yaml"
        )
        checkpoints = " ".join(
            f"runs/benchmark/{dataset}/walltopo_uq/seed_{seed}/"
            "checkpoints/best.pt"
            for seed in seeds
        )
        output = (
            f"runs/benchmark/{dataset}/walltopo_uq/"
            "uncertainty_corruptions_test.json"
        )
        name = f"uncertainty_corruptions__{dataset}"
        command = [
            "scripts/evaluate_uncertainty_corruptions.py",
            "--config",
            config,
            "--checkpoints",
            checkpoints,
            "--split",
            "test",
            "--methods",
            "eus",
            "softmax",
            "mc_dropout",
            "ensemble",
            "--passes",
            str(passes),
            "--dropout-p",
            "0.10",
            "--coverage",
            "0.80",
            "--device",
            "cuda",
            "--output",
            output,
        ]
        jobs.append(
            _write_job(
                output_dir,
                name,
                command,
                1440,
                f"../../../{output}",
            )
        )
    return jobs


def _efficiency_jobs(
    output_dir: Path,
    models: list[str],
    batch_size: int,
) -> list[Path]:
    output = "results/round2/efficiency.csv"
    command = [
        "scripts/benchmark_models.py",
        "--config",
        "configs/dais.yaml",
        "--models",
        *models,
        "--size",
        "256",
        "--batch-size",
        str(batch_size),
        "--iterations",
        "20",
        "--warmup",
        "5",
        "--device",
        "cuda",
        "--output",
        output,
    ]
    return [
        _write_job(
            output_dir,
            "efficiency__all_models",
            command,
            120,
            f"../../../{output}",
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "hpc_jobs" / "round2",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409, 3410, 3411])
    parser.add_argument("--profile", default="full")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument(
        "--baseline-datasets",
        nargs="+",
        default=DATASETS,
    )
    parser.add_argument(
        "--ablation-datasets",
        nargs="+",
        default=["crack500", "crackforest"],
    )
    parser.add_argument(
        "--lodo-models",
        nargs="+",
        default=["walltopo_uq", "unet", "smp_unetplusplus"],
    )
    parser.add_argument(
        "--corruption-datasets",
        nargs="+",
        default=DATASETS,
    )
    parser.add_argument(
        "--efficiency-models",
        nargs="+",
        default=["walltopo_uq", "unet", "deeplabv3plus", "smp_unetplusplus"],
    )
    args = parser.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stages = {
        "baseline_unetplusplus": _baseline_jobs(
            root / "baseline_unetplusplus",
            list(args.baseline_datasets),
            list(args.seeds),
            args.profile,
            int(args.batch_size),
        ),
        "ablation_missing_datasets": _ablation_jobs(
            root / "ablation_missing_datasets",
            list(args.ablation_datasets),
            list(args.seeds),
            args.profile,
            int(args.batch_size),
        ),
        "lodo": _lodo_jobs(
            root / "lodo",
            list(args.lodo_models),
            list(args.seeds),
            int(args.batch_size),
        ),
        "uncertainty_corruptions": _uncertainty_corruption_jobs(
            root / "uncertainty_corruptions",
            list(args.corruption_datasets),
            list(args.seeds),
            int(args.passes),
            int(args.batch_size),
        ),
        "efficiency": _efficiency_jobs(
            root / "efficiency",
            list(args.efficiency_models),
            int(args.batch_size),
        ),
    }
    for stage, jobs in stages.items():
        submit = _write_submit_script(root / stage, jobs, stage)
        print(f"{stage}: {len(jobs)} jobs -> {submit}")


if __name__ == "__main__":
    main()
