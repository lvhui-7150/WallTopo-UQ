"""Generate JSUB job files for the reviewer-response experiment package."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401


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

TOPOLOGY_MODELS = [
    "unet_cldice",
    "unet_connectivity",
]

CROSS_DOMAIN_SCENARIOS = [
    "pavement_to_masonry",
    "masonry_to_pavement",
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
    return submit_path


def _result_probe(job_path: Path) -> str:
    text = job_path.read_text(encoding="utf-8")
    match = re.search(r"^#WALLTOPO_RESULT=(.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "../../../runs/reviewer/.completed"


def _common_env() -> str:
    return ""


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


def _topology_jobs(
    output_dir: Path,
    datasets: list[str],
    seeds: list[int],
    profile: str,
    batch_size: int,
) -> list[Path]:
    jobs: list[Path] = []
    for dataset in datasets:
        for model in TOPOLOGY_MODELS:
            for seed in seeds:
                name = f"topology__{dataset}__{model}__seed_{seed}"
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
                    "reviewer_topology",
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
                            "../../../runs/reviewer_topology/"
                            f"{dataset}/{model}/seed_{seed}/test_metrics.json"
                        ),
                    )
                )
    return jobs


def _cross_domain_jobs(
    output_dir: Path,
    models: list[str],
    seeds: list[int],
    batch_size: int,
) -> list[Path]:
    jobs: list[Path] = []
    for scenario in CROSS_DOMAIN_SCENARIOS:
        for model in models:
            for seed in seeds:
                name = f"crossdomain__{scenario}__{model}__seed_{seed}"
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
                    "reviewer_crossdomain",
                    "--resume",
                ]
                jobs.append(
                    _write_job(
                        output_dir,
                        name,
                        command,
                        1440,
                        (
                            "../../../runs/reviewer_crossdomain/"
                            f"{scenario}/{model}/seed_{seed}/"
                            "cross_domain_summary.json"
                        ),
                    )
                )
    return jobs


def _uncertainty_jobs(
    output_dir: Path,
    datasets: list[str],
    models: list[str],
    seeds: list[int],
    passes: int,
) -> list[Path]:
    jobs: list[Path] = []
    for dataset in datasets:
        for model in models:
            name = f"uncertainty__{dataset}__{model}"
            config = (
                f"configs/generated/benchmark/{dataset}/{model}/"
                f"seed_{seeds[0]}.yaml"
            )
            checkpoints = " ".join(
                f"runs/benchmark/{dataset}/{model}/seed_{seed}/"
                "checkpoints/best.pt"
                for seed in seeds
            )
            output = (
                f"runs/benchmark/{dataset}/{model}/"
                "uncertainty_baselines_test.json"
            )
            command = [
                "scripts/evaluate_uncertainty_baselines.py",
                "--config",
                config,
                "--checkpoints",
                checkpoints,
                "--split",
                "test",
                "--methods",
                "softmax",
                "mc_dropout",
                "ensemble",
                "--passes",
                str(passes),
                "--dropout-p",
                "0.10",
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
                    480,
                    f"../../../{output}",
                )
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "hpc_jobs" / "reviewer",
    )
    parser.add_argument(
        "--ablation-datasets",
        nargs="+",
        default=["dais", "deepcrack"],
    )
    parser.add_argument(
        "--topology-datasets",
        nargs="+",
        default=["crackforest", "dais", "deepcrack"],
    )
    parser.add_argument(
        "--uncertainty-datasets",
        nargs="+",
        default=["crack500", "crackforest", "dais", "deepcrack"],
    )
    parser.add_argument(
        "--cross-domain-models",
        nargs="+",
        default=["walltopo_uq", "unet"],
    )
    parser.add_argument(
        "--uncertainty-models",
        nargs="+",
        default=["walltopo_uq", "walltopo_uq_v2", "unet"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[3407, 3408, 3409, 3410, 3411],
    )
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--profile", default="full")
    args = parser.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stages = {
        "ablation": _ablation_jobs(
            root / "ablation",
            args.ablation_datasets,
            args.seeds,
            args.profile,
            args.batch_size,
        ),
        "topology": _topology_jobs(
            root / "topology",
            args.topology_datasets,
            args.seeds,
            args.profile,
            args.batch_size,
        ),
        "crossdomain": _cross_domain_jobs(
            root / "crossdomain",
            args.cross_domain_models,
            args.seeds,
            args.batch_size,
        ),
        "uncertainty": _uncertainty_jobs(
            root / "uncertainty",
            args.uncertainty_datasets,
            args.uncertainty_models,
            args.seeds,
            args.passes,
        ),
    }
    for stage, jobs in stages.items():
        submit = _write_submit_script(root / stage, jobs, stage)
        print(f"{stage}: {len(jobs)} jobs -> {submit}")


if __name__ == "__main__":
    main()
