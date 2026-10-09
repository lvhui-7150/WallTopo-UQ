"""Generate JHINNO JSUB job scripts for the WallTopo-UQ experiment suite."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SUITES: dict[str, dict[str, object]] = {
    "decision": {
        "datasets": ["dais", "deepcrack", "crack500"],
        "models": ["walltopo_uq", "unet", "deeplabv3plus"],
        "seeds": [3407, 3408, 3409],
        "run_prefix": "decision",
    },
    "core": {
        "datasets": ["dais", "deepcrack", "crack500", "crackforest"],
        "models": ["walltopo_uq", "unet", "deeplabv3plus"],
        "seeds": [3407, 3408, 3409, 3410, 3411],
        "run_prefix": "benchmark",
    },
    "v2": {
        "datasets": ["dais", "deepcrack", "crack500", "crackforest"],
        "models": ["walltopo_uq_v2"],
        "seeds": [3407, 3408, 3409, 3410, 3411],
        "run_prefix": "benchmark",
    },
    "decision_v2": {
        "datasets": ["dais", "deepcrack", "crack500"],
        "models": ["walltopo_uq_v2"],
        "seeds": [3407, 3408, 3409],
        "run_prefix": "benchmark",
    },
    "ablations": {
        "datasets": ["dais", "deepcrack"],
        "models": [
            "ablation_no_mdsm",
            "ablation_single_delay",
            "ablation_no_morphology",
            "ablation_no_topology_tokens",
            "ablation_no_uncertainty",
        ],
        "seeds": [3407, 3408, 3409, 3410, 3411],
        "run_prefix": "benchmark",
    },
    "all": {
        "datasets": ["dais", "deepcrack", "crack500", "crackforest"],
        "models": ["walltopo_uq", "unet", "deeplabv3plus"],
        "seeds": [3407, 3408, 3409, 3410, 3411],
        "run_prefix": "benchmark",
    },
    "all_ablations": {
        "datasets": ["dais", "deepcrack"],
        "models": [
            "ablation_no_mdsm",
            "ablation_single_delay",
            "ablation_no_morphology",
            "ablation_no_topology_tokens",
            "ablation_no_dual_cross",
            "ablation_no_topology_loss",
            "ablation_no_geometry_consistency",
            "ablation_no_uncertainty",
        ],
        "seeds": [3407, 3408, 3409, 3410, 3411],
        "run_prefix": "benchmark",
    },
}


def _job_text(
    *,
    job_name: str,
    project_root: str,
    queue: str,
    cores: int,
    gpu_request: str,
    walltime: int,
    env_bin: str,
    pretrained_baseline: bool,
    num_workers: int,
    dataset: str,
    model: str,
    seed: int,
    run_prefix: str,
) -> str:
    pretrained_flag = (
        "--pretrained-baseline"
        if pretrained_baseline
        else "--no-pretrained-baseline"
    )
    return f"""#!/bin/bash
#JSUB -q {queue}
#JSUB -n {cores}
#JSUB -gpgpu {gpu_request}
#JSUB -J {job_name}
#JSUB -W {walltime}
#JSUB -o logs/%J.out
#JSUB -e logs/%J.err

set -euo pipefail
PROJECT_ROOT="${{PROJECT_ROOT:-{project_root}}}"
cd "$PROJECT_ROOT"

WALLTOPO_ENV_BIN="${{WALLTOPO_ENV_BIN:-{env_bin}}}"
export PATH="$WALLTOPO_ENV_BIN:$PATH"
unset PYTHONPATH || true
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${{NSLOTS:-{cores}}}"
export MKL_NUM_THREADS="${{NSLOTS:-{cores}}}"
export TORCH_HOME="$HOME/.cache/torch"

mkdir -p logs runs results
python scripts/run_benchmark_suite.py \\
  --profile full \\
  --datasets {dataset} \\
  --models {model} \\
  --seeds {seed} \\
  --run-prefix {run_prefix} \\
  --resume \\
  --no-collect \\
  {pretrained_flag} \\
  --num-workers {num_workers} \\
  --device cuda
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="decision",
    )
    parser.add_argument(
        "--project-root",
        default="$HOME/projects/WallTopo-UQ-HPC",
    )
    parser.add_argument("--queue", default="gpu4")
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument(
        "--gpu-request",
        default="1",
        help="Use '1' for a full GPU or e.g. '1 mig=2' on MIG queues.",
    )
    parser.add_argument("--walltime", type=int, default=720)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--env-bin",
        default="$HOME/miniconda3/bin",
        help="Existing environment bin directory; no new env is installed.",
    )
    parser.add_argument(
        "--pretrained-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ImageNet initialization for the DeepLabV3+ baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hpc_jobs"),
    )
    args = parser.parse_args()

    suite = SUITES[args.suite]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for dataset in suite["datasets"]:
        for model in suite["models"]:
            for seed in suite["seeds"]:
                stem = f"{dataset}__{model}__seed_{seed}"
                job_name = f"wt_{dataset[:8]}_{model[:18]}_{seed}"
                path = output / f"{stem}.sub"
                path.write_text(
                    _job_text(
                        job_name=job_name,
                        project_root=args.project_root,
                        queue=args.queue,
                        cores=args.cores,
                        gpu_request=args.gpu_request,
                        walltime=args.walltime,
                        env_bin=args.env_bin,
                        pretrained_baseline=args.pretrained_baseline,
                        num_workers=int(args.num_workers),
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        run_prefix=str(suite["run_prefix"]),
                    ),
                    encoding="utf-8",
                )
                generated.append(path)

    submit = output / f"submit_{args.suite}.sh"
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        "mkdir -p logs",
        "",
    ]
    for path in generated:
        lines.extend(
            [
                (
                    'result_path="../../runs/'
                    f'{suite["run_prefix"]}/{path.stem.split("__")[0]}/'
                    f'{path.stem.split("__")[1]}/'
                    f'seed_{path.stem.split("__")[2].replace("seed_", "")}'
                    '/test_metrics.json"'
                ),
                'if [ -f "$result_path" ]; then',
                f'  echo "skip completed {path.name}"',
                "else",
                f'  echo "submit {path.name}"',
                f'  jsub < "{path.name}"',
                "fi",
            ]
        )
    submit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Generated {len(generated)} JSUB scripts for suite={args.suite} "
        f"in {output}"
    )


if __name__ == "__main__":
    main()
