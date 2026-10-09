"""Audit reviewer-job completion and classify incomplete jobs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


LOG_PATTERN = re.compile(
    r"reviewer_(ablation|topology|crossdomain)/([^/]+)/([^/]+)/seed_(\d+)\.yaml"
)


def classify_error(text: str) -> str:
    if "weight of size" in text:
        return "mdsm_channel_mismatch"
    if "Sizes of tensors must match" in text:
        return "dual_cross_size_mismatch"
    if re.search(r"CUDA out of memory|out of memory", text, flags=re.IGNORECASE):
        return "cuda_oom"
    if "BadZipFile" in text:
        return "derived_cache_corruption"
    if "Traceback" in text:
        return "other_traceback"
    if text.strip():
        return "incomplete_without_exception"
    return "missing_log"


def expected_result(
    project_root: Path,
    stage: str,
    dataset_or_scenario: str,
    variant_or_model: str,
    seed: int,
) -> Path:
    if stage in {"ablation", "topology"}:
        return (
            project_root
            / "runs"
            / f"reviewer_{stage}"
            / dataset_or_scenario
            / variant_or_model
            / f"seed_{seed}"
            / "test_metrics.json"
        )
    return (
        project_root
        / "runs"
        / "reviewer_crossdomain"
        / dataset_or_scenario
        / variant_or_model
        / f"seed_{seed}"
        / "cross_domain_summary.json"
    )


def load_log_errors(project_root: Path) -> dict[tuple[str, str, str, int], str]:
    errors: dict[tuple[str, str, str, int], str] = {}
    job_root = project_root / "hpc_jobs" / "reviewer"
    for stage in ("ablation", "topology", "crossdomain"):
        for log_path in (job_root / stage / "logs").glob("*.err"):
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            for match in LOG_PATTERN.finditer(text):
                key = (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                    int(match.group(4)),
                )
                errors[key] = classify_error(text)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["all", "ablation", "topology", "crossdomain"],
        default="all",
    )
    parser.add_argument(
        "--status",
        choices=["all", "completed", "missing"],
        default="all",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    job_root = project_root / "hpc_jobs" / "reviewer"
    log_errors = load_log_errors(project_root)
    stages = (
        ["ablation", "topology", "crossdomain"]
        if args.stage == "all"
        else [args.stage]
    )

    rows: list[dict[str, object]] = []
    for stage in stages:
        for job_path in sorted((job_root / stage).glob("*.sub")):
            parts = job_path.stem.split("__")
            if len(parts) != 4:
                continue
            dataset_or_scenario = parts[1]
            variant_or_model = parts[2]
            seed = int(re.search(r"seed_(\d+)", parts[3]).group(1))
            result = expected_result(
                project_root,
                stage,
                dataset_or_scenario,
                variant_or_model,
                seed,
            )
            completed = result.exists()
            status = "completed" if completed else "missing"
            error_type = (
                ""
                if completed
                else log_errors.get(
                    (stage, dataset_or_scenario, variant_or_model, seed),
                    "missing_log",
                )
            )
            if args.status != "all" and args.status != status:
                continue
            rows.append(
                {
                    "stage": stage,
                    "dataset_or_scenario": dataset_or_scenario,
                    "variant_or_model": variant_or_model,
                    "seed": seed,
                    "status": status,
                    "error_type": error_type,
                    "result_path": str(result),
                }
            )

    if not rows:
        raise SystemExit("No jobs matched the selected filters.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(str(row["status"]) for row in rows)
    print(f"Saved progress report to {args.output.resolve()}")
    print(
        f"selected={len(rows)} completed={counts.get('completed', 0)} "
        f"missing={counts.get('missing', 0)}"
    )
    for stage in stages:
        stage_rows = [row for row in rows if row["stage"] == stage]
        stage_counts = Counter(str(row["status"]) for row in stage_rows)
        print(
            f"{stage:<11} total={len(stage_rows):<3} "
            f"completed={stage_counts.get('completed', 0):<3} "
            f"missing={stage_counts.get('missing', 0):<3}"
        )

    print("\nCompleted jobs:")
    for row in rows:
        if row["status"] == "completed":
            print(
                f"{row['stage']:<11} {row['dataset_or_scenario']:<18} "
                f"{row['variant_or_model']:<35} seed={row['seed']}"
            )

    print("\nMissing jobs:")
    for row in rows:
        if row["status"] == "missing":
            print(
                f"{row['stage']:<11} {row['dataset_or_scenario']:<18} "
                f"{row['variant_or_model']:<35} seed={row['seed']} "
                f"error={row['error_type']}"
            )


if __name__ == "__main__":
    main()
