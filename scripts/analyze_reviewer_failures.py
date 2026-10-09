"""Summarize reviewer-response jobs and classify their failure causes."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


LOG_PATTERN = re.compile(
    r"reviewer_(ablation|topology|crossdomain)/([^/]+)/([^/]+)/seed_(\d+)\.yaml"
)


def _classify(text: str) -> str:
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
    return "incomplete_without_exception"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Previous reviewer_failure_analysis CSV used for comparison.",
    )
    parser.add_argument(
        "--newly-completed-output",
        type=Path,
        default=None,
        help="CSV path for jobs that changed from failed to completed.",
    )
    args = parser.parse_args()

    project = args.project_root.resolve()
    job_root = project / "hpc_jobs" / "reviewer"
    run_root = project / "runs"

    log_errors: dict[tuple[str, str, str, int], str] = {}
    for log_path in (job_root / "ablation" / "logs").glob("*.err"):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        for match in LOG_PATTERN.finditer(text):
            log_errors[
                (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                    int(match.group(4)),
                )
            ] = _classify(text)
    for stage in ("topology", "crossdomain"):
        for log_path in (job_root / stage / "logs").glob("*.err"):
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            for match in LOG_PATTERN.finditer(text):
                log_errors[
                    (
                        match.group(1),
                        match.group(2),
                        match.group(3),
                        int(match.group(4)),
                    )
                ] = _classify(text)

    rows: list[dict[str, object]] = []
    for stage in ("ablation", "topology", "crossdomain"):
        for job_path in sorted((job_root / stage).glob("*.sub")):
            parts = job_path.stem.split("__")
            seed = int(re.search(r"seed_(\d+)", job_path.stem).group(1))
            dataset_or_scenario = parts[1]
            variant_or_model = parts[2]
            if stage in {"ablation", "topology"}:
                result = (
                    run_root
                    / f"reviewer_{stage}"
                    / dataset_or_scenario
                    / variant_or_model
                    / f"seed_{seed}"
                    / "test_metrics.json"
                )
            else:
                result = (
                    run_root
                    / "reviewer_crossdomain"
                    / dataset_or_scenario
                    / variant_or_model
                    / f"seed_{seed}"
                    / "cross_domain_summary.json"
                )
            status = "completed" if result.exists() else "failed"
            error_type = (
                ""
                if status == "completed"
                else log_errors.get(
                    (stage, dataset_or_scenario, variant_or_model, seed),
                    "missing_log",
                )
            )
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.baseline is not None:
        with args.baseline.open("r", encoding="utf-8-sig", newline="") as handle:
            baseline_rows = list(csv.DictReader(handle))
        key_fields = (
            "stage",
            "dataset_or_scenario",
            "variant_or_model",
            "seed",
        )
        baseline_status = {
            tuple(str(row[field]) for field in key_fields): row["status"]
            for row in baseline_rows
        }
        current_status = {
            tuple(str(row[field]) for field in key_fields): row["status"]
            for row in rows
        }
        newly_completed = [
            row
            for row in rows
            if row["status"] == "completed"
            and baseline_status.get(
                tuple(str(row[field]) for field in key_fields)
            )
            != "completed"
        ]
        newly_failed = [
            row
            for row in rows
            if row["status"] != "completed"
            and baseline_status.get(
                tuple(str(row[field]) for field in key_fields)
            )
            == "completed"
        ]
        print(f"\nnewly completed={len(newly_completed)}")
        for row in newly_completed:
            print(
                f"NEW_OK   {row['stage']:<11} "
                f"{row['dataset_or_scenario']:<18} "
                f"{row['variant_or_model']:<35} "
                f"seed={row['seed']}"
            )
        print(f"newly failed={len(newly_failed)}")
        for row in newly_failed:
            print(
                f"NEW_FAIL {row['stage']:<11} "
                f"{row['dataset_or_scenario']:<18} "
                f"{row['variant_or_model']:<35} "
                f"seed={row['seed']} error={row['error_type']}"
            )
        if args.newly_completed_output is not None:
            args.newly_completed_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            with args.newly_completed_output.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(rows[0]),
                )
                writer.writeheader()
                writer.writerows(newly_completed)
            print(
                "Saved newly completed jobs to "
                f"{args.newly_completed_output.resolve()}"
            )

    print(f"Saved failure analysis to {args.output.resolve()}")
    print(f"total={len(rows)} completed={sum(row['status'] == 'completed' for row in rows)}")
    failures = [row for row in rows if row["status"] != "completed"]
    if failures:
        print("\nAbnormal jobs:")
        for row in failures:
            print(
                f"{row['stage']:<11} "
                f"{row['dataset_or_scenario']:<18} "
                f"{row['variant_or_model']:<35} "
                f"seed={row['seed']} "
                f"error={row['error_type']}"
            )
    for stage in ("ablation", "topology", "crossdomain"):
        counts = Counter(
            str(row["error_type"])
            for row in rows
            if row["stage"] == stage and row["status"] != "completed"
        )
        print(stage, dict(counts))


if __name__ == "__main__":
    main()
