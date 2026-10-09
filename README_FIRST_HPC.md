# WallTopo-UQ HPC Upload Bundle

This bundle contains:

- Linux-compatible WallTopo-UQ code;
- relative-path manifests and the four prepared public datasets;
- JHINNO/JSUB job templates for the CDUT "Lizhiyun" platform;
- no torch weight cache; DeepLabV3+ runs without ImageNet initialization;
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

## Reviewer-response experiments

This package adds the reviewer-response experiment generator and job set.

Generate the job files:

```bash
python scripts/generate_reviewer_hpc_jobs.py
```

Submit the stages separately:

```bash
bash hpc_jobs/reviewer/ablation/submit_ablation.sh
bash hpc_jobs/reviewer/topology/submit_topology.sh
bash hpc_jobs/reviewer/crossdomain/submit_crossdomain.sh
bash hpc_jobs/reviewer/uncertainty/submit_uncertainty.sh
```

Collect the completed results:

```bash
python scripts/collect_reviewer_results.py
```

See `REVIEWER_EXPERIMENTS_ZH.md` for the full Chinese workflow.
