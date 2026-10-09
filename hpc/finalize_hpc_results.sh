#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/WallTopo-UQ-HPC}"
cd "$PROJECT_ROOT"

WALLTOPO_ENV_BIN="${WALLTOPO_ENV_BIN:-$HOME/miniconda3/bin}"
export PATH="$WALLTOPO_ENV_BIN:$PATH"
unset PYTHONPATH || true
export KMP_DUPLICATE_LIB_OK=TRUE
export TORCH_HOME="$HOME/.cache/torch"

python scripts/collect_benchmark_results.py \
  --root runs/benchmark \
  --output results/benchmark \
  --proposed "${WALLTOPO_PROPOSED:-walltopo_uq}"

python scripts/benchmark_models.py \
  --config configs/dais.yaml \
  --models walltopo_uq unet deeplabv3plus \
  --size 224 \
  --batch-size 2 \
  --iterations 10 \
  --warmup 3 \
  --device cuda \
  --output results/benchmark/efficiency.csv

python scripts/make_publication_figures.py \
  --results results/benchmark \
  --output figures/publication

python scripts/make_model_comparison_gallery.py \
  --run-prefix benchmark \
  --datasets dais deepcrack crack500 crackforest \
  --models unet deeplabv3plus "${WALLTOPO_PROPOSED:-walltopo_uq}" \
  --seed 3407 \
  --samples 3 \
  --device cuda \
  --output-dir figures/qualitative
