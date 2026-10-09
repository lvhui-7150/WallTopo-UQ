#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/WallTopo-UQ-HPC}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-walltopo}"

cd "$PROJECT_ROOT"

if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  echo "Miniconda not found at $CONDA_ROOT."
  echo "Install Miniconda3-py39_23.5.2-0 first."
  exit 1
fi

export PATH="$CONDA_ROOT/bin:$PATH"
unset PYTHONPATH || true

if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  conda create -y -n "$ENV_NAME" python=3.9
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements-hpc.txt

mkdir -p "$HOME/.cache/torch/hub/checkpoints"
if [ -f "$PROJECT_ROOT/torch_cache/hub/checkpoints/resnet50-11ad3fa6.pth" ]; then
  cp "$PROJECT_ROOT/torch_cache/hub/checkpoints/resnet50-11ad3fa6.pth" \
    "$HOME/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
fi

python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
echo "Environment ready. Activate it with:"
echo "  source $CONDA_ROOT/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
