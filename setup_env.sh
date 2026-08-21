#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="bio_diffusion"
PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda was not found on PATH. Install Miniconda or Anaconda first." >&2
    exit 1
fi

# Make conda activate available in non-interactive shells.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Using existing Conda environment: ${ENV_NAME}"
else
    conda create --yes --name "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected; installing PyTorch with CUDA 12.1 support."
    python -m pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected; installing the CPU PyTorch build."
    python -m pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
fi

python -m pip install --requirement "${PROJECT_ROOT}/requirements.txt"

mkdir -p "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/checkpoints" "${PROJECT_ROOT}/outputs"
echo "Environment '${ENV_NAME}' is ready. Activate it with: conda activate ${ENV_NAME}"
