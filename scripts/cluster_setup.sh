#!/bin/bash
# One-time setup of the conda environment on the cluster.
# Run on the cluster: bash ~/lenia-world-model/scripts/cluster_setup.sh

set -e

PYTHON="$HOME/.conda/envs/lenia-wm/bin/python"
PIP="$HOME/.conda/envs/lenia-wm/bin/pip"

echo "=== STEP 1: Load Anaconda module ==="
module load Anaconda3

echo "=== STEP 2: Create conda environment (idempotent) ==="
if conda info --envs | grep -q "lenia-wm"; then
    echo "Environment 'lenia-wm' already exists — skipping."
else
    conda create -n lenia-wm python=3.10 -y
fi

echo "=== STEP 3: Install dependencies ==="
$PIP install -r ~/lenia-world-model/requirements.txt

echo "=== STEP 4: Create data directory ==="
mkdir -p ~/lenia-world-model/data/

echo "=== STEP 5: Verify installation ==="
$PYTHON -c "import torch; print(f'PyTorch:  {torch.__version__}'); print(f'CUDA:     {torch.cuda.is_available()}')"
$PYTHON -c "import h5py; print(f'h5py:     {h5py.__version__}')"

echo ""
echo "=== Setup complete. ==="
