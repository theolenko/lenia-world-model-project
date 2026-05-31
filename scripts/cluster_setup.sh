#!/bin/bash
# One-time setup of the conda environment on the cluster.
# Run on the cluster: bash ~/lenia-world-model/scripts/cluster_setup.sh

set -e

echo "=== STEP 1: Load Anaconda module ==="
module load Anaconda3

echo "=== STEP 2: Create conda environment (idempotent) ==="
conda create -n lenia-wm python=3.10 -y 2>/dev/null || echo "Environment 'lenia-wm' already exists — skipping."

echo "=== STEP 3: Activate environment ==="
source activate lenia-wm

echo "=== STEP 4: Install dependencies ==="
pip install -r ~/lenia-world-model/requirements.txt

echo "=== STEP 5: Create data directory ==="
mkdir -p ~/lenia-world-model/data/

echo "=== STEP 6: Verify installation ==="
python -c "import torch; print(f'PyTorch:  {torch.__version__}'); print(f'CUDA:     {torch.cuda.is_available()}')"
python -c "import h5py; print(f'h5py:     {h5py.__version__}')"

echo ""
echo "=== Setup complete. ==="
