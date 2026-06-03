#!/bin/bash
#SBATCH --job-name=lenia_hpo_pixel
#SBATCH --output=logs_lenia_hpo_pixel_%j.out
#SBATCH --error=logs_lenia_hpo_pixel_%j.err
#SBATCH --partition=clara
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

WORK_DIR="/lscratch/$SLURM_JOB_ID"
PROJECT_DIR="$HOME/lenia-world-model"
RESULTS_DIR="$HOME/lenia-results/hpo_pixel_$SLURM_JOB_ID"
PYTHON="$HOME/.conda/envs/lenia-wm/bin/python"

echo "=== STEP 1: Load environment ==="
module load Anaconda3

echo "=== STEP 2: Copy code to lscratch ==="
mkdir -p "$WORK_DIR"
cp -r "$PROJECT_DIR"/* "$WORK_DIR/"
cd "$WORK_DIR"

echo "=== STEP 3: Link data directory ==="
rm -rf "$WORK_DIR/data"
ln -s "$PROJECT_DIR/data" "$WORK_DIR/data"

echo "=== STEP 4: Verify CUDA availability ==="
$PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')" || exit 1

echo "=== STEP 5: Set WandB to offline ==="
export WANDB_MODE=offline
mkdir -p "$WORK_DIR/experiments"

echo "=== STEP 6: HPO — pixel setup (20 trials, 10 epochs/trial) ==="
$PYTHON scripts/run_hpo.py \
    --setup pixel \
    --config config_cluster.yaml \
    --storage "sqlite:///experiments/hpo_pixel.db" \
    --study-name pixel_hpo \
    --n-trials 20 \
    --hpo-epochs 10
EXIT_CODE=$?

echo "=== STEP 7: Save results ==="
mkdir -p "$RESULTS_DIR"
cp -r "$WORK_DIR/experiments/" "$RESULTS_DIR/" 2>/dev/null || true
echo "Results saved to $RESULTS_DIR"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Pixel HPO failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "=== JOB DONE. ==="
