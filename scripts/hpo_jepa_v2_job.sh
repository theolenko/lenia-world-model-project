#!/bin/bash
#SBATCH --job-name=lenia_hpo_jepa_v2
#SBATCH --output=logs_lenia_hpo_jepa_v2_%j.out
#SBATCH --error=logs_lenia_hpo_jepa_v2_%j.err
#SBATCH --partition=clara
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

WORK_DIR="/lscratch/$SLURM_JOB_ID"
PROJECT_DIR="$HOME/lenia-world-model"
RESULTS_DIR="$HOME/lenia-results/hpo_jepa_v2_$SLURM_JOB_ID"
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

echo "=== STEP 4: Verify CUDA ==="
$PYTHON -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')" || exit 1

echo "=== STEP 5: Run JEPA-CNN HPO ==="
export WANDB_MODE=offline
mkdir -p "$WORK_DIR/experiments"

$PYTHON scripts/run_hpo.py \
    --setup jepa \
    --config config_cluster_jepa_v2.yaml \
    --storage "sqlite:///$PROJECT_DIR/experiments/hpo_jepa_v2.db" \
    --study-name jepa_v2_hpo
EXIT_CODE=$?

echo "=== STEP 6: Save results ==="
mkdir -p "$RESULTS_DIR"
cp -r "$WORK_DIR/experiments/" "$RESULTS_DIR/" 2>/dev/null || true
echo "Results saved to $RESULTS_DIR"

[ $EXIT_CODE -ne 0 ] && echo "ERROR: HPO failed (exit $EXIT_CODE)" && exit $EXIT_CODE
echo "=== JOB DONE. ==="
