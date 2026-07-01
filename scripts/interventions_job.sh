#!/bin/bash
#SBATCH --job-name=lenia_interventions
#SBATCH --output=logs_lenia_interventions_%j.out
#SBATCH --error=logs_lenia_interventions_%j.err
#SBATCH --partition=clara
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

WORK_DIR="/lscratch/$SLURM_JOB_ID"
PROJECT_DIR="$HOME/lenia-world-model"
RESULTS_DIR="$HOME/lenia-results/interventions_$SLURM_JOB_ID"
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

echo "=== STEP 5: Run intervention analysis ==="
mkdir -p "$WORK_DIR/experiments/intervention_results"

$PYTHON interventions/run_all_interventions.py --n-traj 5 --t-star 10 --n-steps 30
EXIT_CODE=$?

echo "=== STEP 6: Save results ==="
mkdir -p "$RESULTS_DIR"
cp -r "$WORK_DIR/experiments/intervention_results/" "$RESULTS_DIR/" 2>/dev/null || true
echo "Results saved to $RESULTS_DIR"

[ $EXIT_CODE -ne 0 ] && echo "ERROR: Interventions failed (exit $EXIT_CODE)" && exit $EXIT_CODE
echo "=== JOB DONE. ==="
