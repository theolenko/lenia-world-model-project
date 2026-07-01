#!/bin/bash
#SBATCH --job-name=lenia_eval_interventions
#SBATCH --output=logs_lenia_eval_interventions_%j.out
#SBATCH --error=logs_lenia_eval_interventions_%j.err
#SBATCH --partition=clara
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

WORK_DIR="/lscratch/$SLURM_JOB_ID"
PROJECT_DIR="$HOME/lenia-world-model"
RESULTS_DIR="$HOME/lenia-results/eval_interventions_$SLURM_JOB_ID"
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

echo "=== STEP 5: Evaluation (one-step / multi-step / OOD) ==="
mkdir -p "$WORK_DIR/evaluation/results"
$PYTHON evaluation/evaluate_all.py \
    --data data/v2_lenia_val_chunked.h5 \
    --n-traj 5 \
    --rollout-steps 30 \
    --max-samples 2000
EVAL_EXIT=$?
[ $EVAL_EXIT -ne 0 ] && echo "ERROR: Evaluation failed (exit $EVAL_EXIT)" && exit $EVAL_EXIT

echo "=== STEP 6: Intervention analysis ==="
mkdir -p "$WORK_DIR/experiments/intervention_results"
$PYTHON interventions/run_all_interventions.py \
    --data data/v2_lenia_val_chunked.h5 \
    --n-traj 5 \
    --t-star 10 \
    --n-steps 30
INTV_EXIT=$?
[ $INTV_EXIT -ne 0 ] && echo "ERROR: Interventions failed (exit $INTV_EXIT)" && exit $INTV_EXIT

echo "=== STEP 7: Save results ==="
mkdir -p "$RESULTS_DIR"
cp -r "$WORK_DIR/evaluation/results/"          "$RESULTS_DIR/evaluation/"         2>/dev/null || true
cp -r "$WORK_DIR/experiments/intervention_results/" "$RESULTS_DIR/interventions/" 2>/dev/null || true
echo "Results saved to $RESULTS_DIR"

echo "=== JOB DONE. ==="
