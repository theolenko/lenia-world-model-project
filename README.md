# lenia-world-model-project

Code related to seminar project in "Mathematics and AI Internship" module of Data Science course at Uni Leipzig.

Comparison of two world model architectures on Lenia cellular automata:
- **Setup A (Pixel)**: CNN encoder → decoder, predicts next frame in pixel space
- **Setup B (JEPA)**: CNN encoder → MLP predictor, predicts next frame's embedding (latent space)

## Project Structure

```text
lenia-world-model-project/
├── data/                              # Lenia trajectory files (gitignored)
│   ├── lenia_train_chunked.h5         # Training data — per-frame chunks (use this)
│   ├── lenia_val_chunked.h5           # Validation data — per-frame chunks (use this)
│   ├── lenia_train.h5                 # Original (per-trajectory chunks, slow random access)
│   └── lenia_val.h5                   # Original
├── models/                            # Neural network architectures
│   ├── encoder.py                     # CNN/Transformer backbone
│   ├── decoder.py                     # Pixel-prediction decoder (Setup A)
│   ├── predictor.py                   # JEPA latent predictor (Setup B)
│   └── world_model.py                 # PixelWorldModel and JEPAWorldModel
├── training/                          # Training logic
│   ├── trainer.py                     # Training loop, validation, checkpointing
│   └── losses.py                      # MSE loss, VICReg variance/covariance terms
├── scripts/                           # Executable scripts
│   ├── generate_data.py               # Lenia simulation generator
│   ├── train_single.py                # Training entry point + LeniaDataset
│   ├── run_hpo.py                     # Optuna hyperparameter optimisation
│   ├── rechunk_data.py                # Convert HDF5 from per-trajectory to per-frame chunks
│   ├── benchmark_loading.py           # Measure DataLoader throughput
│   ├── smoke_test.py                  # End-to-end pipeline smoke test
│   ├── check_data.py                  # Data validation and visualisation
│   ├── visualize_results.py           # Plot loss curves + pixel predictions from saved checkpoints
│   ├── sync_to_cluster.sh             # Sync code to Clara cluster via rsync
│   ├── sync_data_to_cluster.sh        # Transfer HDF5 data to cluster (resumable)
│   ├── cluster_setup.sh               # One-time conda env setup on cluster
│   ├── hpo_pixel_job.sh               # SLURM job: pixel HPO
│   ├── hpo_jepa_job.sh                # SLURM job: JEPA HPO
│   ├── train_pixel_job.sh             # SLURM job: pixel full training
│   └── train_jepa_job.sh              # SLURM job: JEPA full training
├── experiments/                       # Checkpoints, logs, and results
│   ├── cluster/                       # Results downloaded from Clara
│   │   ├── pixel_<jobid>/             # Pixel training checkpoint + TensorBoard logs
│   │   ├── jepa_<jobid>/              # JEPA training checkpoint + TensorBoard logs
│   │   ├── hpo_pixel_<jobid>/         # Pixel HPO results + best params YAML
│   │   └── hpo_jepa_<jobid>/          # JEPA HPO results + best params YAML
│   └── exploration.ipynb              # Analysis notebook
├── checkpoints/                       # Trained model weights (tracked via Git LFS)
│   ├── pixel_v1.pt                    # Pixel world model — 100 epochs, val loss 0.025
│   └── jepa_v1.pt                     # JEPA world model — 100 epochs (see notes in config)
├── config.yaml                        # Local development config
├── config_cluster_pixel.yaml          # Cluster config — pixel training (HPO best params)
├── config_cluster_jepa.yaml           # Cluster config — JEPA training (HPO best params)
└── requirements.txt                   # Python dependencies
```

## Branching Strategy

- **main**: Stable, production-ready code.
- **develop**: Integration branch — all pull requests target this branch.

---

## Synchronization & Conflict Management

Always fetch and sync before starting work to avoid merge conflicts.

```bash
git fetch origin
git status
git pull origin develop
```

---

## Development Workflow

```bash
# 1. Sync local develop branch
git checkout develop && git pull origin develop

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Commit changes
git add .
git commit -m "Brief description"

# 4. Pull one last time before pushing
git pull origin develop

# 5. Push and open PR targeting develop
git push origin feature/your-feature-name
```

---

## Running the Training Scripts

### Setup

```bash
source lenia-wm/bin/activate
```

All scripts are run from the **repository root**.

---

### Data

#### HDF5 files (primary format)

Place raw `.h5` files in `data/` and rechunk them before training. The dataset class reads the array stored under the key `"frames"`.

```
data/
├── lenia_train.h5           ← raw (slow random access, keep as backup)
├── lenia_val.h5             ← raw
├── lenia_train_chunked.h5   ← rechunked, use for training
└── lenia_val_chunked.h5     ← rechunked, use for training
```

#### Why rechunk?

The raw files use chunk layout `(1, 200, 64, 64)` — one 3 MB chunk per trajectory. Shuffled DataLoader access decompresses the full chunk for every frame, giving ~1200 ms/batch.

After rechunking to `(1, 1, 64, 64)` (one 16 KB chunk per frame), only the requested frame is decompressed: **~25 ms/batch — 47× faster**.

```bash
python scripts/rechunk_data.py --input data/lenia_train.h5 --output data/lenia_train_chunked.h5
python scripts/rechunk_data.py --input data/lenia_val.h5   --output data/lenia_val_chunked.h5
```

#### Dataset size

- Train: 1800 trajectories × 200 frames = 360,000 frames → 358,200 consecutive pairs
- Val: 200 trajectories × 200 frames = 40,000 frames
- Spatial resolution: 64×64, single channel

#### No real data? Use `--dummy`

```bash
python scripts/train_single.py --setup pixel --dummy
python scripts/train_single.py --setup jepa  --dummy
python scripts/run_hpo.py      --setup pixel --dummy --n-trials 3 --hpo-epochs 2
```

---

### Full training workflow

#### Step 0 — Prepare the data (once)

```bash
python scripts/rechunk_data.py --input data/lenia_train.h5 --output data/lenia_train_chunked.h5
python scripts/rechunk_data.py --input data/lenia_val.h5   --output data/lenia_val_chunked.h5
python scripts/check_data.py
```

#### Step 1 — Quick environment check

```bash
python scripts/train_single.py --setup pixel --dummy
```

#### Step 2 — Hyperparameter optimisation (optional)

`run_hpo.py` runs many short trials and finds the best hyperparameters. Best params are saved to `experiments/hpo_{setup}_best_params.yaml`.

```bash
python scripts/run_hpo.py --setup pixel \
    --storage sqlite:///experiments/hpo_pixel.db \
    --study-name pixel_hpo

python scripts/run_hpo.py --setup jepa \
    --storage sqlite:///experiments/hpo_jepa.db \
    --study-name jepa_hpo
```

#### Step 3 — Update config with best params

Copy the values from `hpo_{setup}_best_params.yaml` into `config.yaml` (local) or the relevant cluster config.

#### Step 4 — Full training

```bash
python scripts/train_single.py --setup pixel
python scripts/train_single.py --setup jepa

# Monitor
tensorboard --logdir experiments/
```

Training uses early stopping (default patience: 15 epochs) and saves the best checkpoint as `checkpoint_best.pt`.

---

## Cluster Training (Clara, Uni Leipzig)

### Prerequisites

- SC account with access to the Clara cluster
- SSH key registered at [portal.sc.uni-leipzig.de](https://portal.sc.uni-leipzig.de)
- VPN active (openconnect or Cisco AnyConnect)

### One-time setup

```bash
# Transfer code to cluster
bash scripts/sync_to_cluster.sh

# Log in
ssh <unilogin>@login01.sc.uni-leipzig.de

# Create conda environment (on cluster)
bash ~/lenia-world-model/scripts/cluster_setup.sh

# Transfer data (several GB — resumable)
bash scripts/sync_data_to_cluster.sh
```

### Per training run

```bash
# Sync latest code (run locally)
bash scripts/sync_to_cluster.sh

# Submit jobs
ssh <unilogin>@login01.sc.uni-leipzig.de
sbatch ~/lenia-world-model/scripts/train_pixel_job.sh
sbatch ~/lenia-world-model/scripts/train_jepa_job.sh

# Check status
squeue -u <unilogin>

# Follow logs
tail -f logs_lenia_pixel_JOBID.out
tail -f logs_lenia_jepa_JOBID.out

# Download results (run locally)
rsync -avz <unilogin>@login01.sc.uni-leipzig.de:~/lenia-results/ ./experiments/cluster/
```

### HPO on cluster

```bash
sbatch ~/lenia-world-model/scripts/hpo_pixel_job.sh
sbatch ~/lenia-world-model/scripts/hpo_jepa_job.sh
```

### Cluster rules

| Rule | Details |
|---|---|
| No direct execution | Submit all jobs via `sbatch` — never run on login nodes |
| No sudo | Install packages via `conda` or `pip` only |
| No remote editors | No VS Code Remote, no Cursor, etc. |
| Use lscratch | Write temporary data to `/lscratch/$SLURM_JOB_ID` — deleted after job ends |
| Save results | Job scripts automatically copy results to `~/lenia-results/` |

### Troubleshooting

**GPU not available:**
→ Check `--gres=gpu:1` is set. Confirm job runs on `clara` partition via `squeue -u <unilogin>`.

**Out of memory:**
→ Reduce `batch_size` in the relevant `config_cluster_*.yaml`, or increase `--mem=64G` in the SBATCH header.

**Job stuck in queue:**
→ Resource contention. Check with `sinfo -p clara`. Try reducing `--time`.

**Transfer interrupted:**
→ `sync_data_to_cluster.sh` uses `--partial` — re-run and rsync resumes automatically.

---

## `config.yaml` reference

| Section | Key | What it controls |
|---|---|---|
| `data` | `train_path` / `val_path` | Paths to rechunked `.h5` files |
| `data` | `batch_size` | Mini-batch size |
| `data` | `num_workers` | DataLoader workers (use `0` with HPO `--n-jobs > 1`) |
| `model` | `embed_dim` | Encoder output dimensionality |
| `model` | `ema_momentum` | EMA decay for JEPA target encoder (per-batch) |
| `model` | `predictor_hidden_dim` | Hidden width of JEPA MLP predictor |
| `training` | `num_epochs` | Maximum training epochs |
| `training` | `early_stopping_patience` | Stop after N epochs without val improvement (`0` = disabled) |
| `training` | `learning_rate` | Adam learning rate |
| `training` | `weight_decay` | Adam weight decay |
| `training` | `checkpoint_interval` | Save checkpoint every N epochs |
| `jepa` | `use_variance_reg` | VICReg variance term — prevents representation collapse |
| `jepa` | `use_covariance_reg` | VICReg covariance term — prevents redundant dimensions |
| `jepa` | `var_reg_weight` / `cov_reg_weight` | Loss weights for the regularisers |
| `hpo` | `n_trials` | Number of Optuna trials |
| `hpo` | `hpo_epochs` | Training epochs per HPO trial |
| `hpo` | `storage` | Optuna backend URL for persistence |
| `hpo` | `search_space` | Bounds / choices for each tuneable hyperparameter |

## HPO options reference

| Flag | Default | Effect |
|---|---|---|
| `--n-trials N` | `hpo.n_trials` (50) | Number of Optuna trials |
| `--hpo-epochs N` | `hpo.hpo_epochs` (20) | Training epochs per trial |
| `--study-name NAME` | `lenia_{setup}_hpo` | Optuna study name |
| `--storage URL` | `hpo.storage` (none) | Backend for persistence |
| `--n-jobs N` | `1` | Parallel trials (requires shared storage + `num_workers: 0`) |
| `--dummy` | off | Use random in-memory data |
