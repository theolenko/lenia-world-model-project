# lenia-world-model-project

code related to seminar-project in "Mathematics and AI Internship" module of Data Science course at Uni Leipzig

## Project Structure

```text
lenia-world-model-project/
├── data/                        # Lenia trajectory files (gitignored)
│   ├── lenia_train_chunked.h5   # Training data — per-frame chunks (1,1,64,64) ← use this
│   ├── lenia_val_chunked.h5     # Validation data — per-frame chunks (1,1,64,64) ← use this
│   ├── lenia_train.h5           # Original (per-trajectory chunks, slow random access)
│   └── lenia_val.h5             # Original (per-trajectory chunks)
├── models/                      # Neural network architectures
│   ├── encoder.py               # CNN/Transformer backbone
│   ├── decoder.py               # Pixel-prediction decoder (Setup A)
│   ├── predictor.py             # JEPA latent predictor (Setup B)
│   └── world_model.py           # Integrated model logic
├── training/                    # Training logic and loops
│   ├── trainer.py               # Training orchestrator
│   └── losses.py                # Loss functions (MSE, VICReg, etc.)
├── scripts/                     # Executable scripts
│   ├── generate_data.py         # Lenia simulation generator
│   ├── train_single.py          # Single training execution + LeniaDataset
│   ├── run_hpo.py               # Optuna hyperparameter optimization
│   ├── rechunk_data.py          # Convert HDF5 from per-trajectory to per-frame chunks
│   ├── benchmark_loading.py     # Measure DataLoader throughput before/after rechunking
│   ├── smoke_test.py            # End-to-end pipeline smoke test (Steps 2–6)
│   └── check_data.py            # Data validation and visualisation (Step 1)
├── experiments/                 # Checkpoints, logs, and plots
│   ├── pipeline_smoke_test.md   # Smoke test report (bugs found, quantitative results)
│   └── rechunk_report.md        # Rechunking benchmark report (47× speedup)
├── utils/                       # Helper functions (visualization, physics)
├── config.yaml                  # Global configuration parameters
└── requirements.txt             # Python dependencies
```

## 1. Branching Strategy

- **main**: Contains the stable, production-ready code.
- **develop**: The integration branch for features. All pull requests should target this branch.

## 2. Branch Protection Rules

To prevent accidental pushes to protected branches, the following rules are applied to both `main` and `develop`:

- **Require a pull request before merging**: No direct pushes allowed.

---

## Synchronization & Conflict Management

To maintain a clean repository and avoid breaking the World Model architecture, always check for remote changes and potential conflicts before starting your work.

### 1. Fetch Remote Changes

Update your local knowledge of the remote repository without affecting your files:

```bash
git fetch origin

```

### 2. Check for Conflicts

See if your local branch has fallen behind the remote branch:

```bash
git status

```

If the output says _"Your branch is behind 'origin/develop' by X commits"_, you must synchronize before proceeding to avoid merge conflicts later.

### 3. Safe Pull & Resolution

Merge the remote changes into your local workspace:

```bash
git pull origin develop

```

_Note: If Git reports a merge conflict, stop and resolve the marked sections in the affected files manually. Once resolved, use `git add` and `git commit` to finalize the merge._

---

## Development Workflow

Please follow this standard workflow for all contributions.

### 1. Sync Local Environment

Ensure your local `develop` branch is up to date:

```bash
git checkout develop && git pull origin develop

```

### 2. Create a Feature Branch

Create a new branch for your specific task:

```bash
git checkout -b feature/your-feature-name

```

### 3. Development & Commits

Make your changes and commit them with descriptive messages:

```bash
git add .
git commit -m "Brief description of what you did"

```

### 4. Final Conflict Check

Before pushing, it is good practice to pull from `develop` one last time to ensure no new conflicts were introduced while you were coding:

```bash
git pull origin develop

```

### 5. Push to GitHub

Push your local branch to the remote repository:

```bash
git push origin feature/your-feature-name

```

### 6. Open a Pull Request (PR)

1. Go to the repository on GitHub.
2. Click the "Compare & pull request" button.
3. Ensure the target branch is set to `develop`.
4. Add a short description of your changes and submit the PR for review.

---

## Running the Training Scripts

### Setup

Activate the virtual environment before running any script:

```bash
source lenia-wm/bin/activate
```

All scripts are run from the **repository root**.

---

### Data

#### HDF5 files (primary format)

Place the raw `.h5` files in `data/` and rechunk them before training (see below).  The dataset class reads the array stored under the key `"frames"`.

```
data/
├── lenia_train.h5           ← raw (slow random access, keep as backup)
├── lenia_val.h5             ← raw
├── lenia_train_chunked.h5   ← rechunked, use for training
└── lenia_val_chunked.h5     ← rechunked, use for training
```

#### Why rechunk?

The raw files use chunk layout `(1, 200, 64, 64)` — one 3 MB chunk per trajectory.  Shuffled `DataLoader` access decompresses the full chunk for every frame, giving ~1 200 ms/batch.

After rechunking to `(1, 1, 64, 64)` (one 16 KB chunk per frame), only the requested frame is decompressed: **~25 ms/batch — 47× faster**.  See `experiments/rechunk_report.md` for benchmark details.

#### Rechunk once after receiving the raw files

```bash
python scripts/rechunk_data.py --input data/lenia_train.h5 --output data/lenia_train_chunked.h5
python scripts/rechunk_data.py --input data/lenia_val.h5   --output data/lenia_val_chunked.h5
```

Both files are verified automatically (5 random frames compared bit-for-bit against the original).

#### Expected array shape

| Shape | Meaning |
|---|---|
| `(N, T, H, W)` | N trajectories, T timesteps, spatial H×W — **most common** |
| `(N, T, 1, H, W)` | Same with explicit channel dim |
| `(T, H, W)` | Single trajectory |

- `H = W = 64` — spatial resolution required by the encoder
- Values in any positive range are fine; the loader normalises by the global max
- With N=1 800 trajectories of T=200 frames the training set has 1 800×199 = 358 200 frame pairs

#### No real data yet? Use `--dummy`

Both training scripts accept `--dummy`, which generates random in-memory tensors and skips disk I/O entirely:

```bash
python scripts/train_single.py --setup pixel --dummy
python scripts/train_single.py --setup jepa  --dummy
python scripts/run_hpo.py      --setup pixel --dummy --n-trials 3 --hpo-epochs 2
```

---

### Full training workflow (step by step)

> The same workflow applies to both `pixel` and `jepa` setups.  Run it once for each.

#### Step 0 — Prepare the data (once)

Rechunk the raw HDF5 files and validate the pipeline on real data:

```bash
# 1. Rechunk to per-frame layout (run once after receiving raw files)
python scripts/rechunk_data.py --input data/lenia_train.h5 --output data/lenia_train_chunked.h5
python scripts/rechunk_data.py --input data/lenia_val.h5   --output data/lenia_val_chunked.h5

# 2. Validate data integrity and inspect first trajectories
python scripts/check_data.py

# 3. Run the full pipeline smoke test (DataLoader → forward pass → 7 epochs → rollout)
python scripts/smoke_test.py
```

Results land in `experiments/smoke_test_<timestamp>/`.  See `experiments/pipeline_smoke_test.md` for expected output values.

#### Step 1 — Quick environment check (optional, no data needed)

```bash
python scripts/train_single.py --setup pixel --dummy
```

If this completes without errors, the environment is working.

#### Step 2 — Find good hyperparameters with HPO

`run_hpo.py` runs many short trials (each `hpo.hpo_epochs` epochs long, default 20) and finds the parameter combination with the lowest validation loss.  It does **not** produce a final trained model — it only identifies good settings.

```bash
# 50 trials, ephemeral (results only in memory)
python scripts/run_hpo.py --setup pixel
python scripts/run_hpo.py --setup jepa

# Recommended: persistent study — survives interruptions, can be resumed
python scripts/run_hpo.py --setup pixel \
    --storage sqlite:///experiments/hpo.db \
    --study-name pixel_hpo

# Resume an interrupted study (same --storage and --study-name)
python scripts/run_hpo.py --setup pixel \
    --storage sqlite:///experiments/hpo.db \
    --study-name pixel_hpo --n-trials 20   # adds 20 more trials
```

When finished, the best parameters are printed to the terminal and saved to:

```
experiments/hpo_pixel_best_params.yaml
experiments/hpo_jepa_best_params.yaml
```

Example output file:

```yaml
best_val_loss: 0.002341
params:
  batch_size: 64
  embed_dim: 256
  learning_rate: 0.0028
  weight_decay: 2.1e-06
  # JEPA only:
  ema_momentum: 0.97
  predictor_hidden_dim: 512
  use_variance_reg: false
  use_covariance_reg: false
```

#### Step 3 — Copy the best params into `config.yaml`

Open `config.yaml` and update the values that appear in the HPO result.  The mapping is one-to-one:

| Key in `hpo_*_best_params.yaml` | Where it goes in `config.yaml` |
|---|---|
| `embed_dim` | `model.embed_dim` |
| `learning_rate` | `training.learning_rate` |
| `weight_decay` | `training.weight_decay` |
| `batch_size` | `data.batch_size` |
| `ema_momentum` | `model.ema_momentum` *(JEPA only)* |
| `predictor_hidden_dim` | `model.predictor_hidden_dim` *(JEPA only)* |
| `use_variance_reg` | `jepa.use_variance_reg` *(JEPA only)* |
| `var_reg_weight` | `jepa.var_reg_weight` *(JEPA only, if reg enabled)* |
| `use_covariance_reg` | `jepa.use_covariance_reg` *(JEPA only)* |
| `cov_reg_weight` | `jepa.cov_reg_weight` *(JEPA only, if reg enabled)* |

Keys not listed (e.g. `num_epochs`) are not touched by HPO — keep them as they are.

After editing, `config.yaml` might look like this for a pixel run:

```yaml
data:
  batch_size: 64          # ← from HPO

model:
  embed_dim: 256          # ← from HPO

training:
  num_epochs: 100         # ← unchanged, set this as high as you want
  learning_rate: 0.0028   # ← from HPO
  weight_decay: 2.1e-06   # ← from HPO
```

#### Step 4 — Run the full training

```bash
python scripts/train_single.py --setup pixel
python scripts/train_single.py --setup jepa
```

Checkpoints and TensorBoard logs are written to `experiments/{setup}_{timestamp}/`.

```bash
# Monitor training live
tensorboard --logdir experiments/
```

Both setups use the same random seed (42), so the encoder is initialised identically — this is required for a fair comparison between the two paradigms.

---

## Cluster Training (Clara, Uni Leipzig)

### Prerequisites

- SC account with access to the Clara cluster
- SSH key registered at [portal.sc.uni-leipzig.de](https://portal.sc.uni-leipzig.de)
- VPN active (openconnect or Cisco AnyConnect)

### Step by step

**One-time setup:**

```bash
# 1. Transfer code to the cluster
bash scripts/sync_to_cluster.sh

# 2. Log in
ssh jv72unec@login01.sc.uni-leipzig.de

# 3. Create the conda environment and install dependencies (on the cluster)
bash ~/lenia-world-model/scripts/cluster_setup.sh

# 4. Transfer data (several GB — only needed once)
bash scripts/sync_data_to_cluster.sh
```

**Per training run:**

```bash
# Sync code (run locally)
bash scripts/sync_to_cluster.sh

# Log in and submit job
ssh jv72unec@login01.sc.uni-leipzig.de
sbatch ~/lenia-world-model/scripts/train_job.sh

# Check status
squeue -u jv72unec

# Follow logs live
tail -f logs_lenia_JOBID.out

# Download results (run locally)
rsync -avz jv72unec@login01.sc.uni-leipzig.de:~/lenia-results/ ./experiments/cluster/
```

**Test without transferring anything:**

```bash
bash scripts/sync_to_cluster.sh --dry-run
```

### Cluster rules (summary)

| Rule | Details |
|---|---|
| No direct execution | Submit all jobs via `sbatch` — never run them on login nodes |
| No sudo | Install packages via `conda` or `pip` only |
| No remote editors | No VS Code Remote, no Cursor, etc. |
| Use lscratch | Write temporary data to `/lscratch/$SLURM_JOB_ID` — deleted after the job ends |
| Save results | The job script automatically copies results to `~/lenia-results/` |

### Troubleshooting

**GPU not available:**
```
AssertionError: CUDA nicht verfuegbar!
```
→ Check that `--gres=gpu:1` is set in the job script. Use `squeue -u jv72unec` to confirm the job is running on the `clara` partition.

**Out of memory (OOM):**
```
RuntimeError: CUDA out of memory
```
→ Reduce `batch_size` in `config_cluster.yaml` (e.g. to 32), or increase `--mem=64G` in the SBATCH header.

**Job stuck in queue:**
```
squeue -u jv72unec  →  job stays in PD (pending)
```
→ May be caused by resource contention. Check node availability with `sinfo -p clara`. Try reducing `--time`.

**Transfer interrupted:**
`sync_data_to_cluster.sh` uses `--partial` — just run it again and rsync will resume where it left off.

---

### HPO options reference

| Flag | Default | Effect |
|---|---|---|
| `--n-trials N` | `hpo.n_trials` (50) | Number of Optuna trials to run |
| `--hpo-epochs N` | `hpo.hpo_epochs` (20) | Training epochs per trial |
| `--study-name NAME` | `lenia_{setup}_hpo` | Name of the Optuna study |
| `--storage URL` | `hpo.storage` (none) | Backend for persistence, e.g. `sqlite:///experiments/hpo.db` |
| `--n-jobs N` | `1` | Parallel trials; requires shared storage and `num_workers: 0` in config |
| `--dummy` | off | Use random in-memory data instead of loading from disk |

CLI flags always override the corresponding values in `config.yaml`.

---

### `config.yaml` reference

All paths and hyperparameters live in **`config.yaml`** at the repository root.

| Section | Key | What it controls |
|---|---|---|
| `data` | `train_path` / `val_path` | Paths to rechunked `.h5` files (default: `data/lenia_train_chunked.h5`, `data/lenia_val_chunked.h5`) |
| `data` | `batch_size` | Mini-batch size |
| `data` | `num_workers` | DataLoader worker processes (use `0` with HPO `--n-jobs > 1`) |
| `model` | `embed_dim` | Encoder output dimensionality |
| `model` | `ema_momentum` | EMA decay for the JEPA target encoder |
| `model` | `predictor_hidden_dim` | Hidden width of the JEPA MLP predictor |
| `training` | `num_epochs` | Total epochs for a full training run |
| `training` | `learning_rate` | Adam learning rate |
| `training` | `weight_decay` | Adam weight decay |
| `training` | `checkpoint_interval` | Save a checkpoint every N epochs |
| `jepa` | `use_variance_reg` | Enable VICReg variance anti-collapse term |
| `jepa` | `use_covariance_reg` | Enable VICReg covariance anti-redundancy term |
| `jepa` | `var_reg_weight` / `cov_reg_weight` | Loss weights for the regularisers above |
| `hpo` | `n_trials` | Default number of Optuna trials |
| `hpo` | `hpo_epochs` | Epochs per HPO trial |
| `hpo` | `storage` | Optuna storage URL |
| `hpo` | `search_space` | Bounds / choices for each tuneable hyperparameter |
