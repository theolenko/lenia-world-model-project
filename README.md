# lenia-world-model-project

code related to seminar-project in "Mathematics and AI Internship" module of Data Science course at Uni Leipzig

## Project Structure

```text
lenia-world-models/
├── data/                   # Raw and processed Lenia trajectories (.npy or .h5)
├── models/                 # Neural network architectures
│   ├── encoder.py          # CNN/Transformer backbone
│   ├── decoder.py          # Pixel-prediction decoder (Setup A)
│   ├── predictor.py        # JEPA latent predictor (Setup B)
│   └── world_model.py      # Integrated model logic
├── training/               # Training logic and loops
│   ├── trainer.py          # Training orchestrator
│   └── losses.py           # Loss functions (MSE, VICReg, etc.)
├── scripts/                # Executable scripts
│   ├── generate_data.py    # Lenia simulation generator
│   ├── train_single.py     # Single training execution
│   └── run_hpo.py          # Optuna hyperparameter optimization
├── experiments/            # Checkpoints, logs, and plots
├── utils/                  # Helper functions (visualization, physics)
├── config.yaml             # Global configuration parameters
└── requirements.txt        # Python dependencies
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

### Quick start — no real data needed (`--dummy`)

Both training scripts accept a `--dummy` flag that generates random in-memory tensors instead of loading files from disk.  Use this to verify the full pipeline works before committing to a full training run.

```bash
# Pixel-prediction setup, synthetic data
python scripts/train_single.py --setup pixel --dummy

# JEPA setup, synthetic data
python scripts/train_single.py --setup jepa --dummy

# HPO with dummy data (3 trials, 2 epochs each — instant smoke-test)
python scripts/run_hpo.py --setup pixel --dummy --n-trials 3 --hpo-epochs 2
python scripts/run_hpo.py --setup jepa  --dummy --n-trials 3 --hpo-epochs 2
```

Dummy data consists of 200 random `(1, 64, 64)` frame pairs for training and 40 for validation.  Loss values will not be meaningful, but the code path is identical to a real run.

---

### Running with real Lenia data

#### Where does the data go?

Place your Lenia trajectory files in the `data/` directory:

```
data/
├── lenia_train.npy   ← training trajectories
└── lenia_val.npy     ← validation trajectories
```

HDF5 files (`.h5` / `.hdf5`) are also supported — the dataset class detects the format automatically and reads the array stored under the key `frames`.

#### Expected data format

The loader accepts NumPy arrays in any of these shapes:

| Shape | Meaning |
|---|---|
| `(N, T, H, W)` | N trajectories, T timesteps, spatial H×W — **most common** |
| `(N, T, 1, H, W)` | Same with explicit channel dim |
| `(T, H, W)` | Single trajectory, no channel |
| `(T, 1, H, W)` | Single trajectory, with channel |

- `N` — number of independent Lenia trajectories
- `T` — length of each trajectory (number of frames); at least 2
- `H = W = 64` — spatial resolution expected by the encoder
- Values must be in `[0, 1]` (or any positive range — the loader normalises by the global max)

Each consecutive pair of frames within a trajectory becomes one training sample `(frame_t, frame_t_plus_1)`.  With N=50 trajectories of T=200 frames you get 50×199 = 9 950 training pairs.

#### Running a full training

```bash
python scripts/train_single.py --setup pixel
python scripts/train_single.py --setup jepa
```

Checkpoints and TensorBoard logs are written to `experiments/{setup}_{timestamp}/`.

```bash
# Monitor training live
tensorboard --logdir experiments/
```

---

### Hyperparameter optimisation (`run_hpo.py`)

`run_hpo.py` uses **Optuna** to search over the hyperparameter space defined in `config.yaml` under `hpo.search_space`.  Each trial trains for `hpo.hpo_epochs` epochs (default 20) and reports the validation loss.  A `MedianPruner` discards unpromising trials early.

```bash
# Basic run — 50 trials, results in memory only
python scripts/run_hpo.py --setup pixel
python scripts/run_hpo.py --setup jepa --n-trials 30

# Persistent study — can be interrupted and resumed
python scripts/run_hpo.py --setup pixel \
    --storage sqlite:///experiments/hpo.db \
    --study-name pixel_hpo

# Resume the same study (same --storage and --study-name)
python scripts/run_hpo.py --setup pixel \
    --storage sqlite:///experiments/hpo.db \
    --study-name pixel_hpo --n-trials 20

# Parallel trials (requires shared storage and num_workers: 0 in config.yaml)
python scripts/run_hpo.py --setup jepa \
    --storage sqlite:///experiments/hpo.db \
    --n-jobs 4
```

After all trials complete, the best hyperparameters are printed and saved to:

```
experiments/hpo_{setup}_best_params.yaml
```

Per-trial TensorBoard logs land in `experiments/hpo_{setup}_trial_XXXX/`.

---

### Configuration knobs (`config.yaml`)

All paths and hyperparameters live in **`config.yaml`** at the repository root.  The table below lists the most relevant knobs:

| Section | Key | What it controls |
|---|---|---|
| `data` | `train_path` / `val_path` | Paths to your `.npy` or `.h5` data files |
| `data` | `batch_size` | Mini-batch size during training |
| `data` | `num_workers` | DataLoader worker processes (set to `0` for HPO with `--n-jobs > 1`) |
| `model` | `embed_dim` | Encoder output / embedding dimensionality (default 128) |
| `model` | `ema_momentum` | EMA decay for the JEPA target encoder (default 0.99) |
| `model` | `predictor_hidden_dim` | Hidden width of the JEPA MLP predictor (default 256) |
| `training` | `num_epochs` | Total epochs for a full `train_single.py` run |
| `training` | `learning_rate` | Adam learning rate |
| `training` | `weight_decay` | Adam weight decay |
| `training` | `checkpoint_interval` | Save a checkpoint every N epochs |
| `jepa` | `use_variance_reg` | Enable VICReg variance anti-collapse term (default `false`) |
| `jepa` | `use_covariance_reg` | Enable VICReg covariance anti-redundancy term (default `false`) |
| `jepa` | `var_reg_weight` / `cov_reg_weight` | Loss weights for the regularisers above |
| `hpo` | `n_trials` | Default number of Optuna trials |
| `hpo` | `hpo_epochs` | Training epochs per HPO trial |
| `hpo` | `storage` | Optuna storage URL for persistent / resumable studies |
| `hpo` | `search_space` | Bounds / choices for each tuneable hyperparameter |

CLI flags (`--n-trials`, `--hpo-epochs`, `--storage`, `--study-name`) always override the corresponding `hpo.*` config values when provided.
