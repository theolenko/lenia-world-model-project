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

### Data

#### Where does the data go?

Place your Lenia trajectory files in the `data/` directory:

```
data/
├── lenia_train.npy   ← training trajectories
└── lenia_val.npy     ← validation trajectories
```

HDF5 files (`.h5` / `.hdf5`) are also supported — the dataset class reads the array stored under the key `frames`.

#### Expected array shape

| Shape | Meaning |
|---|---|
| `(N, T, H, W)` | N trajectories, T timesteps, spatial H×W — **most common** |
| `(N, T, 1, H, W)` | Same with explicit channel dim |
| `(T, H, W)` | Single trajectory |

- `H = W = 64` — spatial resolution required by the encoder
- Values in any positive range are fine; the loader normalises by the global max

Each consecutive frame pair within a trajectory becomes one training sample.  With N=50 trajectories of T=200 frames you get 50×199 = 9 950 training pairs.

#### No real data yet? Use `--dummy`

Both scripts accept `--dummy`, which generates random in-memory tensors and skips disk I/O entirely.  Use it to verify the full pipeline before committing to a real run:

```bash
python scripts/train_single.py --setup pixel --dummy
python scripts/train_single.py --setup jepa  --dummy
python scripts/run_hpo.py      --setup pixel --dummy --n-trials 3 --hpo-epochs 2
```

---

### Full training workflow (step by step)

> The same workflow applies to both `pixel` and `jepa` setups.  Run it once for each.

#### Step 1 — Smoke-test the pipeline

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
| `data` | `train_path` / `val_path` | Paths to your `.npy` or `.h5` data files |
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
