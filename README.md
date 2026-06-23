# lenia-world-model-project

Code for the seminar project in "Mathematics and AI Internship" (Data Science, Uni Leipzig).

Comparison of three world model architectures on Lenia cellular automata, trained on two dataset versions.

| Setup         | Encoder                               | Task                             | Training signal              |
| ------------- | ------------------------------------- | -------------------------------- | ---------------------------- |
| **Pixel-CNN** | CNN (4× stride-2 conv)                | Predict next frame (pixel space) | MSE vs ground truth          |
| **Pixel-ViT** | ViT (patch_size=8, 6 layers, 8 heads) | Predict next frame (pixel space) | MSE vs ground truth          |
| **JEPA-CNN**  | CNN (EMA target encoder)              | Predict next frame's embedding   | MSE in latent space + VICReg |

---

## Project Structure

```text
lenia-world-model-project/
├── data/                              # Lenia trajectory files (gitignored)
│   ├── lenia_train_chunked.h5         # v1 training data — per-frame chunks
│   ├── lenia_val_chunked.h5           # v1 validation data — per-frame chunks
│   ├── v2_lenia_train_chunked.h5      # v2 training data — creatures pre-placed at t=0
│   └── v2_lenia_val_chunked.h5        # v2 validation data
├── models/                            # Neural network architectures
│   ├── encoder.py                     # LeniaEncoder (CNN), SimpleCNNEncoder, ViTEncoder
│   ├── decoder.py                     # Pixel-prediction decoder (transposed conv)
│   ├── predictor.py                   # JEPA latent predictor (2-layer MLP)
│   └── world_model.py                 # PixelWorldModel and JEPAWorldModel + encoder registry
├── training/                          # Training logic
│   ├── trainer.py                     # Training loop, validation, TensorBoard logging, checkpointing
│   └── losses.py                      # MSE loss, VICReg variance/covariance terms
├── scripts/                           # Executable scripts
│   ├── generate_data.py               # Lenia simulation data generator
│   ├── train_single.py                # Training entry point + LeniaDataset
│   ├── run_hpo.py                     # Optuna hyperparameter optimisation
│   ├── rechunk_data.py                # Convert HDF5 from per-trajectory to per-frame chunks
│   ├── visualize_results.py           # Loss curves, pixel predictions, autoregressive GIF
│   ├── sync_to_cluster.sh             # Sync code to Clara cluster via rsync
│   ├── sync_data_to_cluster.sh        # Transfer HDF5 data to cluster (resumable)
│   ├── cluster_setup.sh               # One-time conda env setup on cluster
│   ├── hpo_pixel_v2_job.sh            # SLURM job: Pixel-CNN HPO on v2 data
│   ├── hpo_vit_v2_job.sh              # SLURM job: Pixel-ViT HPO on v2 data
│   ├── hpo_jepa_v2_job.sh             # SLURM job: JEPA-CNN HPO on v2 data
│   ├── train_pixel_v2_job.sh          # SLURM job: Pixel-CNN full training (v2)
│   ├── train_vit_v2_job.sh            # SLURM job: Pixel-ViT full training (v2, 12 h limit)
│   ├── train_jepa_v2_job.sh           # SLURM job: JEPA-CNN full training (v2)
│   ├── hpo_pixel_job.sh               # SLURM job: Pixel-CNN HPO (v1 data, legacy)
│   ├── hpo_jepa_job.sh                # SLURM job: JEPA-CNN HPO (v1 data, legacy)
│   ├── train_pixel_job.sh             # SLURM job: Pixel-CNN training (v1 data, legacy)
│   └── train_jepa_job.sh              # SLURM job: JEPA-CNN training (v1 data, legacy)
├── experiments/                       # Checkpoints, logs, and results (gitignored)
│   ├── cluster/                       # Results downloaded from Clara
│   │   ├── hpo_pixel_v2.db            # Pixel-CNN HPO study (SQLite, 20 trials)
│   │   ├── hpo_vit_v2.db              # Pixel-ViT HPO study (SQLite, 20 trials)
│   │   └── hpo_jepa_v2.db             # JEPA-CNN HPO study (SQLite, 20 trials)
│   ├── exploration.ipynb              # v1 data analysis
│   └── exploration_v2.ipynb           # v2 data analysis (diversity, mass over time, MSE)
├── checkpoints/                       # Trained model weights (tracked via Git LFS)
│   ├── pixel_v1.pt                    # Pixel-CNN on v1 data — 100 epochs, val_loss=0.025
│   └── jepa_v1.pt                     # JEPA-CNN on v1 data — 100 epochs
├── config.yaml                        # Local development config (v1 data paths)
├── config_cluster_pixel_v2.yaml       # Cluster config — Pixel-CNN, v2 data, HPO best params
├── config_cluster_vit_v2.yaml         # Cluster config — Pixel-ViT, v2 data, HPO best params
├── config_cluster_jepa_v2.yaml        # Cluster config — JEPA-CNN, v2 data, HPO best params
├── config_cluster_pixel.yaml          # Cluster config — Pixel-CNN, v1 data (legacy)
├── config_cluster_jepa.yaml           # Cluster config — JEPA-CNN, v1 data (legacy)
└── requirements.txt                   # Python dependencies
```

---

## Dataset Versions

### v1 — Random-noise initialisation

- 2000 trajectories × 200 frames, 64×64 grayscale
- Initial state: random noise; creatures emerge (or not) over time
- Train: 1800 trajectories, Val: 200 trajectories

### v2 — Pre-placed creatures (current)

- Same size as v1 but with Lenia organisms already established at t=0
- More diverse initial configurations (multiple species, varied positions)
- Higher data quality: less variation between trajectories from random-noise variance
- HDF5 keys: `frames`, shape `(N, T, H, W)` — identical format to v1

#### Why rechunk?

Raw files use chunk layout `(1, 200, 64, 64)` — one 3 MB chunk per trajectory. Shuffled DataLoader access decompresses the full chunk for every sample, giving ~1200 ms/batch.

After rechunking to `(1, 1, 64, 64)` (one frame per chunk), only the requested frame is decompressed: **~25 ms/batch — 47× faster**.

```bash
python scripts/rechunk_data.py --input data/v2_lenia_train.h5 --output data/v2_lenia_train_chunked.h5
python scripts/rechunk_data.py --input data/v2_lenia_val.h5   --output data/v2_lenia_val_chunked.h5
```

---

## Model Architectures

### Encoder registry (`world_model.py`)

| `encoder_type` | Class                   | Architecture                                                                 |
| -------------- | ----------------------- | ---------------------------------------------------------------------------- |
| `"cnn"`        | `LeniaEncoder`          | 4× stride-2 conv (1→16→32→64→128), Global AvgPool, Linear projection         |
| `"simple_cnn"` | `SimpleCNNEncoder`      | Lighter CNN variant                                                          |
| `"vit"`        | `ViTEncoder(pool=True)` | patch_size=8 → 64 patches, 6 Transformer layers, 8 heads, GAP to flat vector |

Set `encoder_type` in the `model` section of any config YAML.

### PixelWorldModel

Encoder embeds `frame_t` → decoder (transposed conv) reconstructs predicted `frame_t+1`.  
Loss: MSE in pixel space.

### JEPAWorldModel

- **Online encoder** encodes `frame_t` → `z_context`
- **MLP predictor** maps `z_context` → `z_pred`
- **Target encoder** (EMA copy of online) encodes `frame_t+1` → `z_target` (no grad)
- Loss: MSE(`z_pred`, `z_target`) + VICReg(variance) + VICReg(covariance) applied to `z_context`

#### JEPA pitfalls (do not revert)

| Issue                                           | Fix                                                                                                        |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `ema_momentum` must be close to 1 (~0.99–0.999) | `update_target()` is called **per batch**, not per epoch. `0.92^300 ≈ 0` causes immediate target collapse. |
| `var_reg_weight` should be small (~0.05–0.1)    | Weight 1.0 dominates the loss and prevents the predictor from learning.                                    |
| VICReg only on `z_context`                      | `z_target` is inside `torch.no_grad()` — gradients are zero, regularising it is a no-op.                   |

---

## Training

### Quick smoke test (no data needed)

```bash
python scripts/train_single.py --setup pixel --dummy
python scripts/train_single.py --setup jepa  --dummy
```

### Full training workflow

```bash
# 1. Prepare data (once)
python scripts/rechunk_data.py --input data/v2_lenia_train.h5 --output data/v2_lenia_train_chunked.h5
python scripts/rechunk_data.py --input data/v2_lenia_val.h5   --output data/v2_lenia_val_chunked.h5

# 2. HPO (optional — best params already in v2 configs)
python scripts/run_hpo.py --setup pixel --config config_cluster_pixel_v2.yaml \
    --storage sqlite:///experiments/hpo_pixel_v2.db --study-name pixel_v2_hpo

# 3. Full training
python scripts/train_single.py --setup pixel --config config_cluster_pixel_v2.yaml
python scripts/train_single.py --setup jepa  --config config_cluster_jepa_v2.yaml

# 4. Monitor
tensorboard --logdir experiments/
```

### LR scheduling

All v2 configs use `use_lr_schedule: true`, which enables **Cosine Annealing** (`T_max = num_epochs`). The learning rate decays smoothly from the initial value to ~0 over the full run. Disable by setting `use_lr_schedule: false`.

### Early stopping

Training stops automatically when validation loss does not improve for `early_stopping_patience` epochs (default: 15). Best checkpoint saved as `checkpoint_best.pt`.

---

## Hyperparameter Optimisation

HPO uses [Optuna](https://optuna.org) with a **SQLite backend** for persistence — the study survives SLURM job cancellations and can be resumed.

```bash
python scripts/run_hpo.py --setup pixel \
    --storage sqlite:///experiments/hpo_pixel_v2.db \
    --study-name pixel_v2_hpo \
    --n-trials 20 --hpo-epochs 20
```

### v2 HPO best results

| Model     | Trial | val_loss | lr       | embed_dim | Notes                    |
| --------- | ----- | -------- | -------- | --------- | ------------------------ |
| Pixel-CNN | #17   | 0.01897  | 5.992e-4 | 128       | weight_decay=3.641e-6    |
| Pixel-ViT | #20   | 0.02242  | 6.440e-5 | 256       | weight_decay=4.032e-5    |
| JEPA-CNN  | #12   | 0.04926  | 3.856e-4 | 256       | ema=0.9926, var_w=0.0503 |

Best params are already written into the `config_cluster_*_v2.yaml` files.

---

## Cluster Training (Clara, Uni Leipzig)

### Prerequisites

- SC account, SSH key at [portal.sc.uni-leipzig.de](https://portal.sc.uni-leipzig.de)
- VPN active (`openconnect` or Cisco AnyConnect)

### One-time setup

```bash
bash scripts/sync_to_cluster.sh
ssh <unilogin>@login01.sc.uni-leipzig.de
bash ~/lenia-world-model/scripts/cluster_setup.sh
# exit, then:
bash scripts/sync_data_to_cluster.sh
```

### Per training run

```bash
# Sync code
bash scripts/sync_to_cluster.sh

# Submit v2 training jobs
ssh <unilogin>@login01.sc.uni-leipzig.de "cd ~/lenia-world-model && \
  sbatch scripts/train_pixel_v2_job.sh && \
  sbatch scripts/train_jepa_v2_job.sh  && \
  sbatch scripts/train_vit_v2_job.sh"

# Monitor
ssh <unilogin>@login01.sc.uni-leipzig.de "squeue -u <unilogin>"
ssh <unilogin>@login01.sc.uni-leipzig.de "tail -f logs_lenia_pixel_v2_JOBID.out"

# Download results
rsync -avz <unilogin>@login01.sc.uni-leipzig.de:~/lenia-results/ ./experiments/cluster/
```

### Job time limits

| Job       | Script                  | Time limit | Why                                |
| --------- | ----------------------- | ---------- | ---------------------------------- |
| Pixel-CNN | `train_pixel_v2_job.sh` | 8 h        | ~4–5 h for 100 epochs              |
| JEPA-CNN  | `train_jepa_v2_job.sh`  | 8 h        | ~4–5 h for 100 epochs              |
| Pixel-ViT | `train_vit_v2_job.sh`   | **12 h**   | ~9 h for 100 epochs (~320 s/epoch) |

### HPO on cluster

HPO results are stored to `~/lenia-world-model/experiments/hpo_*.db` (persistent home dir, not lscratch — survives job cancellation).

```bash
ssh <unilogin>@login01.sc.uni-leipzig.de "cd ~/lenia-world-model && \
  sbatch scripts/hpo_pixel_v2_job.sh && \
  sbatch scripts/hpo_vit_v2_job.sh   && \
  sbatch scripts/hpo_jepa_v2_job.sh"
```

### Cluster rules

| Rule                | Details                                                                                                               |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| No direct execution | Submit via `sbatch` — never run training on login nodes                                                               |
| No sudo             | Install packages via `conda`/`pip` only                                                                               |
| Use lscratch        | Write temp data to `/lscratch/$SLURM_JOB_ID` — auto-deleted after job                                                 |
| Save results early  | HPO DB written to `~/lenia-world-model/experiments/` (persistent), results copied to `~/lenia-results/` in final step |

### Troubleshooting

**GPU not available:** Check `--gres=gpu:1` and `--partition=clara` in job script.

**Out of memory:** Reduce `batch_size` in the config YAML, or increase `--mem=64G`.

**VPN tun device failure:** Kernel module mismatch — reboot fixes it. Confirm with `ip link show`.

**HPO DB has only 1 RUNNING trial after cancellation:** The DB at `~/lenia-world-model/experiments/` is authoritative. Filter completed trials with `t.value is not None`.

---

## Config YAML reference

| Section    | Key                       | What it controls                                            |
| ---------- | ------------------------- | ----------------------------------------------------------- |
| `data`     | `train_path` / `val_path` | Paths to rechunked `.h5` files                              |
| `data`     | `batch_size`              | Mini-batch size                                             |
| `data`     | `num_workers`             | DataLoader workers (use `0` with HPO `--n-jobs > 1`)        |
| `model`    | `encoder_type`            | `"cnn"` / `"vit"` — selects encoder architecture            |
| `model`    | `embed_dim`               | Encoder output dimensionality                               |
| `model`    | `ema_momentum`            | EMA decay for JEPA target encoder (per-batch — keep ≥ 0.99) |
| `model`    | `predictor_hidden_dim`    | Hidden width of JEPA MLP predictor                          |
| `training` | `num_epochs`              | Maximum training epochs                                     |
| `training` | `early_stopping_patience` | Stop after N epochs without val improvement (0 = disabled)  |
| `training` | `learning_rate`           | Adam initial learning rate                                  |
| `training` | `weight_decay`            | Adam weight decay                                           |
| `training` | `use_lr_schedule`         | `true` → CosineAnnealingLR over full training run           |
| `training` | `checkpoint_interval`     | Save checkpoint every N epochs                              |
| `jepa`     | `use_variance_reg`        | VICReg variance term — prevents representation collapse     |
| `jepa`     | `use_covariance_reg`      | VICReg covariance term — prevents redundant dimensions      |
| `jepa`     | `var_reg_weight`          | Weight for variance loss term (keep small, ~0.05–0.1)       |
| `jepa`     | `cov_reg_weight`          | Weight for covariance loss term                             |
| `hpo`      | `n_trials`                | Number of Optuna trials                                     |
| `hpo`      | `hpo_epochs`              | Training epochs per HPO trial                               |
| `hpo`      | `storage`                 | Optuna SQLite URL (e.g. `sqlite:///experiments/hpo.db`)     |
| `hpo`      | `search_space`            | Bounds / choices for each tuneable hyperparameter           |

---

## Branching Strategy

- **main**: Stable, production-ready code.
- **develop**: Integration branch — all pull requests target this branch.

```bash
git fetch origin && git pull origin develop
git checkout -b feature/your-feature-name
# ... work ...
git push origin feature/your-feature-name
# open PR → develop
```
