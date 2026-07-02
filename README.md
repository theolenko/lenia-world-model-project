# lenia-world-model-project

Code for the seminar project in "Mathematics and AI Internship" (Data Science, Uni Leipzig).

Comparison of three world model architectures on Lenia cellular automata, trained on two dataset versions.

| Setup            | Encoder                                      | Task                             | Training signal                     |
| ---------------- | -------------------------------------------- | -------------------------------- | ----------------------------------- |
| **Pixel-CNN**    | CNN (4× stride-2 conv)                       | Predict next frame (pixel space) | MSE vs ground truth                 |
| **Pixel-ViT**    | ViT (patch_size=8, 6 layers, 8 heads)        | Predict next frame (pixel space) | MSE vs ground truth                 |
| **JEPA-CNN**     | CNN (EMA target encoder, flat embedding)     | Predict next frame's embedding   | MSE in latent space + VICReg        |
| **Patch-JEPA**   | ViT (pool=False, patch embeddings, EMA)      | Predict next frame's patch embs  | MSE over patches + VICReg           |

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
│   ├── decoder.py                     # LeniaDecoder, SpatialLeniaDecoder, PatchDecoder
│   ├── predictor.py                   # LeniaPredictor (flat MLP), PatchPredictor (per-patch MLP)
│   └── world_model.py                 # PixelWorldModel, JEPAWorldModel, PatchJEPAWorldModel
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

- **Online encoder** encodes `frame_t` → `z_context` (flat vector, shape `(B, embed_dim)`)
- **MLP predictor** maps `z_context` → `z_pred`
- **Target encoder** (EMA copy of online) encodes `frame_t+1` → `z_target` (no grad)
- Loss: MSE(`z_pred`, `z_target`) + VICReg(variance) + VICReg(covariance) applied to `z_context`

The encoder uses **Global Average Pooling (GAP)** which collapses the `(B, 256, 4, 4)` spatial feature map into a flat `(B, 256)` vector. This discards all positional information — limiting reconstruction quality.

### PatchJEPAWorldModel

Same JEPA principle but with **ViTEncoder(pool=False)**, keeping the 64 patch embeddings as separate vectors instead of collapsing them with GAP.

- **Online encoder** `frame_t` → `(B, 64, embed_dim)` patch sequences
- **PatchPredictor** maps patch embeddings → predicted patch embeddings (same MLP applied per-patch via `nn.Linear`)
- **Target encoder** (EMA) encodes `frame_t+1` → `(B, 64, embed_dim)` (no grad)
- Loss: MSE over all patch positions + VICReg on flattened `(B×64, embed_dim)`

After JEPA training, a **PatchDecoder** is trained separately:
```
frame_t → frozen ViTEncoder → z_enc → frozen Predictor → z_pred → PatchDecoder → frame_t
```
The decoder is trained to reconstruct `frame_t` from `predictor(encoder(frame_t))`.

> The JEPA EMA property ensures `predictor(encoder(frame_t)) ≈ encoder(frame_{t+1})`, so the
> decoder is trained to invert a next-frame embedding back to pixels — the correct objective.
> See `evaluation/README.md` for quantitative results.

Autoregressive rollout at inference:
```
z_0 = encode(frame_0)
z_1 = predictor(z_0)  →  decoder(z_1)  =  frame_1_pred
z_2 = predictor(z_1)  →  decoder(z_2)  =  frame_2_pred
...
```

### Why JEPA works better with ViT than CNN

The key difference is what gets passed through the prediction bottleneck:

| | CNN + GAP (JEPA-CNN) | ViT, pool=False (Patch-JEPA) |
|---|---|---|
| Encoder output | `(B, D)` — one vector | `(B, 64, D)` — 64 spatial vectors |
| Spatial structure | Lost (GAP averages everything) | Preserved (each patch = one 8×8 region) |
| Predictor input | Single global summary | Per-patch local features |
| Decodable? | Not spatially (flat → no positional info) | Yes — reshape patches back to 8×8 grid |
| Autoregressive rollout | Reconstruct only, not predict | Predict **and** decode next frame |

ViT also produces better embeddings for JEPA because its **self-attention** explicitly models relationships between all 64 patches. The predictor can thus learn *which patches change* between frames — a much richer signal than predicting a single pooled vector.

#### JEPA pitfalls (do not revert)

| Issue                                           | Fix                                                                                                        |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `ema_momentum` must be close to 1 (~0.99–0.999) | `update_target()` is called **per batch**, not per epoch. `0.92^300 ≈ 0` causes immediate target collapse. |
| `var_reg_weight` should be small (~0.05–0.1)    | Weight 1.0 dominates the loss and prevents the predictor from learning.                                    |
| VICReg only on `z_context`                      | `z_target` is inside `torch.no_grad()` — gradients are zero, regularising it is a no-op.                   |
| VICReg on patch embeddings needs flattening     | Functions expect `(B, D)`. Flatten `(B, N, D)` → `(B*N, D)` before applying.                              |

---

## Training

### Quick smoke test (no data needed)

```bash
python scripts/train_single.py --setup pixel      --dummy
python scripts/train_single.py --setup jepa       --dummy
python scripts/train_single.py --setup patch_jepa --dummy
```

### Full training workflow

```bash
# 1. Prepare data (once)
python scripts/rechunk_data.py --input data/v2_lenia_train.h5 --output data/v2_lenia_train_chunked.h5
python scripts/rechunk_data.py --input data/v2_lenia_val.h5   --output data/v2_lenia_val_chunked.h5

# 2. HPO (optional — best params already in v2 configs)
python scripts/run_hpo.py --setup patch_jepa --config config_cluster_patch_jepa_v2.yaml \
    --storage sqlite:///experiments/hpo_patch_jepa_v2.db --study-name patch_jepa_v2_hpo

# 3. Full training
python scripts/train_single.py --setup pixel      --config config_cluster_pixel_v2.yaml
python scripts/train_single.py --setup jepa       --config config_cluster_jepa_v2.yaml
python scripts/train_single.py --setup patch_jepa --config config_cluster_patch_jepa_v2.yaml

# 4. Train PatchDecoder on frozen Patch-JEPA encoder
python scripts/train_patch_decoder.py --jepa-checkpoint experiments/.../checkpoint_best.pt

# 5. Monitor
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

| Model       | Trial | val_loss (10 epochs) | lr       | embed_dim | Notes                              |
| ----------- | ----- | -------------------- | -------- | --------- | ---------------------------------- |
| Pixel-CNN   | #17   | 0.01897              | 5.992e-4 | 128       | weight_decay=3.641e-6              |
| Pixel-ViT   | #20   | 0.02242              | 6.440e-5 | 256       | weight_decay=4.032e-5              |
| JEPA-CNN    | #12   | 0.04926              | 3.856e-4 | 256       | ema=0.9926, var_w=0.0503           |
| Patch-JEPA  | #6    | 0.01794              | 1.523e-4 | 128       | ema=0.9919, var_w=0.3769, batch=16 |

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

| Job           | Script                       | Time limit | Why                                      |
| ------------- | ---------------------------- | ---------- | ---------------------------------------- |
| Pixel-CNN     | `train_pixel_v2_job.sh`      | 8 h        | ~4–5 h for 100 epochs                    |
| JEPA-CNN      | `train_jepa_v2_job.sh`       | 8 h        | ~4–5 h for 100 epochs                    |
| Pixel-ViT     | `train_vit_v2_job.sh`        | **12 h**   | ~9 h for 100 epochs (~320 s/epoch)       |
| Patch-JEPA    | `train_patch_jepa_v2_job.sh` | **24 h**   | ~20 h for 100 epochs (~730 s/epoch)      |
| PatchDecoder  | `train_patch_decoder_v2_job.sh` | 6 h     | ~1.5 h for 50 epochs (~110 s/epoch)      |

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

## Results

### v2 Full Training Results

| Model        | Final val_loss | Epochs | Notes                                      |
| ------------ | -------------- | ------ | ------------------------------------------ |
| Pixel-CNN    | ~0.019         | 100    | Pixel MSE — direct next-frame prediction   |
| Pixel-ViT    | ~0.022         | 100    | Pixel MSE — ViT encoder, same task         |
| JEPA-CNN     | ~0.049         | 100    | Embedding MSE + VICReg (not comparable)    |
| Patch-JEPA   | 0.01340        | 100    | Embedding MSE over 64 patches + VICReg     |
| PatchDecoder | —              | 100    | val_loss=0.000845 (next-frame MSE); trained on predictor(encoder(frame_t)) → frame_{t+1} |

> Note: JEPA and Pixel model losses are in different spaces (embedding vs pixel MSE) and are **not directly comparable**.

### Pixel-space evaluation (latest, job 25045438 — active trajectories only)

| Model | One-step MSE ↓ | PSNR ↑ | SSIM ↑ | Rollout horizon (active traj.) |
|-------|---------------|--------|--------|-------------------------------|
| Pixel-CNN | 0.00829 | 20.93 dB | 0.918 | none (above identity at step 1) |
| Pixel-ViT | 0.00841 | 20.92 dB | 0.869 | step 8 |
| Patch-JEPA | **0.00074** | **31.34 dB** | **0.949** | **step 10** |

Identity baseline (no-change): MSE ≈ 0.00045 one-step, step-1 rollout ≈ 0.00333, step-30 ≈ 0.113.

Evaluation is on **active trajectories** (organism does not die during the eval window). Earlier
numbers using random trajectory selection were dominated by dying-organism trajectories where all
models trivially predict black, making Pixel-CNN appear best. On active organisms, Patch-JEPA is
numerically best to step 10, Pixel-ViT to step 8; Pixel-CNN fails to beat the no-change baseline.
Patch-JEPA one-step outputs have a visible brightness bias (pred mean ~0.14 vs GT ~0.10) and
rollouts degrade to salt-and-pepper noise after ~step 10. See `evaluation/README.md`.

### Generated visualizations (`experiments/plots/`)

| File | Description |
|---|---|
| `results_loss_curves_pixel.png` | Pixel-CNN vs Pixel-ViT training curves (shared y-axis) |
| `results_loss_curves_jepa.png` | JEPA-CNN stage 1 + decoder stage 2 |
| `results_loss_curves_patch_jepa.png` | Patch-JEPA stage 1 + PatchDecoder stage 2 |
| `results_predictions_pixel.png` | Pixel-CNN: input \| predicted t+1 \| ground truth |
| `results_predictions_vit.png` | Pixel-ViT: input \| predicted t+1 \| ground truth |
| `results_predictions_jepa.png` | JEPA-CNN reconstruction via spatial decoder |
| `results_predictions_patch_jepa.png` | Patch-JEPA: encode → predictor → decode |
| `trajectory_comparison_cnn.gif` | Pixel-CNN teacher-forced rollout |
| `trajectory_comparison_jepa.gif` | JEPA-CNN reconstruction (teacher-forced) |
| `trajectory_autoregressive_cnn.gif` | Pixel-CNN autoregressive rollout |
| `trajectory_autoregressive_vit.gif` | Pixel-ViT autoregressive rollout |
| `trajectory_autoregressive_jepa.gif` | JEPA-CNN autoregressive (spatial decode loop) |
| `trajectory_autoregressive_patch_jepa.gif` | Patch-JEPA true rollout: encode → predictor×T → decoder |

---

## Evaluation & Intervention Analysis

### Pixel-space evaluation (`evaluation/evaluate_all.py`)

Fair comparison of Pixel-CNN, Pixel-ViT, and Patch-JEPA in pixel space.
JEPA-CNN is excluded — its predictor outputs flat embeddings with no pixel decoder.

| Scenario | What is measured | Data |
|---|---|---|
| One-step prediction | MSE / PSNR / SSIM at t+1 | 2000 frame pairs from val set |
| Multi-step rollout | MSE per step vs. identity baseline → competence horizon | 5 val trajectories, 30 steps |
| OOD robustness | One-step MSE at σ ∈ {0.00, 0.02, 0.05, 0.10, 0.20} noise | 256 val pairs |

```bash
python evaluation/evaluate_all.py --n-traj 5 --rollout-steps 30 --max-samples 2000
```

Results and plots: `evaluation/results/`  
Detailed findings: `evaluation/README.md`

### Intervention analysis (`interventions/run_all_interventions.py`)

Causal probing of whether each model correctly tracks Lenia physics after a state perturbation.

**Protocol:**
1. GT: real Lenia runs `t_star` steps from `traj[0]`, intervention applied to GT's own frame → Lenia continues
2. Each model: autoregressively rolls `t_star` steps, intervention applied to **model's own predicted frame** (same random parameters as GT) → model continues
3. MSE between model prediction and **shared Lenia GT** per step, averaged over 5 trajectories

All models are compared against the same ground truth, so numbers are directly comparable across models.

| Intervention | What it tests |
|---|---|
| `inject_blob` | New organism seed added at a random location |
| `zero_region` | Circular region killed (set to 0) |
| `mirror_patch` | Local patch horizontally flipped |
| `scale_density` | All cell values scaled by 1.5 |
| `add_noise` | σ=0.05 Gaussian noise added everywhere |

```bash
python interventions/run_all_interventions.py --n-traj 5 --t-star 10 --n-steps 30
```

Results: `experiments/intervention_results/`  
Detailed findings: `evaluation/README.md`

**Output files:**
- `intervention_{name}.png` — MSE curve (averaged over 5 trajectories) for all models vs. Lenia GT
- `snapshots_{name}.png` — Full timeline grid: pre-rollout → before perturbation → PERTURBED (red border) → post-rollout, one row per model + GT
- `gif_{name}_{model}.gif` — Animated `[Model | Lenia GT]` side by side (15 GIFs total: 5 interventions × 3 models)
- `intervention_heatmap.png` — Mean MSE steps 1–10 for all (model, intervention) combinations

All plots use **grayscale colormap** matching the single-channel Lenia frames.

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
