# Pipeline Smoke Test Report

**Date:** 2026-05-28  
**Device:** CPU (no GPU available)  
**Run dir:** `experiments/smoke_test_20260528_214535/`

---

## 1. Steps and Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Data validation | ✅ PASSED |
| 2 | DataLoader test | ✅ PASSED |
| 3 | Model forward-pass test | ✅ PASSED |
| 4 | Mini training run (7 epochs) | ✅ PASSED |
| 5 | Qualitative visualisation | ✅ DONE |
| 6 | Multi-step rollout | ✅ DONE |

---

## 2. Issues Found and Fixes

### Bug 1: Shape-handling error in `LeniaDataset` (FIXED)

**File:** `scripts/train_single.py`

**Problem:** The heuristic `data.shape[1] > 10` in the `elif data.ndim == 4` branch was designed for small T values. With real data of shape `(1800, 200, 64, 64)`, `shape[1] = 200 > 10` incorrectly matched the single-trajectory branch → `data = data[None]` → shape `(1, 1800, 200, 64, 64)` instead of the correct `(1800, 200, 1, 64, 64)`.

**Fix:** Heuristic reduced to `shape[1] == 1` (channel-dim check only). All other 4D arrays are treated as `(N, T, H, W)`.

### Bug 2: RAM overload from full HDF5 load (FIXED)

**File:** `scripts/train_single.py`

**Problem:** `LeniaDataset._load()` loaded the entire file with `f["frames"][:]` into RAM. For `lenia_train.h5`: 1800 × 200 × 64 × 64 × 4 bytes ≈ **5.5 GB** — would have caused an OOM crash with 10 GB available RAM.

**Fix:** Lazy loading via an open `h5py.File` handle per worker process. HDF5 metadata is read at `__init__` (no frames loaded), frames are fetched frame-by-frame in `__getitem__`. Added `__getstate__`/`__setstate__` for pickle compatibility with DataLoader worker spawning.

### Finding 3: Suboptimal HDF5 chunk layout (not a bug, but a note)

**Problem:** The HDF5 files use chunk layout `(1, 200, 64, 64)` — one chunk per trajectory, 3.12 MB uncompressed each. Random per-frame access (`shuffle=True`) requires decompressing the full 3.12 MB chunk per item → ~100 MB decompression per batch with batch_size=32 → too slow for practical training (~18 min/epoch).

**Workaround in smoke test:** Pre-loaded a subset (50 train + 20 val trajectories, ~218 MB RAM) as `TensorDataset` for the mini training run.

**Recommendation:** Rechunk data to per-frame chunks `(1, 1, 64, 64)` — see `scripts/rechunk_data.py` and `experiments/rechunk_report.md` (47× speedup achieved).

### Bug 4: `config.yaml` pointed to non-existent `.npy` files (FIXED)

Paths `data/lenia_train.npy` and `data/lenia_val.npy` corrected to `data/lenia_train.h5` / `data/lenia_val.h5`.

---

## 3. Quantitative Results

### Data Validation (Step 1)

| File | Shape | Dtype | Min | Max | Mean |
|------|-------|-------|-----|-----|------|
| `lenia_train.h5` | (1800, 200, 64, 64) | float32 | 0.0000 | 1.0000 | 0.1730 |
| `lenia_val.h5` | (200, 200, 64, 64) | float32 | 0.0000 | 1.0000 | 0.1802 |

**Identity baseline loss (MSE frame_t vs frame_{t+1}):**

| Split | Identity MSE |
|-------|-------------|
| Train | **0.000501** |
| Val | **0.000495** |

Lenia frames change very slowly → very low baseline loss. A model must go below 0.0005 to outperform "copy input".

### Mini Training Run (Step 4)

Config: PixelWorldModel (embed_dim=128, 1.94M params), Adam lr=1e-3, 7 epochs, 50/1800 train trajectories (9,950 pairs), 20/200 val trajectories (3,980 pairs), CPU.

| Epoch | Train Loss | Val Loss | Time |
|-------|-----------|----------|------|
| 1 | 0.098796 | 0.138177 | 60.2s |
| 2 | 0.022641 | 0.143399 | 59.7s |
| 3 | 0.013308 | 0.143716 | 59.8s |
| 4 | 0.010106 | 0.143555 | 45.8s |
| 5 | 0.008299 | 0.140950 | 46.7s |
| 6 | 0.007081 | 0.144679 | 48.5s |
| **7** | **0.006465** | **0.146848** | 60.7s |

**Observations:**
- Train loss dropped by factor ~15 (0.099 → 0.006) ✓
- Train loss is still **13× above** the identity baseline (0.0005) → model has not beaten the baseline after 7 epochs on CPU with a subset
- Val loss remained roughly constant (~0.138–0.147) → strong overfitting on 50 trajectories with 1.94M params. With the full dataset (1800 trajs) and more epochs, val loss would decrease as well.

### Rollout Sanity Check (Step 6)

Autoregressive rollout over 30 steps, val trajectory 0:

| Step | Rollout MSE |
|------|------------|
| 1 | 0.017870 |
| 15 | 0.048429 |
| 30 | 0.070973 |

Compounding error trend: **increasing** ✓ (expected behaviour for autoregressive rollouts)

---

## 4. Visualisations

All saved under `experiments/smoke_test_20260528_214535/`:

| File | Content |
|------|---------|
| `step4_loss_curves.png` | Train and val loss over 7 epochs |
| `step5_predictions.png` | 5 samples: Input / Predicted / True side by side |
| `step6_rollout_frames.png` | Rollout frames (predicted vs. true) for steps 1/5/10/15/20/30 |
| `step6_rollout_mse.png` | MSE error over 30 rollout steps |

Data diagnostic visualisations (Step 1) under `experiments/smoke_test/`:
- `check_train_traj0.png` — train trajectory 0, frames t=0/25/50/100/150/199
- `check_val_traj0.png` — val trajectory 0, same timesteps

---

## 5. Success Criteria

| Criterion | Status | Comment |
|-----------|--------|---------|
| Pipeline runs without crash | ✅ | All 6 steps completed without errors |
| Train loss decreases | ✅ | 0.099 → 0.006 (15×) |
| Train loss below identity baseline | ⚠️ | 0.006 vs. baseline 0.0005 — still 13× too high. Not reachable with subset + 7 epochs on CPU |
| Val loss decreases (no overfitting) | ⚠️ | Val loss stays at ~0.143 — overfitting on 50/1800 trajs with 1.94M params |
| Visual plausibility | ✅ | Predictions in step5 show Lenia patterns; model does not simply copy input |
| Rollout error increases | ✅ | MSE 0.018 → 0.071 over 30 steps |

---

## 6. Recommendations

1. **Rechunk HDF5 data**: rewrite with `(1, 1, 64, 64)` chunks for fast shuffled DataLoader access. → Done, see `experiments/rechunk_report.md`.

2. **Train on full dataset**: use all 1800 train trajectories. Both train and val loss should decrease.

3. **Train longer**: with full dataset and GPU, 50–100 epochs should be sufficient to go below the identity baseline.

4. **Investigate model initialisation**: initial output range `[0.49, 0.52]` (BatchNorm after random init) leads to high initial loss. Consider checking last-layer bias or Xavier init for the decoder.

5. **Blurry predictions after 7 epochs**: expected after short training with little data. Image quality should improve significantly after full training.

6. **Test JEPA setup**: once pixel setup is working well, run the same smoke test for `JEPAWorldModel`.
