"""Pipeline End-to-End Smoke Test (Steps 2–6)

Runs sequentially:
  Step 2 – DataLoader verification
  Step 3 – Model forward-pass checks
  Step 4 – Mini training run (N_EPOCHS epochs, pixel setup)
  Step 5 – Qualitative prediction visualisation (input / pred / true)
  Step 6 – Multi-step autoregressive rollout + compounding-error plot

Usage (from project root):
    source lenia-wm/bin/activate
    python scripts/smoke_test.py

Requires the rechunked HDF5 files (see scripts/rechunk_data.py):
    data/lenia_train_chunked.h5  (chunks 1×1×64×64, LZF)
    data/lenia_val_chunked.h5

Step 4 uses lazy LeniaDataset access on a Subset of N_TRAIN_TRAJS / N_VAL_TRAJS
trajectories — no RAM pre-loading needed with per-frame chunks (~25 ms/batch).
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.world_model import PixelWorldModel
from scripts.train_single import LeniaDataset
from training.trainer import Trainer

N_EPOCHS      = 7
BATCH_SIZE    = 32
LR            = 1e-3
SEED          = 42
N_TRAIN_TRAJS = 50   # subset of 1800 train trajectories
N_VAL_TRAJS   = 20   # subset of 200 val trajectories

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR   = ROOT / "experiments" / f"smoke_test_{TIMESTAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = ROOT / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device     : {DEVICE}")
print(f"Output dir : {OUT_DIR.relative_to(ROOT)}")

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def step2_dataloader_test() -> None:
    print("\n" + "=" * 60)
    print("  STEP 2: DataLoader verification")
    print("=" * 60)

    train_ds = LeniaDataset(CFG["data"]["train_path"])
    val_ds   = LeniaDataset(CFG["data"]["val_path"])

    print(f"  train dataset len : {len(train_ds):,}  "
          f"(expected {1800 * 199:,} = 1800 trajs × 199 pairs)")
    print(f"  val   dataset len : {len(val_ds):,}  "
          f"(expected {200 * 199:,} = 200  trajs × 199 pairs)")

    assert len(train_ds) == 1800 * 199, \
        f"train dataset length mismatch: {len(train_ds)}"
    assert len(val_ds)   == 200  * 199, \
        f"val dataset length mismatch: {len(val_ds)}"

    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    ft, ft1 = next(iter(loader))

    print(f"\n  First batch:")
    print(f"    frame_t       shape : {tuple(ft.shape)}   (expected {(BATCH_SIZE, 1, 64, 64)})")
    print(f"    frame_t_plus_1 shape: {tuple(ft1.shape)}  (expected {(BATCH_SIZE, 1, 64, 64)})")
    print(f"    frame_t        range: [{ft.min():.4f}, {ft.max():.4f}]")
    print(f"    frame_t_plus_1 range: [{ft1.min():.4f}, {ft1.max():.4f}]")

    assert ft.shape  == (BATCH_SIZE, 1, 64, 64), f"frame_t shape: {ft.shape}"
    assert ft1.shape == (BATCH_SIZE, 1, 64, 64), f"frame_t+1 shape: {ft1.shape}"
    assert ft.min() >= 0.0 and ft.max() <= 1.0,  "frame_t out of [0,1]"
    assert ft1.min() >= 0.0 and ft1.max() <= 1.0, "frame_t+1 out of [0,1]"

    n, t = train_ds.index[-1]
    assert t == 198, f"Last index t={t}, expected 198 (not t=199)"
    print(f"\n  Boundary check (last index): traj={n}, t={t}  [OK – t=198, not 199]")

    import h5py
    idx_to_test = 0
    n0, t0 = train_ds.index[idx_to_test]
    ft_ds, ft1_ds = train_ds[idx_to_test]
    with h5py.File(CFG["data"]["train_path"], "r") as hf:
        ft_h5  = hf["frames"][n0, t0    ].astype(np.float32)
        ft1_h5 = hf["frames"][n0, t0 + 1].astype(np.float32)
    assert np.allclose(ft_ds.numpy().squeeze(), ft_h5,  atol=1e-6), "frame_t mismatch"
    assert np.allclose(ft1_ds.numpy().squeeze(), ft1_h5, atol=1e-6), "frame_t+1 mismatch"
    print(f"  Consecutive-frame content check: PASSED")

    print("\n  Step 2: PASSED")

def step3_forward_pass() -> None:
    print("\n" + "=" * 60)
    print("  STEP 3: Model forward-pass test")
    print("=" * 60)

    set_seed(SEED)
    model = PixelWorldModel(embed_dim=CFG["model"]["embed_dim"]).to(DEVICE)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PixelWorldModel  embed_dim={CFG['model']['embed_dim']}  "
          f"trainable params={num_params:,}")

    dummy = torch.rand(BATCH_SIZE, 1, 64, 64, device=DEVICE)
    with torch.no_grad():
        z_dummy   = model.encoder(dummy)
        out_dummy = model(dummy)

    print(f"\n  [dummy batch]")
    print(f"    encoder output shape : {tuple(z_dummy.shape)}  "
          f"(expected ({BATCH_SIZE}, {CFG['model']['embed_dim']}))")
    print(f"    model output shape   : {tuple(out_dummy.shape)}  "
          f"(expected ({BATCH_SIZE}, 1, 64, 64))")
    print(f"    output value range   : [{out_dummy.min():.4f}, {out_dummy.max():.4f}]  "
          f"(Sigmoid → should be in [0,1])")

    assert z_dummy.shape   == (BATCH_SIZE, CFG["model"]["embed_dim"])
    assert out_dummy.shape == (BATCH_SIZE, 1, 64, 64)
    assert out_dummy.min() >= 0.0 - 1e-5
    assert out_dummy.max() <= 1.0 + 1e-5

    from torch.utils.data import DataLoader
    ds      = LeniaDataset(CFG["data"]["val_path"])
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    ft, ft1 = next(iter(loader))
    ft = ft.to(DEVICE)

    with torch.no_grad():
        z_real  = model.encoder(ft)
        out_real = model(ft)

    print(f"\n  [real batch]")
    print(f"    encoder output shape : {tuple(z_real.shape)}")
    print(f"    model output shape   : {tuple(out_real.shape)}")
    print(f"    output value range   : [{out_real.min():.4f}, {out_real.max():.4f}]")
    nan_enc = torch.isnan(z_real).any().item()
    nan_out = torch.isnan(out_real).any().item()
    print(f"    NaN in encoder output: {nan_enc}")
    print(f"    NaN in model output  : {nan_out}")

    assert not nan_enc, "NaN in encoder output on real data"
    assert not nan_out, "NaN in model output on real data"

    print("\n  Step 3: PASSED")

def step4_mini_train() -> tuple[PixelWorldModel, list, list]:
    print("\n" + "=" * 60)
    print(f"  STEP 4: Mini training run  ({N_EPOCHS} epochs, batch={BATCH_SIZE})")
    print(f"          {N_TRAIN_TRAJS} train trajs / {N_VAL_TRAJS} val trajs (lazy loading)")
    print("=" * 60)

    set_seed(SEED)

    T_MINUS_1 = 199
    train_ds    = LeniaDataset(CFG["data"]["train_path"])
    val_ds      = LeniaDataset(CFG["data"]["val_path"])
    train_subset = Subset(train_ds, range(N_TRAIN_TRAJS * T_MINUS_1))
    val_subset   = Subset(val_ds,   range(N_VAL_TRAJS   * T_MINUS_1))

    print(f"  train pairs: {len(train_subset):,}   val pairs: {len(val_subset):,}")

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = PixelWorldModel(embed_dim=CFG["model"]["embed_dim"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: PixelWorldModel  trainable params={num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log_dir = str(OUT_DIR / "tensorboard")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
        setup_type="pixel",
        log_dir=log_dir,
        log_interval=200,
        checkpoint_interval=N_EPOCHS,   # only save at end
    )

    train_losses: list[float] = []
    val_losses:   list[float] = []

    print(f"\n  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Time':>8}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}")

    for epoch in range(N_EPOCHS):
        t0         = time.time()
        train_loss = trainer.train_epoch(epoch)
        val_loss   = trainer.validate(epoch)
        elapsed    = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  {epoch+1:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {elapsed:>7.1f}s")

        # NaN guard
        assert not np.isnan(train_loss), f"NaN train loss at epoch {epoch+1}"
        assert not np.isnan(val_loss),   f"NaN val loss at epoch {epoch+1}"

    trainer._save_checkpoint(N_EPOCHS, val_losses[-1])
    trainer.writer.close()

    epochs = list(range(1, N_EPOCHS + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, "o-", label="train loss",  color="steelblue")
    ax.plot(epochs, val_losses,   "s-", label="val loss",    color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Pixel World Model – mini training run ({N_EPOCHS} epochs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = OUT_DIR / "step4_loss_curves.png"
    fig.savefig(loss_path, dpi=110)
    plt.close()
    print(f"\n  Loss curves saved: {loss_path.relative_to(ROOT)}")
    print(f"\n  Final train loss : {train_losses[-1]:.6f}")
    print(f"  Final val   loss : {val_losses[-1]:.6f}")
    print("\n  Step 4: PASSED")

    return trainer.model, train_losses, val_losses

def step5_visualise_predictions(model: PixelWorldModel) -> None:
    print("\n" + "=" * 60)
    print("  STEP 5: Qualitative prediction visualisation")
    print("=" * 60)

    model.eval()
    ds = LeniaDataset(CFG["data"]["val_path"])

    # Pick 5 evenly-spaced indices so we see different trajectories
    n_samples = 5
    indices   = [i * (len(ds) // n_samples) for i in range(n_samples)]

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))

    for row, idx in enumerate(indices):
        ft, ft1 = ds[idx]                              # (1, H, W)
        ft_  = ft.unsqueeze(0).to(DEVICE)             # (1, 1, H, W)

        with torch.no_grad():
            pred = model(ft_).squeeze().cpu().numpy()  # (H, W)

        input_np = ft.squeeze().numpy()
        true_np  = ft1.squeeze().numpy()

        mse = float(F.mse_loss(torch.from_numpy(pred), torch.from_numpy(true_np)))

        for col, (img, title) in enumerate([
            (input_np, f"Input  frame_t"),
            (pred,     f"Predicted t+1\nMSE={mse:.4f}"),
            (true_np,  f"True frame_t+1"),
        ]):
            ax = axes[row, col]
            ax.imshow(img, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    fig.suptitle("Pixel World Model – next-frame predictions (val set)", fontsize=10)
    plt.tight_layout()
    out_path = OUT_DIR / "step5_predictions.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(ROOT)}")
    print("\n  Step 5: DONE")

def step6_rollout(model: PixelWorldModel, n_steps: int = 30) -> None:
    print("\n" + "=" * 60)
    print(f"  STEP 6: Multi-step rollout ({n_steps} steps)")
    print("=" * 60)

    model.eval()

    # Pick the first full trajectory from val set
    import h5py
    with h5py.File(CFG["data"]["val_path"], "r") as hf:
        traj = hf["frames"][0].astype(np.float32)    # (T, H, W)

    rollout_preds = []
    current_frame = torch.from_numpy(traj[0]).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)

    with torch.no_grad():
        for _ in range(n_steps):
            next_frame = model(current_frame)        # (1, 1, H, W)
            rollout_preds.append(next_frame.squeeze().cpu().numpy())
            current_frame = next_frame

    mse_per_step = []
    for step_idx, pred in enumerate(rollout_preds):
        true = traj[step_idx + 1]                    # t+1, t+2, ...
        mse  = float(np.mean((pred - true) ** 2))
        mse_per_step.append(mse)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, n_steps + 1), mse_per_step, "o-", color="tomato")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("MSE vs. true frame")
    ax.set_title(f"Rollout compounding error ({n_steps} steps, val traj 0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    mse_path = OUT_DIR / "step6_rollout_mse.png"
    fig.savefig(mse_path, dpi=110)
    plt.close()
    print(f"  MSE plot saved : {mse_path.relative_to(ROOT)}")

    vis_steps = [0, 4, 9, 14, 19, 29] if n_steps >= 30 else list(range(min(6, n_steps)))
    vis_steps = [s for s in vis_steps if s < n_steps]

    fig, axes = plt.subplots(2, len(vis_steps), figsize=(3 * len(vis_steps), 6))
    for col, s in enumerate(vis_steps):
        for row, (img, row_label) in enumerate([
            (rollout_preds[s],   f"Predicted step {s+1}"),
            (traj[s + 1],        f"True step {s+1}"),
        ]):
            ax = axes[row, col]
            ax.imshow(img, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(row_label, fontsize=8)
            ax.axis("off")

    fig.suptitle("Multi-step rollout: predicted (top) vs. true (bottom)", fontsize=10)
    plt.tight_layout()
    rollout_path = OUT_DIR / "step6_rollout_frames.png"
    fig.savefig(rollout_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Rollout frames : {rollout_path.relative_to(ROOT)}")

    print(f"\n  MSE at step  1 : {mse_per_step[0]:.6f}")
    print(f"  MSE at step {n_steps // 2:>2} : {mse_per_step[n_steps // 2 - 1]:.6f}")
    print(f"  MSE at step {n_steps:>2} : {mse_per_step[-1]:.6f}")
    trend_ok = mse_per_step[-1] > mse_per_step[0]
    print(f"  Error increases over rollout: {trend_ok}  (expected True)")
    if not trend_ok:
        print("  WARNING: error did not grow – model may be predicting a constant")

    print("\n  Step 6: DONE")
    return mse_per_step

def main() -> None:
    print("=" * 60)
    print("  Lenia Pipeline Smoke Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  device: {DEVICE}")
    print("=" * 60)

    step2_dataloader_test()
    step3_forward_pass()
    model, train_losses, val_losses = step4_mini_train()
    step5_visualise_predictions(model)
    mse_per_step = step6_rollout(model, n_steps=30)

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE")
    print("=" * 60)
    print(f"  Results in: {OUT_DIR.relative_to(ROOT)}/")
    print(f"    step4_loss_curves.png")
    print(f"    step5_predictions.png")
    print(f"    step6_rollout_frames.png")
    print(f"    step6_rollout_mse.png")
    print(f"\n  Final train loss : {train_losses[-1]:.6f}")
    print(f"  Final val   loss : {val_losses[-1]:.6f}")
    print(f"  Rollout MSE step 1→30: "
          f"{mse_per_step[0]:.6f} → {mse_per_step[-1]:.6f}")

if __name__ == "__main__":
    main()
