"""Generate result plots from saved training runs.

Produces two figures saved to experiments/:
  - results_loss_curves.png   — train/val loss for pixel and JEPA over all epochs
  - results_predictions.png   — pixel model: input | predicted | ground truth samples

Usage:
    python scripts/visualize_results.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import PixelWorldModel
from scripts.train_single import LeniaDataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent

PIXEL_EVENTS = ROOT / "experiments/cluster/pixel_22004015/experiments/pixel_20260602_220606"
JEPA_EVENTS  = ROOT / "experiments/cluster/jepa_22008994/experiments/jepa_20260602_234512"
PIXEL_CKPT   = ROOT / "experiments/cluster/pixel_22004015/experiments/pixel_20260602_220606/checkpoint_epoch_0100.pt"
VAL_PATH     = ROOT / "data/lenia_val_chunked.h5"
OUT_DIR      = ROOT / "experiments/plots"


def load_scalars(log_dir: Path, tags: list[str]) -> dict:
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    out = {}
    available = ea.Tags().get("scalars", [])
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps  = [e.step + 1 for e in events]
            values = [e.value   for e in events]
            out[tag] = (steps, values)
    return out


def plot_loss_curves() -> None:
    pixel = load_scalars(PIXEL_EVENTS, ["train/loss_epoch", "val/loss_epoch"])
    jepa  = load_scalars(JEPA_EVENTS,  ["train/loss_epoch", "val/loss_epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Results — Lenia World Models (Clara Cluster, 100 epochs)", fontsize=13)

    # Pixel
    ax = axes[0]
    if "train/loss_epoch" in pixel:
        ax.plot(*pixel["train/loss_epoch"], label="train", color="#2196F3")
    if "val/loss_epoch" in pixel:
        ax.plot(*pixel["val/loss_epoch"],   label="val",   color="#FF5722", alpha=0.8)
    ax.set_title("Setup A — Pixel Prediction")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # JEPA
    ax = axes[1]
    if "train/loss_epoch" in jepa:
        ax.plot(*jepa["train/loss_epoch"], label="train", color="#2196F3")
    if "val/loss_epoch" in jepa:
        ax.plot(*jepa["val/loss_epoch"],   label="val",   color="#FF5722", alpha=0.8)
    ax.set_title("Setup B — JEPA (latent prediction + VICReg)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE + VICReg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "results_loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_pixel_predictions(n_samples: int = 5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(PIXEL_CKPT, map_location=device, weights_only=False)
    model = PixelWorldModel(embed_dim=256).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds     = LeniaDataset(str(VAL_PATH))
    loader = DataLoader(ds, batch_size=n_samples, shuffle=True)
    frame_t, frame_t1 = next(iter(loader))

    with torch.no_grad():
        pred = model(frame_t.to(device)).cpu()

    frame_t  = frame_t.numpy()
    frame_t1 = frame_t1.numpy()
    pred     = pred.numpy()

    fig = plt.figure(figsize=(12, 2.8 * n_samples))
    fig.suptitle("Setup A — Pixel Prediction: Input | Predicted t+1 | True t+1", fontsize=12)
    gs = gridspec.GridSpec(n_samples, 3, figure=fig, hspace=0.05, wspace=0.05)

    for i in range(n_samples):
        mse = float(np.mean((pred[i, 0] - frame_t1[i, 0]) ** 2))

        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(frame_t[i, 0], cmap="viridis", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title("Input  (frame t)", fontsize=9)

        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(pred[i, 0], cmap="viridis", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_ylabel(f"MSE={mse:.4f}", fontsize=7, rotation=0, labelpad=50, va="center")
        if i == 0:
            ax.set_title("Predicted (frame t+1)", fontsize=9)

        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(frame_t1[i, 0], cmap="viridis", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title("Ground Truth (frame t+1)", fontsize=9)

    out = OUT_DIR / "results_predictions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    print("Generating loss curves...")
    plot_loss_curves()

    if VAL_PATH.exists() and PIXEL_CKPT.exists():
        print("Generating pixel predictions...")
        plot_pixel_predictions()
    else:
        print("Skipping predictions — val data or checkpoint not found locally.")
