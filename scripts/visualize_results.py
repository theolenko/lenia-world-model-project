"""Generate result plots and animations from saved v2 training runs.

Produces files saved to experiments/plots/:
  - results_loss_curves.png         — train/val loss for all 3 models
  - results_predictions_pixel.png   — Pixel-CNN: input | predicted | ground truth
  - results_predictions_vit.png     — Pixel-ViT: input | predicted | ground truth
  - trajectory_comparison.gif       — Pixel-CNN teacher-forced vs ground truth
  - trajectory_autoregressive.gif   — Pixel-CNN autoregressive rollout vs ground truth

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

# v2 experiment paths
PIXEL_CNN_DIR  = ROOT / "experiments/cluster/pixel_v2_22782974/experiments/pixel_20260622_224456"
PIXEL_VIT_DIR  = ROOT / "experiments/cluster/vit_v2_22782976/experiments/pixel_20260622_224445"
JEPA_DIR       = ROOT / "experiments/cluster/jepa_v2_22782975/experiments/jepa_20260622_224445"

PIXEL_CNN_CKPT = PIXEL_CNN_DIR / "checkpoint_best.pt"
PIXEL_VIT_CKPT = PIXEL_VIT_DIR / "checkpoint_best.pt"

VAL_PATH = ROOT / "data/v2_lenia_val_chunked.h5"
OUT_DIR  = ROOT / "experiments/plots"


def load_scalars(log_dir: Path, tags: list[str]) -> dict:
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    out = {}
    available = ea.Tags().get("scalars", [])
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            out[tag] = ([e.step + 1 for e in events], [e.value for e in events])
    return out


def plot_loss_curves() -> None:
    pixel = load_scalars(PIXEL_CNN_DIR, ["train/loss_epoch", "val/loss_epoch"])
    vit   = load_scalars(PIXEL_VIT_DIR, ["train/loss_epoch", "val/loss_epoch"])
    jepa  = load_scalars(JEPA_DIR,      ["train/loss_epoch", "val/loss_epoch"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Training Results — Lenia World Models v2 (100 epochs, CosineAnnealingLR)", fontsize=13)

    specs = [
        (axes[0], pixel, "Pixel-CNN  (val MSE)"),
        (axes[1], vit,   "Pixel-ViT  (val MSE)"),
        (axes[2], jepa,  "JEPA-CNN  (val loss = MSE + VICReg)"),
    ]
    for ax, scalars, title in specs:
        if "train/loss_epoch" in scalars:
            ax.plot(*scalars["train/loss_epoch"], label="train", color="#2196F3")
        if "val/loss_epoch" in scalars:
            ax.plot(*scalars["val/loss_epoch"],   label="val",   color="#FF5722", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "results_loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def _load_pixel_model(ckpt_path: Path, embed_dim: int, encoder_type: str) -> PixelWorldModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PixelWorldModel(embed_dim=embed_dim, encoder_type=encoder_type).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def plot_pixel_predictions(
    ckpt_path: Path,
    embed_dim: int,
    encoder_type: str,
    out_name: str,
    title: str,
    n_samples: int = 5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_pixel_model(ckpt_path, embed_dim, encoder_type)

    ds     = LeniaDataset(str(VAL_PATH))
    loader = DataLoader(ds, batch_size=n_samples, shuffle=True)
    frame_t, frame_t1 = next(iter(loader))

    with torch.no_grad():
        pred = model(frame_t.to(device)).cpu()

    frame_t  = frame_t.numpy()
    frame_t1 = frame_t1.numpy()
    pred     = pred.numpy()

    fig = plt.figure(figsize=(12, 2.8 * n_samples))
    fig.suptitle(f"{title}: Input | Predicted t+1 | True t+1", fontsize=12)
    gs = gridspec.GridSpec(n_samples, 3, figure=fig, hspace=0.05, wspace=0.05)

    for i in range(n_samples):
        mse = float(np.mean((pred[i, 0] - frame_t1[i, 0]) ** 2))

        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(frame_t[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title("Input  (frame t)", fontsize=9)

        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(pred[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_ylabel(f"MSE={mse:.4f}", fontsize=7, rotation=0, labelpad=50, va="center")
        if i == 0:
            ax.set_title("Predicted (frame t+1)", fontsize=9)

        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(frame_t1[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title("Ground Truth (frame t+1)", fontsize=9)

    out = OUT_DIR / out_name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def make_trajectory_gif(traj_idx: int = 5, fps: int = 10) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_pixel_model(PIXEL_CNN_CKPT, embed_dim=128, encoder_type="cnn")

    import h5py
    with h5py.File(str(VAL_PATH), "r") as f:
        traj = f["frames"][traj_idx].astype(np.float32)
    if traj.max() > 1.0:
        traj /= traj.max()

    T = traj.shape[0]
    preds = []
    with torch.no_grad():
        for t in range(T - 1):
            frame = torch.from_numpy(traj[t]).unsqueeze(0).unsqueeze(0).to(device)
            preds.append(model(frame).squeeze().cpu().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.suptitle("Pixel-CNN — Teacher-Forced Prediction", fontsize=11)
    axes[0].set_title("Predicted t+1", fontsize=9)
    axes[1].set_title("Ground Truth t+1", fontsize=9)
    for ax in axes:
        ax.axis("off")

    im_pred  = axes[0].imshow(preds[0],   cmap="gray", vmin=0, vmax=1)
    im_true  = axes[1].imshow(traj[1],    cmap="gray", vmin=0, vmax=1)
    step_txt = fig.text(0.5, 0.02, "t=0", ha="center", fontsize=9)

    def update(t):
        im_pred.set_data(preds[t])
        im_true.set_data(traj[t + 1])
        step_txt.set_text(f"t={t}")
        return im_pred, im_true, step_txt

    ani = FuncAnimation(fig, update, frames=T - 1, interval=1000 // fps, blit=False)
    out = OUT_DIR / "trajectory_comparison.gif"
    ani.save(str(out), writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Saved: {out}")


def make_autoregressive_gif(traj_idx: int = 5, fps: int = 10) -> None:
    """Autoregressive rollout: model feeds its own predictions back as input.

    Only the first real frame is given to the model — every subsequent input
    is its own previous output. Divergence from ground truth shows how well
    it has learned Lenia dynamics rather than just per-step denoising.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_pixel_model(PIXEL_CNN_CKPT, embed_dim=128, encoder_type="cnn")

    import h5py
    with h5py.File(str(VAL_PATH), "r") as f:
        traj = f["frames"][traj_idx].astype(np.float32)
    if traj.max() > 1.0:
        traj /= traj.max()

    T = traj.shape[0]
    preds = []
    with torch.no_grad():
        frame = torch.from_numpy(traj[0]).unsqueeze(0).unsqueeze(0).to(device)
        for _ in range(T - 1):
            frame = model(frame)
            preds.append(frame.squeeze().cpu().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.suptitle("Pixel-CNN — Autoregressive Rollout", fontsize=11)
    axes[0].set_title("Model rollout", fontsize=9)
    axes[1].set_title("Ground truth", fontsize=9)
    for ax in axes:
        ax.axis("off")

    im_pred  = axes[0].imshow(preds[0],  cmap="gray", vmin=0, vmax=1)
    im_true  = axes[1].imshow(traj[1],   cmap="gray", vmin=0, vmax=1)
    step_txt = fig.text(0.5, 0.02, "t=1", ha="center", fontsize=9)

    def update(t):
        im_pred.set_data(preds[t])
        im_true.set_data(traj[t + 1])
        step_txt.set_text(f"t={t + 1}")
        return im_pred, im_true, step_txt

    ani = FuncAnimation(fig, update, frames=T - 1, interval=1000 // fps, blit=False)
    out = OUT_DIR / "trajectory_autoregressive.gif"
    ani.save(str(out), writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating loss curves (all 3 models)...")
    plot_loss_curves()

    if VAL_PATH.exists():
        if PIXEL_CNN_CKPT.exists():
            print("Generating Pixel-CNN predictions...")
            plot_pixel_predictions(
                PIXEL_CNN_CKPT, embed_dim=128, encoder_type="cnn",
                out_name="results_predictions_pixel.png",
                title="Pixel-CNN",
            )
            print("Generating teacher-forced trajectory GIF (Pixel-CNN)...")
            make_trajectory_gif()
            print("Generating autoregressive rollout GIF (Pixel-CNN)...")
            make_autoregressive_gif()
        else:
            print("Skipping Pixel-CNN visuals — checkpoint not found.")

        if PIXEL_VIT_CKPT.exists():
            print("Generating Pixel-ViT predictions...")
            plot_pixel_predictions(
                PIXEL_VIT_CKPT, embed_dim=256, encoder_type="vit",
                out_name="results_predictions_vit.png",
                title="Pixel-ViT",
            )
        else:
            print("Skipping Pixel-ViT visuals — checkpoint not found.")
    else:
        print("Skipping pixel visuals — v2 val data not found locally.")
