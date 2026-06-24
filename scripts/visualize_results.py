"""Generate result plots and animations from saved v2 training runs.

Produces files saved to experiments/plots/:
  Loss curves:
    results_loss_curves_pixel.png      — Pixel-CNN vs Pixel-ViT (shared y-axis)
    results_loss_curves_jepa.png       — JEPA training + Decoder training (pipeline view)

  Pixel predictions (input | predicted t+1 | ground truth):
    results_predictions_pixel.png      — Pixel-CNN
    results_predictions_vit.png        — Pixel-ViT
    results_predictions_jepa.png       — JEPA decoded (encode → predict → decode)

  Teacher-forced GIFs (real frame_t → model → predicted t+1 vs ground truth):
    trajectory_comparison_cnn.gif
    trajectory_comparison_jepa.gif

  Autoregressive rollout GIFs (only frame_0 given, model feeds own output):
    trajectory_autoregressive_cnn.gif
    trajectory_autoregressive_vit.gif
    trajectory_autoregressive_jepa.gif

Usage:
    python scripts/visualize_results.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decoder import SpatialLeniaDecoder
from models.world_model import JEPAWorldModel, PixelWorldModel
from scripts.train_single import LeniaDataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent

# v2 experiment dirs (TensorBoard events + checkpoints)
PIXEL_CNN_DIR  = ROOT / "experiments/cluster/pixel_v2_22782974/experiments/pixel_20260622_224456"
PIXEL_VIT_DIR  = ROOT / "experiments/cluster/vit_v2_22782976/experiments/pixel_20260622_224445"
JEPA_DIR       = ROOT / "experiments/cluster/jepa_v2_22782975/experiments/jepa_20260622_224445"
DECODER_DIR    = ROOT / "experiments/cluster/decoder_v2_22917956/experiments/decoder_20260623_112743"

# Named checkpoints
PIXEL_CNN_CKPT = ROOT / "checkpoints/pixel_cnn_v2.pt"
PIXEL_VIT_CKPT = ROOT / "checkpoints/pixel_vit_v2.pt"
JEPA_CKPT      = ROOT / "checkpoints/jepa_cnn_v2.pt"
DECODER_CKPT   = ROOT / "checkpoints/decoder_jepa_v2.pt"

VAL_PATH = ROOT / "data/v2_lenia_val_chunked.h5"
OUT_DIR  = ROOT / "experiments/plots"


# helpers
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


def _load_pixel_model(ckpt_path: Path, embed_dim: int, encoder_type: str) -> PixelWorldModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PixelWorldModel(embed_dim=embed_dim, encoder_type=encoder_type).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_jepa_pipeline(
    jepa_ckpt: Path, decoder_ckpt: Path, embed_dim: int = 256
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Return (frozen encoder, frozen predictor, frozen spatial decoder).

    The decoder operates on pre-GAP spatial features (B,256,4,4), not the
    flat embedding — so predictions use: encoder(x, return_spatial=True) → decoder.
    For decoded predictions we use: encoder(x) → predictor → [not decodable spatially].
    Reconstruction uses: encoder(x, return_spatial=True) → decoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jepa_state = torch.load(jepa_ckpt, map_location=device, weights_only=False)["model_state_dict"]
    predictor_hidden_dim = jepa_state["predictor.net.0.weight"].shape[0]
    jepa = JEPAWorldModel(embed_dim=embed_dim, predictor_hidden_dim=predictor_hidden_dim)
    jepa.load_state_dict(jepa_state)
    encoder   = jepa.online_encoder.to(device).eval()
    predictor = jepa.predictor.to(device).eval()

    dec_state = torch.load(decoder_ckpt, map_location=device, weights_only=False)["model_state_dict"]
    decoder = SpatialLeniaDecoder().to(device)
    decoder.load_state_dict(dec_state)
    decoder.eval()

    for m in (encoder, predictor, decoder):
        for p in m.parameters():
            p.requires_grad = False

    return encoder, predictor, decoder


def _load_traj(traj_idx: int) -> np.ndarray:
    import h5py
    with h5py.File(str(VAL_PATH), "r") as f:
        traj = f["frames"][traj_idx].astype(np.float32)
    if traj.max() > 1.0:
        traj /= traj.max()
    return traj


def _ymax_of(*scalar_dicts) -> float | None:
    vals = [v for d in scalar_dicts for tag in ("train/loss_epoch", "val/loss_epoch")
            for _, vs in [d.get(tag, ([], []))] for v in vs]
    return max(vals) * 1.05 if vals else None


# loss curves 

def plot_loss_curves_pixel() -> None:
    pixel = load_scalars(PIXEL_CNN_DIR, ["train/loss_epoch", "val/loss_epoch"])
    vit   = load_scalars(PIXEL_VIT_DIR, ["train/loss_epoch", "val/loss_epoch"])

    ymax = _ymax_of(pixel, vit)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Pixel-space World Models — Training Loss (v2 data)", fontsize=12)

    for ax, scalars, title in [
        (axes[0], pixel, "Pixel-CNN  (MSE)"),
        (axes[1], vit,   "Pixel-ViT  (MSE)"),
    ]:
        if "train/loss_epoch" in scalars:
            ax.plot(*scalars["train/loss_epoch"], label="train", color="#2196F3")
        if "val/loss_epoch" in scalars:
            ax.plot(*scalars["val/loss_epoch"],   label="val",   color="#FF5722", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        if ymax:
            ax.set_ylim(bottom=0, top=ymax)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "results_loss_curves_pixel.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_loss_curves_jepa() -> None:
    jepa    = load_scalars(JEPA_DIR,    ["train/loss_epoch", "val/loss_epoch"])
    decoder = load_scalars(DECODER_DIR, ["train/loss_epoch", "val/loss_epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("JEPA Pipeline — Training Loss (v2 data)", fontsize=12)

    for ax, scalars, title, ylabel in [
        (axes[0], jepa,    "Stage 1 — JEPA-CNN  (MSE + VICReg)", "JEPA Loss"),
        (axes[1], decoder, "Stage 2 — Decoder  (reconstruction MSE)", "MSE Loss"),
    ]:
        if "train/loss_epoch" in scalars:
            ax.plot(*scalars["train/loss_epoch"], label="train", color="#2196F3")
        if "val/loss_epoch" in scalars:
            ax.plot(*scalars["val/loss_epoch"],   label="val",   color="#FF5722", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "results_loss_curves_jepa.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# pixel predictions

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

    _save_predictions_grid(
        frame_t.numpy(), pred.numpy(), frame_t1.numpy(),
        title=f"{title}: Input | Predicted t+1 | True t+1",
        out_name=out_name,
    )


def plot_jepa_predictions(n_samples: int = 5) -> None:
    """Show JEPA encoder reconstruction quality: decoder(encoder_spatial(frame_t)) vs frame_t.

    The decoder uses pre-GAP spatial features, so this shows how well the
    encoder's convolutional features (before pooling) preserve visual structure.
    Note: decoded *predictions* (via predictor) are not shown here because the
    predictor operates on the flat GAP embedding which has no spatial info.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, predictor, decoder = _load_jepa_pipeline(JEPA_CKPT, DECODER_CKPT)

    ds     = LeniaDataset(str(VAL_PATH))
    loader = DataLoader(ds, batch_size=n_samples, shuffle=True)
    frame_t, frame_t1 = next(iter(loader))

    with torch.no_grad():
        spatial = encoder(frame_t.to(device), return_spatial=True)
        recon   = decoder(spatial).cpu()

    _save_predictions_grid(
        frame_t.numpy(), recon.numpy(), frame_t1.numpy(),
        title="JEPA-CNN encoder reconstruction: Input | decoder(encoder_spatial(t)) | True t+1",
        out_name="results_predictions_jepa.png",
    )


def _save_predictions_grid(
    frame_t: np.ndarray,
    pred: np.ndarray,
    frame_t1: np.ndarray,
    title: str,
    out_name: str,
) -> None:
    n = len(frame_t)
    fig = plt.figure(figsize=(12, 2.8 * n))
    fig.suptitle(title, fontsize=11)
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.05, wspace=0.05)

    for i in range(n):
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


# GIFs

def make_teacher_forced_gif(
    ckpt_path: Path,
    embed_dim: int,
    encoder_type: str,
    out_name: str,
    title: str,
    traj_idx: int = 5,
    fps: int = 10,
) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_pixel_model(ckpt_path, embed_dim, encoder_type)
    traj  = _load_traj(traj_idx)
    T = traj.shape[0]

    preds = []
    with torch.no_grad():
        for t in range(T - 1):
            frame = torch.from_numpy(traj[t]).unsqueeze(0).unsqueeze(0).to(device)
            preds.append(model(frame).squeeze().cpu().numpy())

    _save_gif(preds, traj, title, f"{title} — Teacher-Forced", out_name, fps)


def make_autoregressive_gif(
    ckpt_path: Path,
    embed_dim: int,
    encoder_type: str,
    out_name: str,
    title: str,
    traj_idx: int = 5,
    fps: int = 10,
) -> None:
    """Only frame_0 is given; every subsequent input is the model's own previous output."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_pixel_model(ckpt_path, embed_dim, encoder_type)
    traj  = _load_traj(traj_idx)
    T = traj.shape[0]

    preds = []
    with torch.no_grad():
        frame = torch.from_numpy(traj[0]).unsqueeze(0).unsqueeze(0).to(device)
        for _ in range(T - 1):
            frame = model(frame)
            preds.append(frame.squeeze().cpu().numpy())

    _save_gif(preds, traj, title, f"{title} — Autoregressive Rollout", out_name, fps)


def make_jepa_teacher_forced_gif(traj_idx: int = 5, fps: int = 10) -> None:
    """JEPA teacher-forced reconstruction: decoder(encoder_spatial(real frame_t)) vs ground truth.

    Shows frame-by-frame reconstruction quality of the spatial decoder,
    not a prediction — the encoder sees the real frame at every step.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, predictor, decoder = _load_jepa_pipeline(JEPA_CKPT, DECODER_CKPT)
    traj = _load_traj(traj_idx)
    T = traj.shape[0]

    preds = []
    with torch.no_grad():
        for t in range(T - 1):
            frame   = torch.from_numpy(traj[t]).unsqueeze(0).unsqueeze(0).to(device)
            spatial = encoder(frame, return_spatial=True)
            preds.append(decoder(spatial).squeeze().cpu().numpy())

    _save_gif(preds, traj, "JEPA-CNN", "JEPA-CNN — Reconstruction (spatial decoder)",
              "trajectory_comparison_jepa.gif", fps)


def make_jepa_autoregressive_gif(traj_idx: int = 5, fps: int = 10) -> None:
    """JEPA autoregressive rollout: spatial decoder output fed back each step.

    frame_0 → encode_spatial → decode → frame_1_hat → encode_spatial → decode → ...
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, predictor, decoder = _load_jepa_pipeline(JEPA_CKPT, DECODER_CKPT)
    traj = _load_traj(traj_idx)
    T = traj.shape[0]

    preds = []
    with torch.no_grad():
        frame = torch.from_numpy(traj[0]).unsqueeze(0).unsqueeze(0).to(device)
        for _ in range(T - 1):
            spatial = encoder(frame, return_spatial=True)
            frame   = decoder(spatial)
            preds.append(frame.squeeze().cpu().numpy())

    _save_gif(preds, traj, "JEPA-CNN", "JEPA-CNN — Autoregressive (spatial decoder)",
              "trajectory_autoregressive_jepa.gif", fps)


def _save_gif(
    preds: list,
    traj: np.ndarray,
    model_name: str,
    suptitle: str,
    out_name: str,
    fps: int,
) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.suptitle(suptitle, fontsize=11)
    axes[0].set_title("Model output", fontsize=9)
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

    ani = FuncAnimation(fig, update, frames=len(preds), interval=1000 // fps, blit=False)
    out = OUT_DIR / out_name
    ani.save(str(out), writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loss curves ===")
    print("Pixel models (CNN vs ViT)...")
    plot_loss_curves_pixel()
    print("JEPA pipeline (JEPA training + Decoder training)...")
    plot_loss_curves_jepa()

    if not VAL_PATH.exists():
        print("Skipping all model visuals — v2 val data not found locally.")
    else:
        print("\n=== Pixel-CNN ===")
        if PIXEL_CNN_CKPT.exists():
            plot_pixel_predictions(PIXEL_CNN_CKPT, 128, "cnn",
                                   "results_predictions_pixel.png", "Pixel-CNN")
            make_teacher_forced_gif(PIXEL_CNN_CKPT, 128, "cnn",
                                    "trajectory_comparison_cnn.gif", "Pixel-CNN")
            make_autoregressive_gif(PIXEL_CNN_CKPT, 128, "cnn",
                                    "trajectory_autoregressive_cnn.gif", "Pixel-CNN")

        print("\n=== Pixel-ViT ===")
        if PIXEL_VIT_CKPT.exists():
            plot_pixel_predictions(PIXEL_VIT_CKPT, 256, "vit",
                                   "results_predictions_vit.png", "Pixel-ViT")
            make_autoregressive_gif(PIXEL_VIT_CKPT, 256, "vit",
                                    "trajectory_autoregressive_vit.gif", "Pixel-ViT")

        print("\n=== JEPA-CNN (decoded) ===")
        if JEPA_CKPT.exists() and DECODER_CKPT.exists():
            plot_jepa_predictions()
            make_jepa_teacher_forced_gif()
            make_jepa_autoregressive_gif()
        else:
            print("Skipping JEPA visuals — checkpoint(s) not found.")
