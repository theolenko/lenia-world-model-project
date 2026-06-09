"""Evaluate a trained Lenia world model on the validation set.

Computes MSE, PSNR, and SSIM for each batch and reports mean ± std.
Saves a visual grid of N sample predictions to experiments/eval/.

Usage:
    # Evaluate the best checkpoint of the pixel run
    python scripts/evaluate.py --checkpoint experiments/pixel_20260512_231545/checkpoint_epoch_0100.pt --setup pixel

    # Specify a different data split or number of samples to visualise
    python scripts/evaluate.py --checkpoint path/to/checkpoint_best.pt --setup pixel --data data/lenia_val_chunked.h5 --n-samples 8
"""
import argparse
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import JEPAWorldModel, PixelWorldModel
from scripts.train_single import LeniaDataset


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error, averaged over all pixels in the batch."""
    return F.mse_loss(pred, target)


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio (dB).  Higher = better.

    PSNR = 10 * log10(max_val^2 / MSE)
    Frames are assumed to lie in [0, max_val].
    """
    mse_val = F.mse_loss(pred, target)
    return 10.0 * torch.log10(torch.tensor(max_val ** 2) / (mse_val + 1e-10))


def ssim_batch(pred: torch.Tensor, target: torch.Tensor,
               window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Structural Similarity Index (SSIM), averaged over the batch.

    Ranges from -1 to 1; higher = more similar.  A value near 1 means the
    predicted frame looks structurally identical to the ground truth.

    Uses a Gaussian-weighted sliding window (same as the original paper).

    Args:
        pred:   Predicted frames, shape (B, 1, H, W), values in [0, 1].
        target: Ground-truth frames, same shape.
        window_size: Size of the Gaussian kernel. Default: 11.
        sigma:  Standard deviation of the Gaussian. Default: 1.5.

    Returns:
        Scalar tensor with the mean SSIM over the batch.
    """
    # Build a 1-D Gaussian, then make it 2-D via outer product
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss[:, None] * gauss[None, :]          # (W, W)
    kernel = kernel_2d[None, None].to(pred.device)       # (1, 1, W, W)

    pad = window_size // 2

    mu1 = F.conv2d(pred,   kernel, padding=pad)
    mu2 = F.conv2d(target, kernel, padding=pad)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad) - mu2_sq
    sigma12   = F.conv2d(pred   * target, kernel, padding=pad) - mu1_mu2

    # SSIM constants (Lk = 1 for normalised images)
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: torch.device,
             setup_type: str) -> dict:
    """Run the model over the full dataloader and collect per-batch metrics.

    Args:
        model:      Trained PixelWorldModel or JEPAWorldModel.
        loader:     DataLoader yielding (frame_t, frame_t_plus_1) batches.
        device:     Torch device.
        setup_type: "pixel" or "jepa".

    Returns:
        Dict with keys "mse", "psnr", "ssim", each containing a list of
        per-batch scalar values, plus "mean" and "std" summaries.
    """
    model.eval()
    results = {"mse": [], "psnr": [], "ssim": []}

    for frame_t, frame_t1 in loader:
        frame_t  = frame_t.to(device)
        frame_t1 = frame_t1.to(device)

        if setup_type == "pixel":
            pred = model(frame_t)
        else:
            # JEPA does not produce pixel predictions directly.
            # We measure how well the online encoder can predict the next
            # frame's embedding and report embedding-space MSE only.
            z_pred, z_target, _ = model(frame_t, frame_t1)
            mse_val = F.mse_loss(z_pred, z_target).item()
            results["mse"].append(mse_val)
            # PSNR and SSIM are not meaningful in latent space → skip
            continue

        results["mse"].append(mse(pred, frame_t1).item())
        results["psnr"].append(psnr(pred, frame_t1).item())
        results["ssim"].append(ssim_batch(pred, frame_t1).item())

    for key in list(results.keys()):
        vals = results[key]
        if vals:
            results[f"{key}_mean"] = float(np.mean(vals))
            results[f"{key}_std"]  = float(np.std(vals))

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_prediction_grid(model: torch.nn.Module,
                         loader: DataLoader,
                         device: torch.device,
                         out_path: Path,
                         n_samples: int = 6) -> None:
    """Save a figure with n_samples rows: Input | Predicted | Ground Truth.

    Each row also shows the per-sample MSE.

    Args:
        model:     Trained PixelWorldModel.
        loader:    DataLoader (shuffled so we get varied samples).
        device:    Torch device.
        out_path:  Where to save the PNG.
        n_samples: Number of example rows in the figure.
    """
    model.eval()

    # Grab one batch that is large enough
    frame_t, frame_t1 = next(iter(loader))
    frame_t  = frame_t[:n_samples].to(device)
    frame_t1 = frame_t1[:n_samples].to(device)

    with torch.no_grad():
        pred = model(frame_t)

    # Move to CPU numpy for plotting
    inp  = frame_t.cpu().numpy()
    pr   = pred.cpu().numpy()
    gt   = frame_t1.cpu().numpy()

    fig = plt.figure(figsize=(10, 2.6 * n_samples))
    fig.suptitle("Evaluation: Input  |  Predicted t+1  |  Ground Truth t+1",
                 fontsize=12, y=1.01)
    gs = gridspec.GridSpec(n_samples, 3, hspace=0.05, wspace=0.05)

    for i in range(n_samples):
        sample_mse = float(np.mean((pr[i, 0] - gt[i, 0]) ** 2))

        for col, (img, title) in enumerate([
            (inp[i, 0], "Input (t)"),
            (pr[i,  0], "Predicted (t+1)"),
            (gt[i,  0], "Ground Truth (t+1)"),
        ]):
            ax = fig.add_subplot(gs[i, col])
            ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=9)
            if col == 1:
                ax.set_ylabel(f"MSE={sample_mse:.4f}", fontsize=7,
                              rotation=0, labelpad=55, va="center")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction grid → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Lenia world model.")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pt checkpoint file.",
    )
    parser.add_argument(
        "--setup", required=True, choices=["pixel", "jepa"],
        help="Model type: 'pixel' or 'jepa'.",
    )
    parser.add_argument(
        "--data", default="data/lenia_val_chunked.h5",
        help="Path to the HDF5 data file to evaluate on. Default: val set.",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=128,
        help="Embedding dimensionality (must match the checkpoint). Default: 128.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for evaluation. Default: 64.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=6,
        help="Number of sample rows in the prediction grid (pixel only). Default: 6.",
    )
    parser.add_argument(
        "--out-dir", default="experiments/eval",
        help="Directory to save outputs. Default: experiments/eval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch     = ckpt.get("epoch", "?")
    ckpt_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded checkpoint  epoch={epoch}  saved_val_loss={ckpt_loss:.6f}")

    # --- Build model ---
    if args.setup == "pixel":
        model = PixelWorldModel(embed_dim=args.embed_dim)
    else:
        model = JEPAWorldModel(embed_dim=args.embed_dim)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # --- DataLoader ---
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        sys.exit(1)

    dataset = LeniaDataset(str(data_path))
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset):,} frame pairs  ({data_path.name})")

    # --- Run evaluation ---
    print("\nRunning evaluation …")
    results = evaluate(model, loader, device, args.setup)

    print("\n── Results ──────────────────────────────")
    if args.setup == "pixel":
        print(f"  MSE   {results['mse_mean']:.6f}  ±  {results['mse_std']:.6f}")
        print(f"  PSNR  {results['psnr_mean']:.2f} dB  ±  {results['psnr_std']:.2f} dB")
        print(f"  SSIM  {results['ssim_mean']:.4f}  ±  {results['ssim_std']:.4f}")
    else:
        print(f"  Embedding MSE  {results['mse_mean']:.6f}  ±  {results['mse_std']:.6f}")
    print("─────────────────────────────────────────\n")

    # --- Save outputs ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = ckpt_path.stem  # e.g. "checkpoint_epoch_0100"

    if args.setup == "pixel":
        # Prediction grid (needs a loader with enough samples per batch)
        vis_loader = DataLoader(dataset, batch_size=max(args.n_samples, 8),
                                shuffle=True, num_workers=0)
        grid_path = out_dir / f"{stem}_predictions.png"
        save_prediction_grid(model, vis_loader, device, grid_path, args.n_samples)

    # Save numeric results as a simple text file
    txt_path = out_dir / f"{stem}_metrics.txt"
    with open(txt_path, "w") as f:
        f.write(f"checkpoint : {ckpt_path}\n")
        f.write(f"epoch      : {epoch}\n")
        f.write(f"data       : {data_path}\n")
        f.write(f"setup      : {args.setup}\n\n")
        if args.setup == "pixel":
            f.write(f"MSE   {results['mse_mean']:.6f}  ±  {results['mse_std']:.6f}\n")
            f.write(f"PSNR  {results['psnr_mean']:.2f} dB  ±  {results['psnr_std']:.2f} dB\n")
            f.write(f"SSIM  {results['ssim_mean']:.4f}  ±  {results['ssim_std']:.4f}\n")
        else:
            f.write(f"Embedding MSE  {results['mse_mean']:.6f}  ±  {results['mse_std']:.6f}\n")
    print(f"Saved metrics      → {txt_path}")


if __name__ == "__main__":
    main()
