"""Evaluation pipeline for WorldModelLenia.

Three evaluation scenarios:

  1. One-step prediction
     Metrics: MSE, PSNR, SSIM (pixel model); embedding MSE + cosine similarity (JEPA).
     Shows how accurately the model predicts t+1 from t, relative to an identity
     baseline (just copying frame t). The identity baseline MSE ≈ 0.020 is the
     minimum bar: a world model must beat it to be useful.

  2. Multi-step rollout
     Metric: MSE at step t=1..T (averaged over N trajectories) versus a "no-change"
     baseline (pixel) or a "persistent embedding" baseline (JEPA).
     Reveals how quickly errors compound: the step at which the model exceeds the
     baseline is the *competence horizon* — the practical limit of the world model.

  3. Out-of-distribution (OOD) robustness
     Metric: One-step MSE as a function of Gaussian noise σ added to the starting frame.
     A model that learned the underlying dynamics degrades gracefully; a model that
     memorised training patterns breaks sharply. This is the cheapest OOD test that
     requires no extra data.

Why MSE as the primary metric?
  It matches the training objective, is directly comparable to the identity baseline
  (~0.020), and is easy to interpret across both pixel and embedding spaces.

Why PSNR?
  PSNR = 10·log10(1/MSE). The logarithmic scale makes differences at low error levels
  more visible — it highlights whether the model is genuinely sharp or just "good enough".

Why SSIM?
  MSE penalises every pixel equally; SSIM captures luminance, contrast and local
  structure. A blurry prediction can score low MSE by averaging out the uncertainty
  but will be punished by SSIM. Together MSE + SSIM paint a fuller picture of quality.

Why cosine similarity (JEPA only)?
  JEPA lives in embedding space; MSE in that space is scale-sensitive and depends on
  the normalisation of the encoder. Cosine similarity measures directional alignment
  independent of magnitude, so it is robust to representation scale drift.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/pixel_v1.pt --setup pixel
    python scripts/evaluate.py --checkpoint checkpoints/jepa_v1.pt  --setup jepa

    # With explicit data path and rollout length
    python scripts/evaluate.py --checkpoint checkpoints/pixel_v1.pt --setup pixel \\
        --data data/lenia_val_chunked.h5 --rollout-steps 50 --n-traj 15
"""
import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import JEPAWorldModel, PixelWorldModel
from scripts.train_single import LeniaDataset


# ── Metric helpers ────────────────────────────────────────────────────────────

def batch_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred, target).item()


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse_val = F.mse_loss(pred, target)
    return (10.0 * torch.log10(torch.tensor(1.0) / (mse_val + 1e-10))).item()


def batch_ssim(pred: torch.Tensor, target: torch.Tensor,
               window_size: int = 11, sigma: float = 1.5) -> float:
    """SSIM with a Gaussian sliding window. Ranges (-1, 1]; higher is better."""
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss /= gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :])[None, None]  # (1, 1, W, W)
    pad = window_size // 2

    mu1 = F.conv2d(pred,   kernel, padding=pad)
    mu2 = F.conv2d(target, kernel, padding=pad)
    s1  = F.conv2d(pred   * pred,   kernel, padding=pad) - mu1 ** 2
    s2  = F.conv2d(target * target, kernel, padding=pad) - mu2 ** 2
    s12 = F.conv2d(pred   * target, kernel, padding=pad) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()


# ── Model / checkpoint utilities ──────────────────────────────────────────────

def _infer_embed_dim(state_dict: dict, setup: str) -> int:
    prefix = "online_encoder" if setup == "jepa" else "encoder"
    for key in (f"{prefix}.projection.weight", f"{prefix}.patch_embed.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return 128  # fallback


def _infer_encoder_type(state_dict: dict, setup: str) -> str:
    prefix = "online_encoder" if setup == "jepa" else "encoder"
    keys = [k for k in state_dict if k.startswith(prefix)]
    if any("patch_embed" in k for k in keys):
        return "vit"
    if any("conv_blocks.1.weight" in k for k in keys):  # BatchNorm2d present → LeniaEncoder
        return "cnn"
    return "simple_cnn"


def load_model(checkpoint_path: str, setup: str,
               device: torch.device) -> tuple[torch.nn.Module, object, float]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]

    embed_dim = _infer_embed_dim(sd, setup)
    enc_type  = _infer_encoder_type(sd, setup)
    print(f"  Detected embed_dim={embed_dim}, encoder_type={enc_type}")

    if setup == "pixel":
        model = PixelWorldModel(embed_dim=embed_dim, encoder_type=enc_type)
    else:
        model = JEPAWorldModel(embed_dim=embed_dim, encoder_type=enc_type)

    model.load_state_dict(sd)
    model.to(device).eval()
    return model, ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan"))


def _load_trajectories(data_path: str, n_traj: int, device: torch.device) -> torch.Tensor:
    """Load n_traj trajectories from HDF5 → (N, T, 1, H, W) float32 in [0, 1]."""
    with h5py.File(data_path, "r") as f:
        raw = f["frames"][:n_traj]          # (N, T, H, W)
        needs_norm = float(raw[0, 0].max()) > 1.0
    data = torch.tensor(raw, dtype=torch.float32)
    if needs_norm:
        data /= 255.0
    return data.unsqueeze(2).to(device)     # (N, T, 1, H, W)


# ── Evaluation 1: One-step prediction ────────────────────────────────────────

@torch.no_grad()
def eval_one_step(model: torch.nn.Module, loader: DataLoader,
                  setup: str, device: torch.device) -> dict:
    mses, psnrs, ssims, cosines, id_mses = [], [], [], [], []

    for frame_t, frame_t1 in loader:
        frame_t  = frame_t.to(device)
        frame_t1 = frame_t1.to(device)
        id_mses.append(batch_mse(frame_t, frame_t1))

        if setup == "pixel":
            pred = model(frame_t)
            mses.append(batch_mse(pred, frame_t1))
            psnrs.append(batch_psnr(pred, frame_t1))
            ssims.append(batch_ssim(pred, frame_t1))
        else:
            z_pred, z_target, _ = model(frame_t, frame_t1)
            mses.append(batch_mse(z_pred, z_target))
            cosines.append(F.cosine_similarity(z_pred, z_target, dim=1).mean().item())

    out: dict = {
        "mse_mean":     float(np.mean(mses)),
        "mse_std":      float(np.std(mses)),
        "identity_mse": float(np.mean(id_mses)),
    }
    if psnrs:
        out["psnr_mean"] = float(np.mean(psnrs))
        out["ssim_mean"] = float(np.mean(ssims))
    if cosines:
        out["cosine_mean"] = float(np.mean(cosines))
        out["cosine_std"]  = float(np.std(cosines))
    return out


# ── Evaluation 2: Multi-step rollout ─────────────────────────────────────────

@torch.no_grad()
def eval_multistep_rollout(model: torch.nn.Module, data_path: str, setup: str,
                           device: torch.device, n_traj: int = 10,
                           rollout_steps: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Auto-regressive rollout. Returns (model_mses, baseline_mses) of shape (rollout_steps,)."""
    trajs = _load_trajectories(data_path, n_traj, device)  # (N, T, 1, H, W)
    rollout_steps = min(rollout_steps, trajs.shape[1] - 1)

    all_model_mses: list[list[float]] = []
    all_baseline_mses: list[list[float]] = []

    for n in range(n_traj):
        traj = trajs[n]          # (T, 1, H, W)
        model_mses: list[float] = []
        baseline_mses: list[float] = []

        if setup == "pixel":
            current = traj[0:1]  # (1, 1, H, W)
            for t in range(rollout_steps):
                current = model(current)
                target  = traj[t + 1: t + 2]
                model_mses.append(batch_mse(current, target))
                baseline_mses.append(batch_mse(traj[0:1], target))  # "no change" baseline

        else:
            z_current = model.online_encoder(traj[0:1])
            z_persist = z_current.clone()               # persistent-embedding baseline
            for t in range(rollout_steps):
                z_current = model.predictor(z_current)
                z_target  = model.target_encoder(traj[t + 1: t + 2])
                model_mses.append(batch_mse(z_current, z_target))
                baseline_mses.append(batch_mse(z_persist, z_target))

        all_model_mses.append(model_mses)
        all_baseline_mses.append(baseline_mses)

    return np.mean(all_model_mses, axis=0), np.mean(all_baseline_mses, axis=0)


# ── Evaluation 3: OOD robustness ─────────────────────────────────────────────

@torch.no_grad()
def eval_ood(model: torch.nn.Module, loader: DataLoader, setup: str,
             device: torch.device,
             noise_levels: tuple = (0.0, 0.02, 0.05, 0.1, 0.2)) -> dict:
    """One-step MSE as a function of Gaussian noise σ added to the input frame."""
    results: dict[float, float] = {}
    for sigma in noise_levels:
        mses = []
        for frame_t, frame_t1 in loader:
            frame_t  = frame_t.to(device)
            frame_t1 = frame_t1.to(device)
            noisy = (frame_t + sigma * torch.randn_like(frame_t)).clamp(0.0, 1.0)

            if setup == "pixel":
                pred = model(noisy)
                mses.append(batch_mse(pred, frame_t1))
            else:
                z_pred, z_target, _ = model(noisy, frame_t1)
                mses.append(batch_mse(z_pred, z_target))

        results[sigma] = float(np.mean(mses))
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_one_step_grid(model: torch.nn.Module, loader: DataLoader,
                        device: torch.device, out_path: Path, n: int = 6) -> None:
    frame_t, frame_t1 = next(iter(loader))
    frame_t  = frame_t[:n].to(device)
    frame_t1 = frame_t1[:n].to(device)
    with torch.no_grad():
        pred = model(frame_t)

    inp = frame_t.cpu().numpy()
    pr  = pred.cpu().numpy()
    gt  = frame_t1.cpu().numpy()

    fig, axes = plt.subplots(n, 3, figsize=(8, 2.2 * n))
    fig.suptitle("One-step: Input t | Predicted t+1 | Ground truth t+1", fontsize=10)
    for i in range(n):
        sample_mse = float(np.mean((pr[i, 0] - gt[i, 0]) ** 2))
        for col, (img, title) in enumerate([
            (inp[i, 0], "t"),
            (pr[i,  0], f"pred  MSE={sample_mse:.4f}"),
            (gt[i,  0], "t+1"),
        ]):
            ax = axes[i, col]
            ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def _plot_rollout(model_mses: np.ndarray, baseline_mses: np.ndarray,
                  setup: str, out_path: Path) -> None:
    steps = np.arange(1, len(model_mses) + 1)
    model_label    = "Model MSE" if setup == "pixel" else "Model embedding MSE"
    baseline_label = "Identity (no change)" if setup == "pixel" else "Persistent embedding"

    plt.figure(figsize=(8, 4))
    plt.plot(steps, model_mses,    label=model_label,    linewidth=2)
    plt.plot(steps, baseline_mses, label=baseline_label, linewidth=1.5,
             linestyle="--", color="gray")
    plt.xlabel("Rollout step")
    plt.ylabel("MSE")
    plt.title("Multi-step rollout: compounding error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def _plot_ood(ood_results: dict, setup: str, out_path: Path) -> None:
    sigmas = sorted(ood_results)
    mses   = [ood_results[s] for s in sigmas]
    ylabel = "Pixel MSE" if setup == "pixel" else "Embedding MSE"

    plt.figure(figsize=(6, 4))
    plt.plot(sigmas, mses, "o-", linewidth=2, markersize=6)
    plt.xlabel("Input noise σ")
    plt.ylabel(ylabel)
    plt.title("OOD robustness: prediction error vs. input noise")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Lenia world model.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--setup", required=True, choices=["pixel", "jepa"])
    p.add_argument("--data", default="data/lenia_val_chunked.h5",
                   help="Validation HDF5 file. Default: data/lenia_val_chunked.h5")
    p.add_argument("--batch-size",    type=int, default=64)
    p.add_argument("--rollout-steps", type=int, default=50,
                   help="Number of auto-regressive steps. Default: 50")
    p.add_argument("--n-traj",        type=int, default=10,
                   help="Trajectories used for rollout eval. Default: 10")
    p.add_argument("--n-samples",     type=int, default=6,
                   help="Samples in one-step prediction grid (pixel only). Default: 6")
    p.add_argument("--out-dir",       default="experiments/eval",
                   help="Output directory. Default: experiments/eval")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, epoch, saved_loss = load_model(args.checkpoint, args.setup, device)
    print(f"  Epoch {epoch}  |  saved_val_loss={saved_loss:.6f}")

    if not Path(args.data).exists():
        sys.exit(f"ERROR: data file not found: {args.data}")

    dataset = LeniaDataset(args.data)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"  Dataset: {len(dataset):,} pairs  ({args.data})")

    out_dir = Path(args.out_dir) / f"{args.setup}_{Path(args.checkpoint).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. One-step ──────────────────────────────────────────────────────────
    print("\n[1/3] One-step prediction …")
    one_step = eval_one_step(model, loader, args.setup, device)
    ratio = one_step["mse_mean"] / one_step["identity_mse"]
    print(f"  MSE          {one_step['mse_mean']:.5f} ± {one_step['mse_std']:.5f}")
    print(f"  Identity MSE {one_step['identity_mse']:.5f}  (ratio={ratio:.3f}x, lower=better)")
    if "psnr_mean" in one_step:
        print(f"  PSNR         {one_step['psnr_mean']:.2f} dB")
        print(f"  SSIM         {one_step['ssim_mean']:.4f}")
    if "cosine_mean" in one_step:
        print(f"  Cosine sim   {one_step['cosine_mean']:.4f} ± {one_step['cosine_std']:.4f}")

    if args.setup == "pixel":
        vis_loader = DataLoader(
            dataset, batch_size=max(args.n_samples, 8), shuffle=True, num_workers=0
        )
        _plot_one_step_grid(model, vis_loader, device, out_dir / "one_step_grid.png", args.n_samples)

    # ─── 2. Multi-step rollout ────────────────────────────────────────────────
    print(f"\n[2/3] Multi-step rollout ({args.rollout_steps} steps, {args.n_traj} trajectories) …")
    model_mses, baseline_mses = eval_multistep_rollout(
        model, args.data, args.setup, device, args.n_traj, args.rollout_steps
    )
    exceed = np.where(model_mses >= baseline_mses)[0]
    horizon = int(exceed[0]) + 1 if len(exceed) > 0 else args.rollout_steps
    print(f"  MSE at step  1: {model_mses[0]:.5f}   baseline: {baseline_mses[0]:.5f}")
    print(f"  MSE at step {len(model_mses):2d}: {model_mses[-1]:.5f}   baseline: {baseline_mses[-1]:.5f}")
    print(f"  Competence horizon: step {horizon}  (model beats baseline for {horizon-1} steps)")
    _plot_rollout(model_mses, baseline_mses, args.setup, out_dir / "multistep_rollout.png")

    # ─── 3. OOD robustness ───────────────────────────────────────────────────
    print("\n[3/3] OOD robustness (Gaussian noise on input) …")
    ood = eval_ood(model, loader, args.setup, device)
    clean_mse = ood[0.0]
    for sigma, mse_val in ood.items():
        rel = (mse_val - clean_mse) / clean_mse * 100
        print(f"  σ={sigma:.2f}  MSE={mse_val:.5f}  (+{rel:+.1f}% vs clean)")
    _plot_ood(ood, args.setup, out_dir / "ood_robustness.png")

    # ─── Summary file ─────────────────────────────────────────────────────────
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"checkpoint : {args.checkpoint}\n")
        f.write(f"setup      : {args.setup}  epoch={epoch}\n\n")
        f.write("=== One-step ===\n")
        for k, v in one_step.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write(f"\n=== Multi-step rollout ===\n")
        f.write(f"  competence_horizon: {horizon}\n")
        for i, (m, b) in enumerate(zip(model_mses, baseline_mses), 1):
            f.write(f"  step {i:3d}: model={m:.6f}  baseline={b:.6f}\n")
        f.write("\n=== OOD ===\n")
        for sigma, mse_val in ood.items():
            f.write(f"  sigma={sigma:.2f}: {mse_val:.6f}\n")

    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()
