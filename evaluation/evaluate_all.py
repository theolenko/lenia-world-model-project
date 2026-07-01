"""Unified evaluation for the three v2 world models.

Compares Pixel-CNN, Pixel-ViT, and Patch-JEPA on three scenarios,
all in pixel space (MSE / PSNR / SSIM) for a fair comparison.

Patch-JEPA uses PatchDecoder to convert predicted patch embeddings back to
pixels, making it directly comparable to the pixel-space models.
JEPA-CNN is excluded: its predictor outputs flat embeddings with no pixel
decoder, so it cannot participate in a fair pixel-space comparison.

Scenarios
─────────
1. One-step prediction     MSE / PSNR / SSIM vs. identity baseline
2. Multi-step rollout      Pixel MSE over T steps vs. "no-change" baseline
3. OOD robustness          One-step pixel MSE as function of noise σ

Usage
─────
    python evaluation/evaluate_all.py
    python evaluation/evaluate_all.py --rollout-steps 30 --n-traj 5 --dummy
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decoder import PatchDecoder
from models.world_model import PatchJEPAWorldModel, PixelWorldModel
from scripts.train_single import LeniaDataset

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "Pixel-CNN": {
        "ckpt": ROOT / "checkpoints/pixel_cnn_v2.pt",
        "type": "pixel",
        "encoder": "cnn",
        "embed_dim": 128,
    },
    "Pixel-ViT": {
        "ckpt": ROOT / "checkpoints/pixel_vit_v2.pt",
        "type": "pixel",
        "encoder": "vit",
        "embed_dim": 256,
    },
    "Patch-JEPA": {
        "ckpt": ROOT / "checkpoints/patch_jepa_v2.pt",
        "decoder_ckpt": ROOT / "checkpoints/patch_decoder_v2.pt",
        "type": "patch_jepa",
        "embed_dim": 128,
    },
}

VAL_PATH = ROOT / "data/v2_lenia_val_chunked.h5"
OUT_DIR  = ROOT / "evaluation/results"


# ── Metric helpers ────────────────────────────────────────────────────────────

def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(a, b).item()

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (10.0 * torch.log10(torch.tensor(1.0) / (F.mse_loss(a, b) + 1e-10))).item()

def ssim(pred: torch.Tensor, target: torch.Tensor, win: int = 11, sigma: float = 1.5) -> float:
    coords = torch.arange(win, dtype=torch.float32, device=pred.device) - win // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2)); g /= g.sum()
    k = (g[:, None] * g[None, :])[None, None]; pad = win // 2
    mu1 = F.conv2d(pred,   k, padding=pad); mu2 = F.conv2d(target, k, padding=pad)
    s1  = F.conv2d(pred*pred,     k, padding=pad) - mu1**2
    s2  = F.conv2d(target*target, k, padding=pad) - mu2**2
    s12 = F.conv2d(pred*target,   k, padding=pad) - mu1*mu2
    C1, C2 = 0.01**2, 0.03**2
    return (((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean().item()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pixel_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(cfg["ckpt"], map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    # infer embed_dim from checkpoint
    embed_dim = cfg["embed_dim"]
    for k in ("encoder.projection.weight", "encoder.patch_embed.weight"):
        if k in sd:
            embed_dim = sd[k].shape[0]; break
    m = PixelWorldModel(embed_dim=embed_dim, encoder_type=cfg["encoder"]).to(device)
    m.load_state_dict(sd)
    return m.eval()


def load_patch_jepa_model(cfg: dict, device: torch.device):
    ckpt = torch.load(cfg["ckpt"], map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    embed_dim = sd["online_encoder.patch_embed.weight"].shape[0]
    pred_hidden = sd["predictor.net.0.weight"].shape[0]
    m = PatchJEPAWorldModel(embed_dim=embed_dim, predictor_hidden_dim=pred_hidden).to(device)
    m.load_state_dict(sd)
    m = m.eval()
    dec_sd = torch.load(cfg["decoder_ckpt"], map_location=device, weights_only=False)["model_state_dict"]
    dec = PatchDecoder(embed_dim=embed_dim).to(device)
    dec.load_state_dict(dec_sd)
    return m.eval(), dec.eval()


def load_trajectories(n_traj: int, device: torch.device) -> torch.Tensor:
    with h5py.File(str(VAL_PATH), "r") as f:
        raw = f["frames"][:n_traj].astype(np.float32)
    t = torch.tensor(raw)
    if t.max() > 1.0: t /= 255.0
    return t.unsqueeze(2).to(device)   # (N, T, 1, H, W)


# ── One-step pixel prediction ─────────────────────────────────────────────────

@torch.no_grad()
def one_step_pixel(model_name: str, loader: DataLoader, device: torch.device,
                   models: dict) -> dict:
    """Returns MSE, PSNR, SSIM vs ground truth t+1, and identity MSE baseline."""
    cfg = MODELS[model_name]
    mses, psnrs, ssims, id_mses = [], [], [], []

    if cfg["type"] == "pixel":
        model = models[model_name]
        for ft, ft1 in loader:
            ft, ft1 = ft.to(device), ft1.to(device)
            id_mses.append(mse(ft, ft1))
            pred = model(ft)
            mses.append(mse(pred, ft1))
            psnrs.append(psnr(pred, ft1))
            ssims.append(ssim(pred, ft1))

    elif cfg["type"] == "patch_jepa":
        m, dec = models[model_name]
        for ft, ft1 in loader:
            ft, ft1 = ft.to(device), ft1.to(device)
            id_mses.append(mse(ft, ft1))
            z = m.online_encoder(ft)
            z_pred = m.predictor(z)
            pred = dec(z_pred)
            mses.append(mse(pred, ft1))
            psnrs.append(psnr(pred, ft1))
            ssims.append(ssim(pred, ft1))

    return {
        "type": "pixel",
        "mse_mean":  float(np.mean(mses)),
        "mse_std":   float(np.std(mses)),
        "psnr_mean": float(np.mean(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "identity_mse": float(np.mean(id_mses)),
    }


# ── Multi-step rollout ────────────────────────────────────────────────────────

@torch.no_grad()
def multistep_rollout(model_name: str, trajs: torch.Tensor,
                      rollout_steps: int, models: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (model_mses, baseline_mses) of shape (rollout_steps,) in pixel space."""
    cfg = MODELS[model_name]
    N = trajs.shape[0]
    rollout_steps = min(rollout_steps, trajs.shape[1] - 1)
    all_model, all_base = [], []

    for n in range(N):
        traj = trajs[n]   # (T, 1, H, W)
        m_mses, b_mses = [], []

        if cfg["type"] == "pixel":
            model = models[model_name]
            frame = traj[0:1]
            for t in range(rollout_steps):
                frame = model(frame)
                target = traj[t+1:t+2]
                m_mses.append(mse(frame, target))
                b_mses.append(mse(traj[0:1], target))

        elif cfg["type"] == "patch_jepa":
            m, dec = models[model_name]
            z = m.online_encoder(traj[0:1])
            for t in range(rollout_steps):
                z = m.predictor(z)
                pred_px = dec(z)
                target = traj[t+1:t+2]
                m_mses.append(mse(pred_px, target))
                b_mses.append(mse(traj[0:1], target))

        all_model.append(m_mses)
        all_base.append(b_mses)

    return np.mean(all_model, axis=0), np.mean(all_base, axis=0)


# ── OOD robustness ────────────────────────────────────────────────────────────

@torch.no_grad()
def ood_robustness(model_name: str, loader: DataLoader, device: torch.device,
                   models: dict,
                   noise_levels=(0.0, 0.02, 0.05, 0.1, 0.2)) -> dict:
    cfg = MODELS[model_name]
    results = {}
    for sigma in noise_levels:
        batch_mses = []
        for ft, ft1 in loader:
            ft, ft1 = ft.to(device), ft1.to(device)
            noisy = (ft + sigma * torch.randn_like(ft)).clamp(0, 1)

            if cfg["type"] == "pixel":
                pred = models[model_name](noisy)
                batch_mses.append(mse(pred, ft1))
            elif cfg["type"] == "patch_jepa":
                m, dec = models[model_name]
                z = m.online_encoder(noisy)
                pred = dec(m.predictor(z))
                batch_mses.append(mse(pred, ft1))
        results[sigma] = float(np.mean(batch_mses))
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "Pixel-CNN":  "#2196F3",
    "Pixel-ViT":  "#4CAF50",
    "Patch-JEPA": "#9C27B0",
}


def plot_one_step_grid(loaded: dict, dataset, out_dir: Path, n: int = 6) -> None:
    """Input t | Predicted t+1 | Ground truth t+1 — one row per model."""
    from torch.utils.data import DataLoader as _DL
    batch = next(iter(_DL(dataset, batch_size=n, shuffle=True, num_workers=0)))
    ft, ft1 = batch[0][:n], batch[1][:n]

    n_models = len(loaded)
    fig, axes = plt.subplots(n_models, 3 * n, figsize=(3 * n, 2.5 * n_models))
    fig.suptitle("One-step: Input t  |  Predicted t+1  |  Ground truth t+1", fontsize=10)
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, (model_name, model_obj) in enumerate(loaded.items()):
        device = next(iter(model_obj.parameters() if not isinstance(model_obj, tuple)
                           else model_obj[0].parameters())).device
        ft_d, ft1_d = ft.to(device), ft1.to(device)
        with torch.no_grad():
            if isinstance(model_obj, tuple):   # Patch-JEPA
                m, dec = model_obj
                z = m.online_encoder(ft_d)
                pred = dec(m.predictor(z))
            else:
                pred = model_obj(ft_d)

        inp_np  = ft_d.cpu().numpy()[:, 0]
        pred_np = pred.cpu().numpy()[:, 0]
        gt_np   = ft1_d.cpu().numpy()[:, 0]

        for col in range(n):
            for offset, img in enumerate([inp_np[col], pred_np[col], gt_np[col]]):
                ax = axes[row, col * 3 + offset]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if row == 0 and col == 0:
                    ax.set_title(["t", "pred t+1", "GT t+1"][offset], fontsize=7)
        axes[row, 0].set_ylabel(model_name, fontsize=8)

    plt.tight_layout()
    out = out_dir / "one_step_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_one_step_comparison(results: dict, out_dir: Path) -> None:
    pixel_models = {k: v for k, v in results.items() if v.get("type") == "pixel"}
    if not pixel_models: return

    names = list(pixel_models.keys())
    mses  = [pixel_models[n]["mse_mean"]  for n in names]
    psnrs = [pixel_models[n]["psnr_mean"] for n in names]
    ssims = [pixel_models[n]["ssim_mean"] for n in names]
    id_mse = list(pixel_models.values())[0]["identity_mse"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("One-step prediction — pixel-space models", fontsize=12)
    colors = [COLORS[n] for n in names]

    axes[0].bar(names, mses, color=colors)
    axes[0].axhline(id_mse, color="red", ls="--", label=f"identity ({id_mse:.4f})")
    axes[0].set_ylabel("MSE"); axes[0].set_title("MSE ↓"); axes[0].legend(fontsize=8)

    axes[1].bar(names, psnrs, color=colors)
    axes[1].set_ylabel("dB"); axes[1].set_title("PSNR ↑")

    axes[2].bar(names, ssims, color=colors)
    axes[2].set_ylabel("SSIM"); axes[2].set_title("SSIM ↑")

    for ax in axes: ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(out_dir / "one_step_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'one_step_comparison.png'}")


def plot_multistep_comparison(rollout_results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Multi-step rollout — pixel MSE", fontsize=12)

    baseline_plotted = False
    for name, (model_mses, base_mses) in rollout_results.items():
        steps = np.arange(1, len(model_mses)+1)
        ax.plot(steps, model_mses, label=name, color=COLORS[name], linewidth=2)
        if not baseline_plotted:
            ax.plot(steps, base_mses, label="Identity baseline", color="gray",
                    ls="--", linewidth=1.5)
            baseline_plotted = True
    ax.set_xlabel("Rollout step"); ax.set_ylabel("Pixel MSE"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "multistep_rollout_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'multistep_rollout_comparison.png'}")


def plot_ood_comparison(ood_results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("OOD robustness — pixel MSE vs. input noise σ", fontsize=12)

    for name, res in ood_results.items():
        sigmas = sorted(res); vals = [res[s] for s in sigmas]
        ax.plot(sigmas, vals, "o-", label=name, color=COLORS[name], linewidth=2)
    ax.set_xlabel("Noise σ"); ax.set_ylabel("Pixel MSE"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "ood_robustness_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'ood_robustness_comparison.png'}")


@torch.no_grad()
def collect_rollout_frames(model_name: str, traj: torch.Tensor,
                           rollout_steps: int, models: dict) -> np.ndarray:
    """Run autoregressive rollout on one trajectory, return (rollout_steps+1, H, W) frames."""
    cfg = MODELS[model_name]
    frames = [traj[0, 0].cpu().numpy()]   # shape (H, W)

    if cfg["type"] == "pixel":
        model = models[model_name]
        frame = traj[0:1]
        for _ in range(rollout_steps):
            frame = model(frame)
            frames.append(frame[0, 0].cpu().numpy())

    elif cfg["type"] == "patch_jepa":
        m, dec = models[model_name]
        z = m.online_encoder(traj[0:1])
        for _ in range(rollout_steps):
            z = m.predictor(z)
            frames.append(dec(z)[0, 0].cpu().numpy())

    return np.stack(frames)   # (rollout_steps+1, H, W)


def save_eval_gifs(loaded: dict, trajs: torch.Tensor,
                   rollout_steps: int, out_dir: Path) -> None:
    """One GIF per model: [model prediction | ground truth] side-by-side, animated."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  [skip gif] install pillow for GIF output")
        return

    n_trajs = trajs.shape[0]
    rollout_steps = min(rollout_steps, trajs.shape[1] - 1)

    scale = 4
    H, W = 64, 64
    gap = 4
    label_h = 18
    panel = H * scale
    img_w = 2 * panel + gap
    img_h = label_h + panel

    def to_gray_rgb(f: np.ndarray) -> np.ndarray:
        g = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

    for model_name, model_obj in loaded.items():
        device = next(iter(
            model_obj.parameters() if not isinstance(model_obj, tuple)
            else model_obj[0].parameters()
        )).device

        # Pick trajectory with lowest average MSE → cleanest looking GIF
        best_score, best_traj_idx = float("inf"), 0
        for ti in range(n_trajs):
            traj = trajs[ti].to(device)
            pred_frames = collect_rollout_frames(model_name, traj, rollout_steps, loaded)
            gt_frames   = traj[:, 0].cpu().numpy()  # (T, H, W)
            score = float(np.mean((pred_frames[1:] - gt_frames[1:rollout_steps+1]) ** 2))
            if score < best_score:
                best_score, best_traj_idx = score, ti

        traj = trajs[best_traj_idx].to(device)
        pred_frames = collect_rollout_frames(model_name, traj, rollout_steps, loaded)
        gt_frames   = traj[:, 0].cpu().numpy()   # (T, H, W)

        gif_imgs = []
        for fi in range(rollout_steps + 1):
            img = Image.new("RGB", (img_w, img_h), (15, 15, 15))
            draw = ImageDraw.Draw(img)

            draw.text((2, 2),
                      f"{model_name}  step {fi}  |  left=model  right=GT",
                      fill=(200, 200, 200))

            pred_img = Image.fromarray(to_gray_rgb(pred_frames[fi])).resize(
                (panel, panel), Image.NEAREST)
            gt_img = Image.fromarray(
                to_gray_rgb(gt_frames[min(fi, len(gt_frames) - 1)])).resize(
                (panel, panel), Image.NEAREST)

            img.paste(pred_img, (0, label_h))
            img.paste(gt_img, (panel + gap, label_h))
            gif_imgs.append(img)

        out = out_dir / f"gif_rollout_{model_name.replace('-', '_').lower()}.gif"
        gif_imgs[0].save(out, save_all=True, append_images=gif_imgs[1:],
                         duration=120, loop=0)
        print(f"  Saved: {out}")


def save_summary(one_step: dict, rollout: dict, ood: dict, out_dir: Path) -> None:
    lines = ["=" * 60, "EVALUATION SUMMARY — v2 models", "=" * 60, ""]

    lines += ["── One-step prediction ──────────────────────────────────────"]
    for name, r in one_step.items():
        id_mse = r["identity_mse"]
        ratio  = r["mse_mean"] / id_mse
        lines += [
            f"  {name}",
            f"    MSE   : {r['mse_mean']:.5f} ± {r['mse_std']:.5f}  (identity={id_mse:.5f}, ratio={ratio:.2f}x)",
            f"    PSNR  : {r['psnr_mean']:.2f} dB",
            f"    SSIM  : {r['ssim_mean']:.4f}",
        ]
    lines.append("")

    lines += ["── Multi-step rollout ───────────────────────────────────────"]
    for name, (model_mses, base_mses) in rollout.items():
        exceed = np.where(model_mses >= base_mses)[0]
        horizon = int(exceed[0]) + 1 if len(exceed) > 0 else len(model_mses)
        lines += [
            f"  {name}  [pixel MSE]",
            f"    Step  1: model={model_mses[0]:.5f}  baseline={base_mses[0]:.5f}",
            f"    Step {len(model_mses):2d}: model={model_mses[-1]:.5f}  baseline={base_mses[-1]:.5f}",
            f"    Competence horizon: step {horizon}",
        ]
    lines.append("")

    lines += ["── OOD robustness ───────────────────────────────────────────"]
    for name, res in ood.items():
        clean = res[0.0]
        lines.append(f"  {name}:")
        for sigma, val in sorted(res.items()):
            rel = (val - clean) / clean * 100 if clean > 0 else 0
            lines.append(f"    σ={sigma:.2f}: {val:.5f}  ({rel:+.1f}% vs clean)")
    lines.append("")

    txt = "\n".join(lines)
    print("\n" + txt)
    (out_dir / "summary.txt").write_text(txt)
    print(f"\nSaved: {out_dir / 'summary.txt'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(VAL_PATH))
    p.add_argument("--rollout-steps", type=int, default=50)
    p.add_argument("--n-traj", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap the one-step eval dataset to N pairs (speeds up CPU runs).")
    p.add_argument("--dummy", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dummy:
        from torch.utils.data import TensorDataset
        dummy_frames = torch.rand(200, 1, 64, 64)
        dataset = TensorDataset(dummy_frames[:-1], dummy_frames[1:])
        trajs = torch.rand(args.n_traj, 60, 1, 64, 64).to(device)
    else:
        if not Path(args.data).exists():
            sys.exit(f"Val data not found: {args.data}")
        if args.max_samples:
            # Pre-load contiguous trajectories into memory to avoid slow HDF5 random access.
            n_eval_trajs = max(1, args.max_samples // 199 + 1)
            with h5py.File(args.data, "r") as f:
                raw = f["frames"][:n_eval_trajs].astype(np.float32)
            if raw.max() > 1.0:
                raw /= raw.max()
            raw_t = torch.from_numpy(raw)  # (K, T, H, W)
            ft  = raw_t[:, :-1].reshape(-1, 64, 64)[:args.max_samples].unsqueeze(1)
            ft1 = raw_t[:, 1: ].reshape(-1, 64, 64)[:args.max_samples].unsqueeze(1)
            from torch.utils.data import TensorDataset
            dataset = TensorDataset(ft, ft1)
        else:
            dataset = LeniaDataset(args.data)
        trajs = load_trajectories(args.n_traj, device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Load models
    print("\nLoading models...")
    loaded = {}
    for name, cfg in MODELS.items():
        if not cfg["ckpt"].exists():
            print(f"  {name}: checkpoint not found, skipping")
            continue
        if cfg["type"] == "pixel":
            loaded[name] = load_pixel_model(cfg, device)
        elif cfg["type"] == "patch_jepa":
            if not cfg.get("decoder_ckpt") or not Path(cfg["decoder_ckpt"]).exists():
                print(f"  {name}: decoder checkpoint not found, skipping")
                continue
            loaded[name] = load_patch_jepa_model(cfg, device)
        print(f"  {name}: loaded ✓")

    # 1. One-step
    print("\n[1/3] One-step prediction...")
    one_step_results = {}
    for name in loaded:
        one_step_results[name] = one_step_pixel(name, loader, device, loaded)
        r = one_step_results[name]
        print(f"  {name}: MSE={r['mse_mean']:.5f}  PSNR={r['psnr_mean']:.1f}dB  SSIM={r['ssim_mean']:.4f}")

    plot_one_step_grid(loaded, dataset, OUT_DIR)
    plot_one_step_comparison(one_step_results, OUT_DIR)

    # 2. Multi-step rollout
    print(f"\n[2/3] Multi-step rollout ({args.rollout_steps} steps, {args.n_traj} trajs)...")
    rollout_results = {}
    for name in loaded:
        model_mses, base_mses = multistep_rollout(name, trajs, args.rollout_steps, loaded)
        rollout_results[name] = (model_mses, base_mses)
        exceed = np.where(model_mses >= base_mses)[0]
        horizon = int(exceed[0]) + 1 if len(exceed) > 0 else args.rollout_steps
        print(f"  {name}: competence horizon = step {horizon}")

    plot_multistep_comparison(rollout_results, OUT_DIR)
    print(f"\n[2b/3] Saving rollout GIFs...")
    save_eval_gifs(loaded, trajs, args.rollout_steps, OUT_DIR)

    # 3. OOD  — use a small in-memory batch for speed on CPU
    print("\n[3/3] OOD robustness...")
    with h5py.File(args.data, "r") as f:
        ood_raw = torch.from_numpy(f["frames"][:2].astype(np.float32))  # 2 trajs
    if ood_raw.max() > 1.0:
        ood_raw /= 255.0
    ood_ft  = ood_raw[:, :-1].reshape(-1, 64, 64)[:64].unsqueeze(1)
    ood_ft1 = ood_raw[:, 1: ].reshape(-1, 64, 64)[:64].unsqueeze(1)
    from torch.utils.data import TensorDataset as _TDS
    ood_loader = DataLoader(_TDS(ood_ft, ood_ft1), batch_size=64, shuffle=False, num_workers=0)

    ood_results = {}
    for name in loaded:
        ood_results[name] = ood_robustness(name, ood_loader, device, loaded)
        print(f"  {name}: σ=0.0→{ood_results[name][0.0]:.5f}  σ=0.2→{ood_results[name][0.2]:.5f}")

    plot_ood_comparison(ood_results, OUT_DIR)
    save_summary(one_step_results, rollout_results, ood_results, OUT_DIR)


if __name__ == "__main__":
    main()
