"""Run all five interventions on Pixel-CNN, Pixel-ViT, and Patch-JEPA.

For each trajectory the model rolls t_star steps on its own predictions,
then the last predicted frame is perturbed and fed into both the model and
the real Lenia simulator. MSE between model and simulator is measured per step
and averaged over n_traj val-set trajectories.

Usage:
    python interventions/run_all_interventions.py
    python interventions/run_all_interventions.py --n-traj 5 --t-star 10 --n-steps 30
"""
import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interventions.metrics import mse_per_step, time_to_divergence
DIVERGENCE_THRESHOLD = 0.05
from interventions.perturbations import INTERVENTIONS
from models.decoder import PatchDecoder
from models.world_model import PatchJEPAWorldModel, PixelWorldModel
from utils.lenia_core import make_fK, rollout as lenia_rollout

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "intervention_results"
VAL_PATH = ROOT / "data" / "v2_lenia_val_chunked.h5"

MODELS = {
    "Pixel-CNN":  {"ckpt": ROOT / "checkpoints/pixel_cnn_v2.pt",  "type": "pixel", "encoder": "cnn"},
    "Pixel-ViT":  {"ckpt": ROOT / "checkpoints/pixel_vit_v2.pt",  "type": "pixel", "encoder": "vit"},
    "Patch-JEPA": {
        "ckpt":         ROOT / "checkpoints/patch_jepa_v2.pt",
        "decoder_ckpt": ROOT / "checkpoints/patch_decoder_v2.pt",
        "type": "patch_jepa",
    },
}

COLORS = {
    "Pixel-CNN":  "#2196F3",
    "Pixel-ViT":  "#4CAF50",
    "Patch-JEPA": "#9C27B0",
}


# ── Model loading ──────────────────────────────────────────────────────────────

def load_all_models(device):
    loaded = {}
    print("Loading models...")
    for name, cfg in MODELS.items():
        if not cfg["ckpt"].exists():
            print(f"  {name}: checkpoint not found, skipping")
            continue
        sd = torch.load(cfg["ckpt"], map_location=device, weights_only=False)["model_state_dict"]

        if cfg["type"] == "pixel":
            embed_dim = 128
            for k in ("encoder.projection.weight", "encoder.patch_embed.weight"):
                if k in sd:
                    embed_dim = sd[k].shape[0]
                    break
            m = PixelWorldModel(embed_dim=embed_dim, encoder_type=cfg["encoder"]).to(device)
            m.load_state_dict(sd)
            loaded[name] = m.eval()

        elif cfg["type"] == "patch_jepa":
            dec_path = cfg.get("decoder_ckpt")
            if not dec_path or not Path(dec_path).exists():
                print(f"  {name}: decoder checkpoint not found, skipping")
                continue
            embed_dim = sd["online_encoder.patch_embed.weight"].shape[0]
            pred_h = sd["predictor.net.0.weight"].shape[0]
            m = PatchJEPAWorldModel(embed_dim=embed_dim, predictor_hidden_dim=pred_h).to(device)
            m.load_state_dict(sd)
            m.eval()
            dec_sd = torch.load(dec_path, map_location=device, weights_only=False)["model_state_dict"]
            dec = PatchDecoder(embed_dim=embed_dim).to(device)
            dec.load_state_dict(dec_sd)
            dec.eval()
            loaded[name] = (m, dec)

        print(f"  {name}: loaded ✓")
    return loaded


# ── Rollout helpers ────────────────────────────────────────────────────────────

@torch.no_grad()
def model_rollout(model_name: str, start_frame: np.ndarray,
                  n_steps: int, loaded: dict, device) -> np.ndarray:
    """Autoregressive rollout: feed the model's own prediction back in as input."""
    frame_t = torch.from_numpy(start_frame).unsqueeze(0).unsqueeze(0).to(device)
    frames = [start_frame]

    if MODELS[model_name]["type"] == "pixel":
        model = loaded[model_name]
        for _ in range(n_steps):
            frame_t = model(frame_t)
            frames.append(frame_t.squeeze().cpu().numpy())

    else:  # patch_jepa
        m, dec = loaded[model_name]
        z = m.online_encoder(frame_t)
        for _ in range(n_steps):
            z = m.predictor(z)
            frames.append(dec(z).squeeze().cpu().numpy())

    return np.stack(frames)  # (n_steps+1, H, W)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_frame_snapshots(intv_name: str, model_posts: dict, gt_post: np.ndarray,
                         intervened: np.ndarray, n_steps: int, out_dir: Path) -> None:
    """Model predictions vs Lenia GT at key steps — one row per model + one GT row."""
    vis_steps = sorted({0, 1, n_steps // 2, n_steps})
    models = list(model_posts.keys())
    n_rows = len(models) + 1  # models + GT row

    fig, axes = plt.subplots(n_rows, len(vis_steps),
                             figsize=(3 * len(vis_steps), 2.5 * n_rows))
    fig.suptitle(f"Intervention: {intv_name} — frame snapshots", fontsize=10)

    for col, s in enumerate(vis_steps):
        axes[0, col].set_title(f"step {s}", fontsize=8)

    for row, model_name in enumerate(models):
        frames = model_posts[model_name]
        axes[row, 0].set_ylabel(model_name, fontsize=8)
        for col, s in enumerate(vis_steps):
            img = frames[min(s, len(frames) - 1)]
            axes[row, col].imshow(img, cmap="viridis", vmin=0, vmax=1)
            axes[row, col].axis("off")

    axes[-1, 0].set_ylabel("Lenia GT", fontsize=8)
    for col, s in enumerate(vis_steps):
        img = gt_post[min(s, len(gt_post) - 1)]
        axes[-1, col].imshow(img, cmap="viridis", vmin=0, vmax=1)
        axes[-1, col].axis("off")

    plt.tight_layout()
    out = out_dir / f"snapshots_{intv_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_intervention(intv_name: str, results: dict, n_steps: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(1, n_steps + 1)
    for name, mses in results.items():
        ax.plot(steps[:len(mses)], mses, label=name, color=COLORS[name], linewidth=2)
    ax.set_title(f"Intervention: {intv_name} — model vs. Lenia ground truth")
    ax.set_xlabel("Steps after intervention (t*)")
    ax.set_ylabel("Pixel MSE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"intervention_{intv_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_summary_heatmap(table: dict, out_dir: Path) -> None:
    models = list(table.keys())
    interventions = list(INTERVENTIONS.keys())
    data = np.full((len(models), len(interventions)), np.nan)
    for i, m in enumerate(models):
        for j, intv in enumerate(interventions):
            if intv in table[m]:
                data[i, j] = table[m][intv]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto",
                   vmin=np.nanmin(data), vmax=np.nanpercentile(data, 90))
    ax.set_xticks(range(len(interventions)))
    ax.set_xticklabels(interventions, rotation=20)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    plt.colorbar(im, ax=ax, label="Mean pixel MSE (steps 1–10, model vs. Lenia GT)")
    ax.set_title("Intervention sensitivity — lower = model stays closer to real Lenia physics", fontsize=10)
    for i in range(len(models)):
        for j in range(len(interventions)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    out = out_dir / "intervention_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-traj",  type=int, default=5,
                   help="Number of val-set trajectories to average over.")
    p.add_argument("--t-star",  type=int, default=10,
                   help="Steps the model rolls before the intervention.")
    p.add_argument("--n-steps", type=int, default=30,
                   help="Steps to compare after the intervention.")
    p.add_argument("--data",    type=str, default=str(VAL_PATH),
                   help="Path to val HDF5 file.")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    print(f"Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, fK = make_fK()

    if not Path(args.data).exists():
        sys.exit(f"Val data not found: {args.data}")

    # Load first n_traj trajectories from val set — use traj[0] as start frame
    with h5py.File(args.data, "r") as f:
        raw = f["frames"][:args.n_traj].astype(np.float32)
    if raw.max() > 1.0:
        raw /= raw.max()
    # raw shape: (n_traj, T, H, W)

    loaded = load_all_models(device)
    if not loaded:
        sys.exit("No models loaded.")

    # table[model][intervention] = mean MSE over steps 1–10
    table = {m: {} for m in loaded}
    summary_lines = ["=" * 60, "INTERVENTION SUMMARY", "=" * 60, ""]

    for intv_name in INTERVENTIONS:
        print(f"\n── Intervention: {intv_name} ────────────────────────────")

        accum = {m: [] for m in loaded}
        last_model_posts = {}  # keep last traj's frames for snapshot plot
        last_gt_post = None
        last_intervened = None

        for traj_i in range(args.n_traj):
            t0 = raw[traj_i, 0]  # first frame of trajectory — same as run_intervention.py

            for model_name in loaded:
                pre_rollout = model_rollout(model_name, t0, args.t_star, loaded, device)
                pre_frame = pre_rollout[-1]

                intervened = INTERVENTIONS[intv_name](pre_frame, rng=rng)

                model_post = model_rollout(model_name, intervened, args.n_steps, loaded, device)
                gt_post = lenia_rollout(intervened, args.n_steps, fK)

                # index 0 is the intervened frame itself — skip it
                mses = mse_per_step(model_post[1:], gt_post[1:])
                accum[model_name].append(mses)

                last_model_posts[model_name] = model_post
                last_gt_post = gt_post
                last_intervened = intervened

        # Average over trajectories, report divergence step
        intv_results = {}
        for model_name in loaded:
            mean_mses = list(np.mean(accum[model_name], axis=0))
            intv_results[model_name] = mean_mses
            mean_10 = float(np.mean(mean_mses[:10]))
            table[model_name][intv_name] = mean_10
            exceed = [i for i, v in enumerate(mean_mses) if v > DIVERGENCE_THRESHOLD]
            div_str = f"step {exceed[0] + 1}" if exceed else "never"
            print(f"  {model_name}: mean MSE steps 1–10 = {mean_10:.5f}  "
                  f"step1={mean_mses[0]:.5f}  "
                  f"diverges (>{DIVERGENCE_THRESHOLD}) at {div_str}")

        plot_intervention(intv_name, intv_results, args.n_steps, OUT_DIR)
        if last_gt_post is not None:
            plot_frame_snapshots(intv_name, last_model_posts, last_gt_post,
                                 last_intervened, args.n_steps, OUT_DIR)

        summary_lines.append(f"── {intv_name} ──")
        for model_name, mses in intv_results.items():
            summary_lines.append(
                f"  {model_name}: step1={mses[0]:.5f}  "
                f"step10={mses[min(9, len(mses)-1)]:.5f}  "
                f"step{args.n_steps}={mses[-1]:.5f}"
            )
        summary_lines.append("")

    plot_summary_heatmap(table, OUT_DIR)

    txt = "\n".join(summary_lines)
    print("\n" + txt)
    (OUT_DIR / "summary.txt").write_text(txt)
    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
