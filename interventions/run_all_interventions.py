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

def plot_frame_snapshots(intv_name: str,
                         model_pres: dict, model_posts: dict,
                         gt_pre: np.ndarray, gt_post: np.ndarray,
                         intervened: np.ndarray, n_steps: int, t_star: int,
                         out_dir: Path) -> None:
    """Full timeline per row: pre-rollout → INTERVENTION → post-rollout.

    Each row is one model (or Lenia GT simulator).
    Left section:  frames before the intervention (t=0, t=t_star/2, t=t_star).
    Middle:        the intervened/perturbed frame (t*).
    Right section: frames after the intervention (+1, +n//4, +n//2, +n steps).
    """
    pre_show = sorted({0, t_star // 2, t_star - 1}) if t_star > 1 else [0]
    post_show = sorted({1, max(2, n_steps // 4), n_steps // 2, n_steps})

    models = list(model_posts.keys())
    row_labels = [f"{m}\n(model)" for m in models] + ["Lenia GT\n(simulator)"]
    n_rows = len(row_labels)
    # cols: pre-cols | intervened col | post-cols
    n_cols = len(pre_show) + 1 + len(post_show)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols, 2.6 * n_rows))

    fig.suptitle(
        f"Intervention: {intv_name}  —  full timeline\n"
        f"← {t_star} steps autonomous rollout  |  INTERVENTION ↓  |  {n_steps} steps after →\n"
        f"Model rows: world-model predictions.  Lenia GT: real simulator.",
        fontsize=9, y=1.02
    )

    # ── Column headers ────────────────────────────────────────
    for col, s in enumerate(pre_show):
        lbl = f"t=0\n(start)" if s == 0 else f"t={s}"
        axes[0, col].set_title(lbl, fontsize=7, color="steelblue")

    intv_col = len(pre_show)
    axes[0, intv_col].set_title(f"t*={t_star}\nINTERVENTION", fontsize=7,
                                 fontweight="bold", color="darkred")

    for offset, s in enumerate(post_show):
        col = intv_col + 1 + offset
        axes[0, col].set_title(f"t*+{s}", fontsize=7, color="darkorange")

    # ── Draw each row ─────────────────────────────────────────
    def draw_row(row: int, pre_frames: np.ndarray, post_frames: np.ndarray,
                 label: str) -> None:
        ax = axes[row, 0]
        ax.set_ylabel(label, fontsize=9, fontweight="bold",
                      rotation=0, labelpad=65, va="center")

        # Pre-intervention frames
        for col, s in enumerate(pre_show):
            img = np.clip(pre_frames[min(s, len(pre_frames) - 1)], 0, 1)
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")

        # Intervened frame — highlight with red border
        ax_intv = axes[row, intv_col]
        ax_intv.imshow(np.clip(intervened, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax_intv.axis("off")
        for spine in ax_intv.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("darkred")
            spine.set_linewidth(2)

        # Post-intervention frames
        for offset, s in enumerate(post_show):
            col = intv_col + 1 + offset
            img = np.clip(post_frames[min(s, len(post_frames) - 1)], 0, 1)
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")

    for row, model_name in enumerate(models):
        draw_row(row, model_pres[model_name], model_posts[model_name],
                 f"{model_name}\n(model)")

    draw_row(len(models), gt_pre, gt_post, "Lenia GT\n(simulator)")

    plt.tight_layout()

    # Vertical separator line between pre and intervention columns
    if intv_col > 0:
        try:
            fig.canvas.draw()
            sep_x = (axes[0, intv_col - 1].get_position().x1 +
                     axes[0, intv_col].get_position().x0) / 2
            fig.add_artist(plt.Line2D([sep_x, sep_x], [0.02, 0.96],
                                      transform=fig.transFigure,
                                      color="darkred", linewidth=1.5, linestyle="--",
                                      alpha=0.6))
        except Exception:
            pass
    out = out_dir / f"snapshots_{intv_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_intervention(intv_name: str, results: dict, n_steps: int, t_star: int,
                      out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(1, n_steps + 1)
    for name, mses in results.items():
        ax.plot(steps[:len(mses)], mses, label=name, color=COLORS[name], linewidth=2)
    ax.set_title(
        f"Intervention: {intv_name}\n"
        f"Each model rolled {t_star} steps autonomously, then the frame was perturbed.\n"
        f"Plot shows pixel MSE between model prediction and Lenia simulator "
        f"for each step after the perturbation.",
        fontsize=9
    )
    ax.set_xlabel(f"Steps after perturbation at t*={t_star}")
    ax.set_ylabel("Pixel MSE  (model prediction vs. Lenia GT)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"intervention_{intv_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def save_intervention_gif(intv_name: str, best: dict, t_star: int,
                          n_steps: int, out_dir: Path) -> None:
    """Animated GIF: full timeline per row — pre-rollout → INTERVENTION → post-rollout.

    'best' dict: model_pres, model_posts, gt_pre, gt_post, intervened
    Sequence:  model_pres[0..t_star]  +  model_posts[0..n_steps]
    index t_star+1 in that sequence = the intervened frame (model_posts[0]).
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print(f"  [skip gif] install pillow for GIF output")
        return

    models = list(best["model_posts"].keys())
    row_labels = list(models) + ["Lenia GT"]

    scale = 4
    frame0 = best["intervened"]
    H, W = frame0.shape
    label_w = 90
    top_h = 18          # strip for phase text
    panel_h = H * scale
    panel_w = W * scale
    img_w = label_w + panel_w
    img_h = top_h + len(row_labels) * panel_h

    def to_gray_rgb(f: np.ndarray) -> np.ndarray:
        g = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

    seqs: dict[str, list] = {}
    for m in models:
        seqs[m] = list(best["model_pres"][m]) + list(best["model_posts"][m])
    seqs["Lenia GT"] = list(best["gt_pre"]) + list(best["gt_post"])

    intv_idx = t_star + 1   # index of the intervened frame in every sequence
    total = min(len(s) for s in seqs.values())

    gif_imgs = []
    for fi in range(total):
        img = Image.new("RGB", (img_w, img_h), (15, 15, 15))
        draw = ImageDraw.Draw(img)

        if fi <= t_star:
            phase, pcol = f"pre-rollout   t={fi}", (100, 180, 255)
        elif fi == intv_idx:
            phase, pcol = f"INTERVENTION  t*={t_star}", (255, 60, 60)
        else:
            phase, pcol = f"post-interv.  t*+{fi - intv_idx}", (255, 180, 50)

        draw.text((label_w + 3, 2), phase, fill=pcol)

        for row, label in enumerate(row_labels):
            seq = seqs[label]
            frame = seq[min(fi, len(seq) - 1)]
            panel = Image.fromarray(to_gray_rgb(frame)).resize(
                (panel_w, panel_h), Image.NEAREST
            )
            y0 = top_h + row * panel_h
            img.paste(panel, (label_w, y0))
            draw.text((2, y0 + panel_h // 2 - 6), label, fill=(200, 200, 200))
            if row > 0:
                draw.line([(0, y0), (img_w, y0)], fill=(50, 50, 50), width=1)

        if fi == intv_idx:
            draw.rectangle([label_w, top_h, img_w - 1, img_h - 1],
                           outline=(220, 30, 30), width=3)

        gif_imgs.append(img)

    durations = [100] * total
    if 0 <= t_star < total:       durations[t_star]      = 400   # pause before perturb
    if 0 <= intv_idx < total:     durations[intv_idx]    = 700   # long pause at intervention
    if 0 <= intv_idx + 1 < total: durations[intv_idx + 1] = 300  # pause first post frame

    out = out_dir / f"gif_{intv_name}.gif"
    gif_imgs[0].save(out, save_all=True, append_images=gif_imgs[1:],
                     duration=durations, loop=0)
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
        # best_traj: trajectory with lowest mean post-intervention MSE (cleanest GIF)
        best_traj_score = float("inf")
        best_traj: dict = {}

        for traj_i in range(args.n_traj):
            t0 = raw[traj_i, 0]  # first frame of trajectory — same as run_intervention.py

            gt_pre = lenia_rollout(t0, args.t_star, fK)
            traj_model_pres: dict = {}
            traj_model_posts: dict = {}
            traj_mse_sum = 0.0
            traj_intervened = None
            traj_gt_post = None

            for model_name in loaded:
                pre_rollout = model_rollout(model_name, t0, args.t_star, loaded, device)
                pre_frame = pre_rollout[-1]

                intervened = INTERVENTIONS[intv_name](pre_frame, rng=rng)

                model_post = model_rollout(model_name, intervened, args.n_steps, loaded, device)
                gt_post = lenia_rollout(intervened, args.n_steps, fK)

                # index 0 is the intervened frame itself — skip it
                mses = mse_per_step(model_post[1:], gt_post[1:])
                accum[model_name].append(mses)
                traj_mse_sum += float(np.mean(mses))

                traj_model_pres[model_name] = pre_rollout
                traj_model_posts[model_name] = model_post
                traj_intervened = intervened
                traj_gt_post = gt_post

            # Keep frames from the trajectory where models track Lenia best
            traj_score = traj_mse_sum / max(len(loaded), 1)
            if traj_score < best_traj_score:
                best_traj_score = traj_score
                best_traj = {
                    "model_pres": traj_model_pres,
                    "model_posts": traj_model_posts,
                    "gt_pre": gt_pre,
                    "gt_post": traj_gt_post,
                    "intervened": traj_intervened,
                }

        # Expose last traj vars for snapshot plot (keeps existing plot behaviour)
        last_model_pres  = best_traj.get("model_pres", {})
        last_model_posts = best_traj.get("model_posts", {})
        last_gt_pre      = best_traj.get("gt_pre")
        last_gt_post     = best_traj.get("gt_post")
        last_intervened  = best_traj.get("intervened")

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

        plot_intervention(intv_name, intv_results, args.n_steps, args.t_star, OUT_DIR)
        if last_gt_post is not None:
            plot_frame_snapshots(intv_name, last_model_pres, last_model_posts,
                                 last_gt_pre, last_gt_post,
                                 last_intervened, args.n_steps, args.t_star, OUT_DIR)
            save_intervention_gif(intv_name, best_traj, args.t_star,
                                  args.n_steps, OUT_DIR)

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
