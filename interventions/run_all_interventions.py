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
                         model_intervened: dict,
                         gt_pre: np.ndarray, gt_post: np.ndarray,
                         gt_intervened: np.ndarray,
                         n_steps: int, t_star: int,
                         out_dir: Path) -> None:
    """Full timeline grid: pre-rollout | before-perturbation | PERTURBED | post-rollout.

    GT row:    Lenia runs t* steps → intervention on GT's OWN frame → Lenia continues.
    Model rows: model runs t* steps → same intervention params on model's OWN frame → model continues.
    Comparing model_post vs gt_post shows whether model tracks real Lenia physics.
    """
    pre_show = sorted({0, t_star // 2}) if t_star > 2 else [0]
    post_show = sorted({1, max(2, n_steps // 4), n_steps // 2, n_steps})

    models = list(model_posts.keys())
    first_model = models[0] if models else None
    n_rows = len(models) + 1   # models + GT
    # cols: pre | before-perturbation | PERTURBED | post
    n_cols = len(pre_show) + 1 + 1 + len(post_show)
    before_col = len(pre_show)        # "before perturbation" column
    intv_col   = before_col + 1       # the perturbed frame column

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols, 2.6 * n_rows))

    fig.suptitle(
        f"Intervention: {intv_name}  —  full timeline\n"
        f"Blue: {t_star}-step pre-rollout  |  Red: perturbation at t*  |  "
        f"Orange: {n_steps} steps after (model vs. Lenia GT)",
        fontsize=9, y=1.02
    )

    # ── Column headers ─────────────────────────────────────────
    for col, s in enumerate(pre_show):
        axes[0, col].set_title("t=0\n(start)" if s == 0 else f"t={s}",
                               fontsize=7, color="steelblue")
    axes[0, before_col].set_title(f"t={t_star}\nbefore pert.", fontsize=7,
                                   color="saddlebrown")
    axes[0, intv_col].set_title(f"t*={t_star}\nPERTURBED ↓", fontsize=7,
                                 fontweight="bold", color="darkred")
    for offset, s in enumerate(post_show):
        axes[0, intv_col + 1 + offset].set_title(f"t*+{s}", fontsize=7,
                                                   color="darkorange")

    def _hide_ax(ax):
        """Hide ticks/spines without calling axis('off') so ylabel stays visible."""
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    # ── Draw each row ──────────────────────────────────────────
    def draw_row(row: int, pre_frames: np.ndarray, post_frames: np.ndarray,
                 row_intervened: np.ndarray, label: str) -> None:

        # Pre frames — first panel kept "on" so ylabel is visible
        for col, s in enumerate(pre_show):
            ax = axes[row, col]
            ax.imshow(np.clip(pre_frames[min(s, len(pre_frames) - 1)], 0, 1),
                      cmap="gray", vmin=0, vmax=1)
            if col == 0:
                _hide_ax(ax)
                ax.set_ylabel(label, fontsize=9, fontweight="bold",
                              rotation=0, labelpad=70, va="center")
            else:
                ax.axis("off")

        # Frame right before perturbation (model's own predicted frame at t*)
        axes[row, before_col].imshow(
            np.clip(pre_frames[min(t_star, len(pre_frames) - 1)], 0, 1),
            cmap="gray", vmin=0, vmax=1)
        axes[row, before_col].axis("off")

        # Perturbed frame — each model's own intervened frame
        ax_intv = axes[row, intv_col]
        ax_intv.imshow(np.clip(row_intervened, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax_intv.axis("off")
        for spine in ax_intv.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("darkred")
            spine.set_linewidth(2.5)

        # Post frames
        for offset, s in enumerate(post_show):
            axes[row, intv_col + 1 + offset].imshow(
                np.clip(post_frames[min(s, len(post_frames) - 1)], 0, 1),
                cmap="gray", vmin=0, vmax=1)
            axes[row, intv_col + 1 + offset].axis("off")

    for row, model_name in enumerate(models):
        row_intv = model_intervened.get(model_name, np.zeros_like(gt_pre[0]))
        draw_row(row, model_pres[model_name], model_posts[model_name],
                 row_intv, f"{model_name}\n(model)")

    # GT row: Lenia's own t* frame → intervention → Lenia continues
    draw_row(len(models), gt_pre, gt_post, gt_intervened, "Lenia GT\n(simulator)")

    plt.tight_layout()

    # Red dashed separator between "before" and "perturbed" columns
    try:
        fig.canvas.draw()
        sep_x = (axes[0, before_col].get_position().x1 +
                 axes[0, intv_col].get_position().x0) / 2
        fig.add_artist(plt.Line2D([sep_x, sep_x], [0.02, 0.96],
                                  transform=fig.transFigure,
                                  color="darkred", linewidth=1.5, linestyle="--",
                                  alpha=0.7))
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


def save_model_vs_gt_gif(model_name: str, intv_name: str, best: dict,
                          t_star: int, n_steps: int, out_dir: Path) -> None:
    """One GIF per model: [Model | Lenia GT] side by side, time animated.

    Model runs t* steps autonomously → its own intervention → model prediction.
    GT runs t* Lenia steps → intervention on GT's own frame → Lenia continues.
    Both trajectories shown in parallel so differences are immediately visible.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  [skip gif] install pillow for GIF output")
        return

    def to_gray_rgb(f: np.ndarray) -> np.ndarray:
        g = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

    scale = 4
    H, W = best["gt_intervened"].shape
    gap = 6
    top_h = 20
    bot_h = 16
    panel_h = H * scale
    panel_w = W * scale
    # Two columns: model (left) | GT (right)
    img_w = 2 * panel_w + gap
    img_h = top_h + panel_h + bot_h

    # Full sequences: pre (t*+1 frames) + post (n_steps+1 frames, index 0 = intervened)
    seq_model = list(best["model_pres"][model_name]) + list(best["model_posts"][model_name])
    seq_gt    = list(best["gt_pre"])                 + list(best["gt_post"])
    intv_idx  = t_star + 1   # index of the intervened frame in both sequences
    total     = min(len(seq_model), len(seq_gt))

    gif_imgs = []
    for fi in range(total):
        img  = Image.new("RGB", (img_w, img_h), (15, 15, 15))
        draw = ImageDraw.Draw(img)

        is_intv = (fi == intv_idx)
        if fi <= t_star:
            phase, pcol = f"pre-rollout  t={fi}", (100, 180, 255)
        elif is_intv:
            phase, pcol = f"INTERVENTION  t*={t_star}", (255, 60, 60)
        else:
            phase, pcol = f"post-interv.  t*+{fi - intv_idx}", (255, 180, 50)

        draw.text((4, 2), phase, fill=pcol)

        for ci, (label, seq) in enumerate([
            (model_name, seq_model),
            ("Lenia GT",  seq_gt),
        ]):
            x0    = ci * (panel_w + gap)
            frame = seq[min(fi, len(seq) - 1)]
            panel = Image.fromarray(to_gray_rgb(frame)).resize(
                (panel_w, panel_h), Image.NEAREST)
            img.paste(panel, (x0, top_h))
            short = label.replace("Lenia GT", "GT").replace("Pixel-", "Px-")
            draw.text((x0 + 2, top_h + panel_h + 2), short, fill=(200, 200, 200))
            if is_intv:
                draw.rectangle([x0, top_h, x0 + panel_w - 1, top_h + panel_h - 1],
                               outline=(220, 30, 30), width=3)

        gif_imgs.append(img)

    durations = [100] * total
    if 0 <= t_star     < total: durations[t_star]       = 400
    if 0 <= intv_idx   < total: durations[intv_idx]     = 700
    if 0 <= intv_idx+1 < total: durations[intv_idx + 1] = 300

    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    out = out_dir / f"gif_{intv_name}_{safe_name}.gif"
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

    # Load n_traj trajectories where the organism stays active over the evaluation window.
    # Filters by minimum mean pixel value over the first (t_star + n_steps) frames so we
    # don't evaluate on trajectories where the organism dies early.
    min_activity = 0.05
    eval_window  = args.t_star + args.n_steps + 1
    with h5py.File(args.data, "r") as f:
        all_frames = f["frames"]   # (N, T, H, W)
        selected = []
        for i in range(all_frames.shape[0]):
            chunk = all_frames[i, :eval_window].astype(np.float32)
            if chunk.max() > 1.0:
                chunk /= 255.0
            if chunk.mean(axis=(1, 2)).min() >= min_activity:
                selected.append(all_frames[i].astype(np.float32))
            if len(selected) == args.n_traj:
                break
    raw = np.stack(selected)
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
            t0 = raw[traj_i, 0]

            # Ground truth: Lenia runs t* steps from t0, then the intervention
            # is applied to GT's OWN frame at t*.  GT continues as real Lenia.
            gt_pre = lenia_rollout(t0, args.t_star, fK)  # (t_star+1, H, W)

            # Save rng state so GT and all models use identical intervention params
            # (same random blob position / noise seed), just on different frames.
            rng_state = rng.bit_generator.state

            rng.bit_generator.state = rng_state
            gt_intervened = INTERVENTIONS[intv_name](gt_pre[-1], rng=rng)
            gt_post       = lenia_rollout(gt_intervened, args.n_steps, fK)
            # gt_post is the shared GT reference for all models this trajectory

            traj_model_pres:       dict = {}
            traj_model_posts:      dict = {}
            traj_model_intervened: dict = {}
            traj_mse_sum = 0.0

            for model_name in loaded:
                # Reset rng → same intervention params as GT, applied to model's frame
                rng.bit_generator.state = rng_state
                pre_rollout   = model_rollout(model_name, t0, args.t_star, loaded, device)
                pre_frame     = pre_rollout[-1]
                model_intv    = INTERVENTIONS[intv_name](pre_frame, rng=rng)
                model_post    = model_rollout(model_name, model_intv, args.n_steps, loaded, device)

                # MSE vs shared Lenia GT (index 0 = intervened frame itself — skip)
                mses = mse_per_step(model_post[1:], gt_post[1:])
                accum[model_name].append(mses)
                traj_mse_sum += float(np.mean(mses))

                traj_model_pres[model_name]       = pre_rollout
                traj_model_posts[model_name]      = model_post
                traj_model_intervened[model_name] = model_intv

            # Advance rng past this trajectory so subsequent trajs differ
            rng.bit_generator.state = rng_state
            _ = INTERVENTIONS[intv_name](gt_pre[-1], rng=rng)

            traj_score = traj_mse_sum / max(len(loaded), 1)
            if traj_score < best_traj_score:
                best_traj_score = traj_score
                best_traj = {
                    "model_pres":       traj_model_pres,
                    "model_posts":      traj_model_posts,
                    "model_intervened": traj_model_intervened,
                    "gt_pre":           gt_pre,
                    "gt_post":          gt_post,
                    "gt_intervened":    gt_intervened,
                }

        last_model_pres       = best_traj.get("model_pres", {})
        last_model_posts      = best_traj.get("model_posts", {})
        last_model_intervened = best_traj.get("model_intervened", {})
        last_gt_pre           = best_traj.get("gt_pre")
        last_gt_post          = best_traj.get("gt_post")
        last_gt_intervened    = best_traj.get("gt_intervened")

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
            plot_frame_snapshots(intv_name,
                                 last_model_pres, last_model_posts, last_model_intervened,
                                 last_gt_pre, last_gt_post, last_gt_intervened,
                                 args.n_steps, args.t_star, OUT_DIR)
            for model_name in loaded:
                save_model_vs_gt_gif(model_name, intv_name, best_traj,
                                     args.t_star, args.n_steps, OUT_DIR)

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
