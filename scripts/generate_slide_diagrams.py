"""Generate architecture diagrams for presentation slides."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "final-result-visuals" / "slides"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#1a1a2e"
BOX_PIXEL = "#2196F3"
BOX_ENC   = "#1565C0"
BOX_PRED  = "#7B1FA2"
BOX_DEC   = "#2E7D32"
BOX_EMA   = "#E65100"
BOX_FRAME = "#37474F"
ARROW_COL = "#ECEFF1"
TEXT_COL  = "#ECEFF1"
LOSS_COL  = "#FF5722"


def _box(ax, x, y, w, h, label, sublabel=None, color=BOX_FRAME, fontsize=11, radius=0.04):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.01,rounding_size={radius}",
                         facecolor=color, edgecolor="white", linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.06 if sublabel else 0), label, ha="center", va="center",
            color=TEXT_COL, fontsize=fontsize, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.12, sublabel, ha="center", va="center",
                color=TEXT_COL, fontsize=fontsize - 2.5, alpha=0.75, zorder=4)


def _arrow(ax, x0, y0, x1, y1, label=None, color=ARROW_COL, lw=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=18),
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my + 0.07, label, ha="center", va="bottom",
                color=color, fontsize=8.5, alpha=0.9, zorder=5)


def _base_fig(w=12, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ── Slide 1: Pixel-CNN / Pixel-ViT ────────────────────────────────────────────

def pixel_architecture():
    fig, ax = _base_fig(11, 3.5)
    ax.text(0.5, 0.93, "Pixel-CNN  /  Pixel-ViT — Direct Next-Frame Prediction",
            ha="center", va="top", color=TEXT_COL, fontsize=13, fontweight="bold")

    xs = [0.12, 0.35, 0.58, 0.81]
    y  = 0.47
    bw, bh = 0.18, 0.28

    _box(ax, xs[0], y, bw, bh, "frame$_t$",      color=BOX_FRAME)
    _box(ax, xs[1], y, bw, bh, "Encoder",  "CNN / ViT", color=BOX_ENC)
    _box(ax, xs[2], y, bw, bh, "Decoder",  "transposed conv", color=BOX_DEC)
    _box(ax, xs[3], y, bw, bh, "frame$_{t+1}$\npredicted", color=BOX_FRAME)

    _arrow(ax, xs[0]+bw/2, y, xs[1]-bw/2, y)
    _arrow(ax, xs[1]+bw/2, y, xs[2]-bw/2, y, "z  (B, D)")
    _arrow(ax, xs[2]+bw/2, y, xs[3]-bw/2, y)

    # loss
    ax.annotate("", xy=(xs[3], y - bh/2 - 0.09), xytext=(xs[3], y - bh/2),
                arrowprops=dict(arrowstyle="-", color=LOSS_COL, lw=1.5, linestyle="dashed"))
    ax.annotate("", xy=(xs[2], y - bh/2 - 0.09), xytext=(xs[3], y - bh/2 - 0.09),
                arrowprops=dict(arrowstyle="-|>", color=LOSS_COL, lw=1.5, mutation_scale=14))
    ax.text((xs[2]+xs[3])/2, y - bh/2 - 0.16, "MSE loss  (pixel space)",
            ha="center", va="top", color=LOSS_COL, fontsize=9.5)

    # ground truth label
    ax.text(xs[3], y - bh/2 - 0.27, "ground truth\nframe$_{t+1}$",
            ha="center", va="top", color=TEXT_COL, fontsize=8.5, alpha=0.65)

    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "arch_pixel.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Slide 2: Patch-JEPA Stage 1 (JEPA training) ───────────────────────────────

def patch_jepa_stage1():
    fig, ax = _base_fig(13, 4.2)
    ax.text(0.5, 0.96, "Patch-JEPA — Stage 1: JEPA Training  (EMA target)",
            ha="center", va="top", color=TEXT_COL, fontsize=13, fontweight="bold")

    bw, bh = 0.14, 0.28
    y = 0.52

    # Online path
    ox = [0.08, 0.26, 0.44, 0.62]
    _box(ax, ox[0], y, bw, bh, "frame$_t$",    color=BOX_FRAME)
    _box(ax, ox[1], y, bw, bh, "Online\nEncoder", color=BOX_ENC)
    _box(ax, ox[2], y, bw, bh, "Predictor",    color=BOX_PRED)
    _box(ax, ox[3], y, bw, bh, "z$_{pred}$",   color=BOX_PRED, fontsize=10)

    _arrow(ax, ox[0]+bw/2, y, ox[1]-bw/2, y)
    _arrow(ax, ox[1]+bw/2, y, ox[2]-bw/2, y, "(B, 64, D)")
    _arrow(ax, ox[2]+bw/2, y, ox[3]-bw/2, y, "(B, 64, D)")

    # Target path
    ty = 0.18
    tx = [0.08, 0.26, 0.62]
    _box(ax, tx[0], ty, bw, bh, "frame$_{t+1}$", color=BOX_FRAME)
    _box(ax, tx[1], ty, bw, bh, "Target\nEncoder", color=BOX_EMA,
         sublabel="EMA copy")
    _box(ax, tx[2], ty, bw, bh, "z$_{target}$",  color=BOX_EMA, fontsize=10)

    _arrow(ax, tx[0]+bw/2, ty, tx[1]-bw/2, ty)
    _arrow(ax, tx[1]+bw/2, ty, tx[2]-bw/2, ty, "(B, 64, D)", color="#FF8F00")

    # EMA update arrow
    ax.annotate("", xy=(ox[1], y - bh/2), xytext=(tx[1], ty + bh/2),
                arrowprops=dict(arrowstyle="<-", color=BOX_EMA, lw=1.4,
                                linestyle="dashed", mutation_scale=12), zorder=2)
    ax.text(0.205, (y - bh/2 + ty + bh/2)/2, "EMA\nupdate",
            ha="right", va="center", color=BOX_EMA, fontsize=8, alpha=0.9)

    # MSE loss between z_pred and z_target
    ax.annotate("", xy=(ox[3], y - bh/2), xytext=(ox[3], ty + bh/2),
                arrowprops=dict(arrowstyle="-|>", color=LOSS_COL, lw=1.8,
                                mutation_scale=14), zorder=2)
    ax.text(ox[3] + 0.05, (y + ty)/2, "MSE + VICReg\n(latent space)",
            ha="left", va="center", color=LOSS_COL, fontsize=9)

    # no_grad label
    ax.text(tx[1], ty - bh/2 - 0.06, "no grad",
            ha="center", color=BOX_EMA, fontsize=8, alpha=0.75)

    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "arch_patch_jepa_stage1.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Slide 3: Patch-JEPA Stage 2 (Decoder training) ────────────────────────────

def patch_jepa_stage2():
    fig, ax = _base_fig(13, 3.8)
    ax.text(0.5, 0.96, "Patch-JEPA — Stage 2: PatchDecoder Training",
            ha="center", va="top", color=TEXT_COL, fontsize=13, fontweight="bold")

    bw, bh = 0.14, 0.28
    y = 0.50
    xs = [0.09, 0.27, 0.45, 0.63, 0.82]

    _box(ax, xs[0], y, bw, bh, "frame$_t$",     color=BOX_FRAME)
    _box(ax, xs[1], y, bw, bh, "Encoder\n(frozen)", color=BOX_ENC)
    _box(ax, xs[2], y, bw, bh, "Predictor\n(frozen)", color=BOX_PRED)
    _box(ax, xs[3], y, bw, bh, "Patch\nDecoder",  color=BOX_DEC)
    _box(ax, xs[4], y, bw, bh, "frame$_{t+1}$\npredicted", color=BOX_FRAME)

    _arrow(ax, xs[0]+bw/2, y, xs[1]-bw/2, y)
    _arrow(ax, xs[1]+bw/2, y, xs[2]-bw/2, y, "z$_{enc}$")
    _arrow(ax, xs[2]+bw/2, y, xs[3]-bw/2, y, "z$_{pred}$\n≈ enc(t+1)")
    _arrow(ax, xs[3]+bw/2, y, xs[4]-bw/2, y)

    # MSE loss
    gt_y = y - bh/2 - 0.18
    ax.annotate("", xy=(xs[4], y - bh/2), xytext=(xs[4], gt_y + 0.04),
                arrowprops=dict(arrowstyle="-", color=LOSS_COL, lw=1.5, linestyle="dashed"))
    ax.annotate("", xy=(xs[3], gt_y + 0.04), xytext=(xs[4], gt_y + 0.04),
                arrowprops=dict(arrowstyle="-|>", color=LOSS_COL, lw=1.5, mutation_scale=14))
    ax.text((xs[3]+xs[4])/2, gt_y - 0.01, "MSE loss  vs  frame$_{t+1}$ (GT)",
            ha="center", va="top", color=LOSS_COL, fontsize=9.5)

    ax.text(0.5, 0.08,
            "JEPA property:  predictor(encoder(frame$_t$))  ≈  encoder(frame$_{t+1}$)"
            "   →   decoder learns to invert next-frame embedding",
            ha="center", va="bottom", color=TEXT_COL, fontsize=9, alpha=0.7,
            style="italic")

    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "arch_patch_jepa_stage2.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Slide 4: Patch-JEPA Inference (autoregressive rollout) ────────────────────

def patch_jepa_inference():
    fig, ax = _base_fig(14, 3.6)
    ax.text(0.5, 0.97, "Patch-JEPA — Inference: Autoregressive Rollout",
            ha="center", va="top", color=TEXT_COL, fontsize=13, fontweight="bold")

    bw, bh = 0.10, 0.30
    y = 0.52
    # encode once
    ex = [0.07, 0.20]
    _box(ax, ex[0], y, bw, bh, "frame$_0$",   color=BOX_FRAME)
    _box(ax, ex[1], y, bw, bh, "Encoder",      color=BOX_ENC, fontsize=9)
    _arrow(ax, ex[0]+bw/2, y, ex[1]-bw/2, y)
    _arrow(ax, ex[1]+bw/2, y, 0.32, y, "z$_0$")

    # rolling steps
    steps = [0.34, 0.52, 0.70]
    for i, sx in enumerate(steps):
        _box(ax, sx,       y, 0.10, bh, "Predictor", color=BOX_PRED, fontsize=9)
        _box(ax, sx+0.115, y, 0.10, bh, "Decoder",   color=BOX_DEC,  fontsize=9)
        if i < len(steps)-1:
            _arrow(ax, sx+0.165, y, steps[i+1]-0.05, y, f"z$_{i+2}$")
        # frame output below
        ax.text(sx+0.115, y - bh/2 - 0.10, f"frame$_{i+1}$",
                ha="center", color=TEXT_COL, fontsize=8.5, alpha=0.8)
        ax.annotate("", xy=(sx+0.115, y - bh/2 - 0.06), xytext=(sx+0.115, y - bh/2),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_COL, lw=1.2,
                                    mutation_scale=12), zorder=2)
        if i > 0:
            _arrow(ax, steps[i-1]+0.165, y, sx-0.05, y, f"z$_{i+1}$")

    ax.text(steps[-1]+0.165+0.07, y, "…", ha="left", va="center",
            color=TEXT_COL, fontsize=18)

    ax.text(0.5, 0.10, "Encoder runs once  —  only Predictor + Decoder loop at each step",
            ha="center", color=TEXT_COL, fontsize=9, alpha=0.65, style="italic")

    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "arch_patch_jepa_inference.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating architecture diagrams...")
    pixel_architecture()
    patch_jepa_stage1()
    patch_jepa_stage2()
    patch_jepa_inference()
    print(f"\nAll saved to {OUT_DIR}/")
