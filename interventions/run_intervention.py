"""Setup A (pixel model) state intervention: edit a frame mid-rollout and check
whether the model's continuation matches the real Lenia rule's continuation.

Mechanics: the model is a single-step Markov map (frame_t -> frame_t+1, no
hidden state — see PixelWorldModel.forward), and so is Lenia itself. So
intervening means picking a step t*, editing that frame directly, and feeding
the edited frame into both the model's next forward pass and a fresh
ground-truth Lenia rollout — then comparing the two continuations.

Usage:
    python -m interventions.run_intervention --intervention inject_blob --t-star 20 --n-steps 30
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interventions.metrics import mse_per_step, time_to_divergence
from interventions.perturbations import INTERVENTIONS
from models.world_model import PixelWorldModel
from utils.lenia_core import rollout as lenia_rollout

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a state intervention on Setup A.")
    parser.add_argument("--checkpoint", default="checkpoints/pixel_v1.pt")
    parser.add_argument(
        "--config", default="config_cluster_pixel.yaml",
        help="Must match the embed_dim the checkpoint was trained with — "
             "pixel_v1.pt was trained via config_cluster_pixel.yaml (embed_dim=256), "
             "not the local config.yaml (embed_dim=128).",
    )
    parser.add_argument("--data", default="data_gen/lenia_val.h5")
    parser.add_argument("--traj-idx", type=int, default=0)
    parser.add_argument("--t-star", type=int, default=20, help="Rollout step at which to intervene.")
    parser.add_argument("--n-steps", type=int, default=30, help="Steps to roll out after the intervention.")
    parser.add_argument(
        "--intervention", required=True, choices=list(INTERVENTIONS),
        help="Which perturbation to apply at t-star.",
    )
    parser.add_argument("--divergence-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def model_rollout(model: PixelWorldModel, start_frame: np.ndarray, n_steps: int, device: torch.device) -> np.ndarray:
    """Autoregressive rollout: feed the model's own prediction back in as input."""
    current = torch.from_numpy(start_frame).unsqueeze(0).unsqueeze(0).to(device)
    frames = [start_frame]
    with torch.no_grad():
        for _ in range(n_steps):
            current = model(current)
            frames.append(current.squeeze().cpu().numpy())
    return np.stack(frames)


def main() -> None:
    args = parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    model = PixelWorldModel(embed_dim=cfg["model"]["embed_dim"]).to(device)
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with h5py.File(ROOT / args.data, "r") as hf:
        traj = hf["frames"][args.traj_idx].astype(np.float32)  # (T, H, W)

    # Run the model up to t_star on its own predictions (not ground truth) —
    # the intervened frame should be one the model itself "believes in".
    pre_rollout = model_rollout(model, traj[0], args.t_star, device)
    pre_frame = pre_rollout[-1]

    intervened_frame = INTERVENTIONS[args.intervention](pre_frame, rng=rng)

    model_post = model_rollout(model, intervened_frame, args.n_steps, device)
    gt_post = lenia_rollout(intervened_frame, args.n_steps)

    # Index 0 of both is the intervened frame itself (identical by construction) — skip it.
    per_step_mse = mse_per_step(model_post[1:], gt_post[1:])
    divergence_step = time_to_divergence(model_post[1:], gt_post[1:], args.divergence_threshold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / f"intervention_{args.intervention}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, args.n_steps + 1), per_step_mse, "o-", color="tomato")
    if divergence_step is not None:
        ax.axvline(divergence_step + 1, color="gray", ls="--", lw=0.8,
                   label=f"diverges at step {divergence_step + 1}")
        ax.legend()
    ax.set_xlabel("Steps after intervention")
    ax.set_ylabel("MSE (model vs. ground truth)")
    ax.set_title(f"Intervention: {args.intervention}  (t*={args.t_star})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mse_curve.png", dpi=110)
    plt.close(fig)

    vis_steps = sorted(set(min(s, args.n_steps) for s in [0, 1, args.n_steps // 2, args.n_steps]))
    fig, axes = plt.subplots(2, len(vis_steps), figsize=(3 * len(vis_steps), 6))
    for col, s in enumerate(vis_steps):
        axes[0, col].imshow(model_post[s], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0, col].set_title(f"Model  step {s}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(gt_post[s], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, col].set_title(f"Ground truth  step {s}", fontsize=8)
        axes[1, col].axis("off")
    fig.suptitle(f"Post-intervention rollout ({args.intervention}, t*={args.t_star})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "frames.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"Intervention: {args.intervention}  |  t*={args.t_star}  |  traj={args.traj_idx}")
    print(f"MSE step 1 : {per_step_mse[0]:.6f}")
    print(f"MSE step {args.n_steps:>2} : {per_step_mse[-1]:.6f}")
    print(f"Diverges (>{args.divergence_threshold}) at step: {divergence_step + 1 if divergence_step is not None else 'never'}")
    print(f"Saved plots to: {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
