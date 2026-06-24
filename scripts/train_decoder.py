"""Train a spatial reconstruction decoder on top of a frozen JEPA encoder.

Uses the pre-GAP spatial feature map (B, 256, 4, 4) from LeniaEncoder instead
of the pooled flat embedding. This preserves spatial structure that
AdaptiveAvgPool discards, enabling proper frame reconstruction.

Pipeline:
    frame_t → encoder.conv_blocks → (B,256,4,4) → SpatialLeniaDecoder → frame_t

The encoder's GAP + projection layers are bypassed entirely for this task.

Usage:
    python scripts/train_decoder.py --jepa-checkpoint path/to/checkpoint_best.pt
    python scripts/train_decoder.py --jepa-checkpoint path/to/checkpoint_best.pt --config config_cluster_decoder_v2.yaml
    python scripts/train_decoder.py --jepa-checkpoint path/to/checkpoint_best.pt --dummy
"""
import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decoder import SpatialLeniaDecoder
from models.world_model import JEPAWorldModel
from scripts.train_single import build_dataloaders

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_frozen_encoder(jepa_ckpt_path: str, embed_dim: int, device: torch.device) -> nn.Module:
    """Load the online encoder from a JEPA checkpoint and freeze all parameters."""
    ckpt = torch.load(jepa_ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # Infer predictor_hidden_dim from saved weights to avoid shape mismatch
    predictor_hidden_dim = state["predictor.net.0.weight"].shape[0]

    jepa = JEPAWorldModel(
        embed_dim=embed_dim,
        predictor_hidden_dim=predictor_hidden_dim,
        encoder_type="cnn",
    )
    jepa.load_state_dict(state)

    encoder = jepa.online_encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    return encoder


def train_decoder(
    cfg: dict,
    jepa_ckpt_path: str,
    log_dir: Optional[str] = None,
    dummy: bool = False,
) -> float:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embed_dim: int = cfg["model"]["embed_dim"]
    lr: float = cfg["training"]["learning_rate"]
    weight_decay: float = cfg["training"]["weight_decay"]
    epochs: int = cfg["training"]["num_epochs"]
    patience: int = cfg["training"].get("early_stopping_patience", 0)
    use_lr_schedule: bool = cfg["training"].get("use_lr_schedule", True)

    train_loader, val_loader = build_dataloaders(cfg, dummy=dummy)

    print(f"Loading frozen JEPA encoder from: {jepa_ckpt_path}")
    encoder = load_frozen_encoder(jepa_ckpt_path, embed_dim, device)

    # SpatialLeniaDecoder takes (B, 256, 4, 4) pre-GAP features directly,
    # preserving spatial structure that the flat embedding discards.
    decoder = SpatialLeniaDecoder().to(device)
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"SpatialLeniaDecoder trainable params: {num_params:,}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_lr_schedule else None
    )
    criterion = nn.MSELoss()

    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = str(Path("experiments") / f"decoder_{timestamp}")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_path))

    print(f"Starting decoder training for {epochs} epochs  →  {log_dir}")
    if patience > 0:
        print(f"Early stopping patience: {patience}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    val_loss = 0.0
    global_step = 0
    log_interval: int = cfg["training"].get("log_interval", 50)

    for epoch in range(epochs):
        t0 = time.time()

        # --- train ---
        decoder.train()
        total_train = 0.0
        for batch_idx, (frame_t, _) in enumerate(train_loader):
            frame_t = frame_t.to(device)

            with torch.no_grad():
                spatial = encoder(frame_t, return_spatial=True)

            recon = decoder(spatial)
            loss = criterion(recon, frame_t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_train += loss.item()
            global_step += 1
            if (batch_idx + 1) % log_interval == 0:
                writer.add_scalar("train/loss_step", loss.item(), global_step)

        train_loss = total_train / len(train_loader)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)

        # --- validate ---
        decoder.eval()
        total_val = 0.0
        with torch.no_grad():
            for frame_t, _ in val_loader:
                frame_t = frame_t.to(device)
                spatial = encoder(frame_t, return_spatial=True)
                recon = decoder(spatial)
                total_val += criterion(recon, frame_t).item()

        val_loss = total_val / len(val_loader)
        writer.add_scalar("val/loss_epoch", val_loss, epoch)

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        elapsed = time.time() - t0
        print(
            f"  Epoch [{epoch + 1}/{epochs}]  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  ({elapsed:.1f}s)"
        )

        # checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            _save(log_path, decoder, optimizer, epoch + 1, val_loss, tag="best")
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    _save(log_path, decoder, optimizer, epoch + 1, val_loss, tag="last")
    writer.close()
    print(f"Best val loss: {best_val_loss:.6f}")
    return best_val_loss


def _save(
    log_path: Path,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    tag: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        log_path / f"checkpoint_{tag}.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JEPA reconstruction decoder.")
    parser.add_argument(
        "--jepa-checkpoint",
        required=True,
        help="Path to JEPA checkpoint_best.pt from a completed JEPA training run.",
    )
    parser.add_argument(
        "--config",
        default="config_cluster_decoder_v2.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use random in-memory data (smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.dummy:
        print("Running with dummy (randomly generated) data.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str(Path("experiments") / f"decoder_{timestamp}")

    train_decoder(
        cfg=cfg,
        jepa_ckpt_path=args.jepa_checkpoint,
        log_dir=log_dir,
        dummy=args.dummy,
    )


if __name__ == "__main__":
    main()
