"""Train a PatchDecoder on top of a frozen PatchJEPAWorldModel encoder.

Pipeline:
    frame_t → frozen ViTEncoder(pool=False) → (B, N_patches, embed_dim)
            → PatchDecoder → frame_t (reconstructed)

The encoder is loaded from a patch_jepa checkpoint and frozen.
Only PatchDecoder weights are trained (MSE reconstruction loss).

Usage:
    python scripts/train_patch_decoder.py --jepa-checkpoint checkpoints/patch_jepa_v2.pt
    python scripts/train_patch_decoder.py --jepa-checkpoint checkpoints/patch_jepa_v2.pt --dummy
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
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decoder import PatchDecoder
from models.world_model import PatchJEPAWorldModel
from scripts.train_single import build_dataloaders

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_frozen_encoder(ckpt_path: str, embed_dim: int, device: torch.device):
    """Load frozen online encoder + predictor from a PatchJEPA checkpoint.

    Returns (encoder, predictor) both frozen and in eval mode.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    predictor_hidden_dim = state["predictor.net.0.weight"].shape[0]

    model = PatchJEPAWorldModel(
        embed_dim=embed_dim,
        predictor_hidden_dim=predictor_hidden_dim,
    )
    model.load_state_dict(state)
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model.online_encoder, model.predictor


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
    log_interval: int = cfg["training"].get("log_interval", 50)

    train_loader, val_loader = build_dataloaders(cfg, dummy=dummy)

    print(f"Loading frozen PatchJEPA encoder+predictor from: {jepa_ckpt_path}")
    encoder, predictor = load_frozen_encoder(jepa_ckpt_path, embed_dim, device)

    decoder = PatchDecoder(embed_dim=embed_dim).to(device)
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"PatchDecoder trainable params: {num_params:,}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_lr_schedule else None
    )
    criterion = nn.MSELoss()

    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = str(Path("experiments") / f"patch_decoder_{timestamp}")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_path))

    print(f"Starting PatchDecoder training for {epochs} epochs  →  {log_dir}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    val_loss = 0.0
    global_step = 0

    for epoch in range(epochs):
        t0 = time.time()

        decoder.train()
        total_train = 0.0
        for batch_idx, (frame_t, _) in enumerate(train_loader):
            frame_t = frame_t.to(device)

            with torch.no_grad():
                z_enc  = encoder(frame_t)    # (B, N_patches, embed_dim)
                patches = predictor(z_enc)   # always decode predictor output

            recon = decoder(patches)        # (B, 1, 64, 64)
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

        decoder.eval()
        total_val = 0.0
        with torch.no_grad():
            for frame_t, _ in val_loader:
                frame_t = frame_t.to(device)
                # Validate on predictor embeddings — matches inference distribution
                z_enc = encoder(frame_t)
                patches = predictor(z_enc)
                recon = decoder(patches)
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
    parser = argparse.ArgumentParser(description="Train PatchDecoder on frozen PatchJEPA encoder.")
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--config", default="config_cluster_patch_decoder_v2.yaml")
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str(Path("experiments") / f"patch_decoder_{timestamp}")

    train_decoder(
        cfg=cfg,
        jepa_ckpt_path=args.jepa_checkpoint,
        log_dir=log_dir,
        dummy=args.dummy,
    )


if __name__ == "__main__":
    main()
