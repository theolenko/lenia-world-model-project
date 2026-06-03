import os
import time
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.world_model import PixelWorldModel, JEPAWorldModel
from training.losses import (
    pixel_prediction_loss,
    jepa_loss,
    variance_regularization,
    covariance_regularization,
)


class Trainer:
    """Handles training for either the pixel-prediction or JEPA setup.

    Args:
        model: PixelWorldModel or JEPAWorldModel instance.
        train_loader: DataLoader yielding (frame_t, frame_t_plus_1) batches.
        val_loader: DataLoader yielding (frame_t, frame_t_plus_1) batches.
        optimizer: PyTorch optimizer.
        device: torch.device to run on.
        setup_type: "pixel" or "jepa".
        use_variance_reg: Whether to add variance regularisation to JEPA loss.
        use_covariance_reg: Whether to add covariance regularisation to JEPA loss.
        var_reg_weight: Weight for variance regularisation term. Default: 1.0.
        cov_reg_weight: Weight for covariance regularisation term. Default: 0.04.
        log_dir: Directory for TensorBoard logs and checkpoints.
        log_interval: Log training loss every this many batches. Default: 50.
        checkpoint_interval: Save checkpoint every this many epochs. Default: 10.
    """

    def __init__(
        self,
        model: Union[PixelWorldModel, JEPAWorldModel],
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        setup_type: str,
        use_variance_reg: bool = False,
        use_covariance_reg: bool = False,
        var_reg_weight: float = 1.0,
        cov_reg_weight: float = 0.04,
        log_dir: str = "experiments/run",
        log_interval: int = 50,
        checkpoint_interval: int = 10,
    ) -> None:
        assert setup_type in ("pixel", "jepa"), "setup_type must be 'pixel' or 'jepa'"
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.setup_type = setup_type
        self.use_variance_reg = use_variance_reg
        self.use_covariance_reg = use_covariance_reg
        self.var_reg_weight = var_reg_weight
        self.cov_reg_weight = cov_reg_weight
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.global_step = 0

    def _compute_loss(
        self, frame_t: torch.Tensor, frame_t_plus_1: torch.Tensor
    ) -> torch.Tensor:
        """Compute the training loss for one batch."""
        frame_t = frame_t.to(self.device)
        frame_t_plus_1 = frame_t_plus_1.to(self.device)

        if self.setup_type == "pixel":
            predicted_frame = self.model(frame_t)
            loss = pixel_prediction_loss(predicted_frame, frame_t_plus_1)

        else:  # jepa
            z_pred, z_target, z_context = self.model(frame_t, frame_t_plus_1)
            loss = jepa_loss(z_pred, z_target)

            if self.use_variance_reg:
                loss = loss + self.var_reg_weight * variance_regularization(z_context)
            if self.use_covariance_reg:
                loss = loss + self.cov_reg_weight * covariance_regularization(z_context)

        return loss

    def train_epoch(self, epoch: int) -> float:
        """Run one full pass over the training set.

        Args:
            epoch: Current epoch index (0-based), used for logging.

        Returns:
            Mean training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (frame_t, frame_t_plus_1) in enumerate(self.train_loader):
            loss = self._compute_loss(frame_t, frame_t_plus_1)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.setup_type == "jepa":
                self.model.update_target()

            total_loss += loss.item()
            self.global_step += 1

            if (batch_idx + 1) % self.log_interval == 0:
                self.writer.add_scalar(
                    "train/loss_step", loss.item(), self.global_step
                )

        mean_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("train/loss_epoch", mean_loss, epoch)
        return mean_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Compute validation loss without gradients.

        Args:
            epoch: Current epoch index, used for logging.

        Returns:
            Mean validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        for frame_t, frame_t_plus_1 in self.val_loader:
            loss = self._compute_loss(frame_t, frame_t_plus_1)
            total_loss += loss.item()

        mean_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("val/loss_epoch", mean_loss, epoch)
        return mean_loss

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: Optional[str] = None) -> None:
        """Save model and optimizer state to disk."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        name = f"checkpoint_{tag}.pt" if tag else f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, self.log_dir / name)

    def train(self, num_epochs: int) -> None:
        """Run the full training loop.

        Args:
            num_epochs: Total number of training epochs.
        """
        print(f"Starting {self.setup_type} training for {num_epochs} epochs")
        print(f"Logging to: {self.log_dir}")

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            elapsed = time.time() - t0

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"({elapsed:.1f}s)"
            )

            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1, val_loss)

        # Always save a final checkpoint
        self._save_checkpoint(num_epochs, val_loss)
        self.writer.close()
        print("Training complete.")
