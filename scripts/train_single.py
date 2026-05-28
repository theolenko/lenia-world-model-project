"""Train a single Lenia world model (pixel-prediction or JEPA).

Usage:
    python scripts/train_single.py --setup pixel
    python scripts/train_single.py --setup jepa
    python scripts/train_single.py --setup pixel --dummy        # synthetic data, no disk I/O
    python scripts/train_single.py --setup jepa --config path/to/config.yaml
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
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Make the project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import JEPAWorldModel, PixelWorldModel
from training.trainer import Trainer

SEED = 42

def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LeniaDataset(Dataset):
    """Yields consecutive (frame_t, frame_t_plus_1) pairs from Lenia trajectories.

    Supports .npy files with shape:
        (num_trajectories, T, H, W)        — no channel dim
        (num_trajectories, T, 1, H, W)     — with channel dim
        (T, H, W)                           — single trajectory, no channel
        (T, 1, H, W)                        — single trajectory, with channel

    Also supports .h5 / .hdf5 files with a 'frames' dataset of shape (N, T, H, W).
    HDF5 files are read lazily (one frame pair per __getitem__) so that large
    datasets (e.g. 1800 × 200 × 64 × 64 ≈ 5.5 GB) don't need to fit in RAM.

    Frames are normalised to [0, 1] (divided by their maximum if > 1).
    """

    def __init__(self, data_path: str) -> None:
        self._path = data_path
        self._h5file = None   # opened lazily per-worker; never set before fork

        if data_path.endswith(".h5") or data_path.endswith(".hdf5"):
            import h5py
            with h5py.File(data_path, "r") as f:
                shape = f["frames"].shape
                self._needs_norm = float(f["frames"][0, 0].max()) > 1.0
            if len(shape) != 4:
                raise ValueError(
                    f"HDF5 'frames' must be 4-D (N, T, H, W); got shape {shape}"
                )
            N, T, H, W = shape
            self._is_h5 = True
        else:
            data = self._load_npy(data_path)
            self._is_h5 = False
            N, T, C, H, W = data.shape
            self._data = data

        self.index: list[tuple[int, int]] = [
            (n, t) for n in range(N) for t in range(T - 1)
        ]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5file"] = None   # file handles are not picklable
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    @staticmethod
    def _load_npy(path: str) -> np.ndarray:
        data = np.load(path).astype(np.float32)
        if data.max() > 1.0:
            data /= data.max()

        if data.ndim == 3:
            data = data[:, None, :, :][None]
        elif data.ndim == 4:
            if data.shape[1] == 1:
                data = data[None]
            else:
                data = data[:, :, None, :, :]
        elif data.ndim == 5:
            pass
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        return data

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        n, t = self.index[idx]

        if self._is_h5:
            if self._h5file is None:
                import h5py
                self._h5file = h5py.File(self._path, "r")
            ft  = self._h5file["frames"][n, t    ].astype(np.float32)
            ft1 = self._h5file["frames"][n, t + 1].astype(np.float32)
            if self._needs_norm:
                denom = max(ft.max(), ft1.max(), 1e-8)
                ft  /= denom
                ft1 /= denom
            frame_t       = torch.from_numpy(ft ).unsqueeze(0)
            frame_t_plus_1 = torch.from_numpy(ft1).unsqueeze(0)
        else:
            frame_t        = torch.from_numpy(self._data[n, t    ])
            frame_t_plus_1 = torch.from_numpy(self._data[n, t + 1])

        return frame_t, frame_t_plus_1

def _make_dummy_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Create in-memory random dataloaders for smoke-testing without real data.

    Generates 200 training pairs and 40 validation pairs of shape (1, 64, 64).
    """
    def _rand_ds(n: int) -> TensorDataset:
        return TensorDataset(
            torch.rand(n, 1, 64, 64),
            torch.rand(n, 1, 64, 64),
        )

    train_loader = DataLoader(_rand_ds(200), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(_rand_ds(40), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def build_dataloaders(
    cfg: dict,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    dummy: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from config, with optional overrides.

    Args:
        cfg: Full config dict.
        batch_size: Overrides cfg["data"]["batch_size"] when provided.
        num_workers: Overrides cfg["data"]["num_workers"] when provided.
        dummy: If True, return in-memory random loaders (no disk I/O).

    Returns:
        (train_loader, val_loader)
    """
    bs = batch_size if batch_size is not None else cfg["data"]["batch_size"]

    if dummy:
        return _make_dummy_loaders(bs)

    nw = num_workers if num_workers is not None else cfg["data"]["num_workers"]
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        LeniaDataset(cfg["data"]["train_path"]),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        LeniaDataset(cfg["data"]["val_path"]),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )
    return train_loader, val_loader

def train_trial(
    setup_type: str,
    cfg: dict,
    overrides: Optional[dict] = None,
    log_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    seed: int = SEED,
    trial=None,
    dummy: bool = False,
) -> float:
    """Train one model configuration and return the final validation loss.

    This is the shared entry point used by both ``train_single.py`` (full runs)
    and ``run_hpo.py`` (short Optuna trials).

    Args:
        setup_type: "pixel" or "jepa".
        cfg: Base configuration dict loaded from config.yaml.
        overrides: Flat dict of hyperparameter overrides.  Recognised keys:
            embed_dim, learning_rate, weight_decay, batch_size,
            ema_momentum, predictor_hidden_dim,
            use_variance_reg, use_covariance_reg, var_reg_weight, cov_reg_weight.
        log_dir: Output directory for checkpoints and TensorBoard logs.
            Defaults to ``experiments/{setup_type}_{timestamp}``.
        num_epochs: Number of training epochs.  Defaults to
            cfg["training"]["num_epochs"].
        seed: Random seed.  Keeping this fixed across both setups ensures
            identical encoder initialisation — critical for fair comparison.
        trial: Optional ``optuna.Trial`` instance.  When provided, intermediate
            validation losses are reported for pruning.
        dummy: If True, use in-memory random data instead of loading from disk.

    Returns:
        Final validation loss (float).
    """
    ov = overrides or {}

    embed_dim: int = ov.get("embed_dim", cfg["model"]["embed_dim"])
    learning_rate: float = ov.get("learning_rate", cfg["training"]["learning_rate"])
    weight_decay: float = ov.get("weight_decay", cfg["training"]["weight_decay"])
    batch_size: int = ov.get("batch_size", cfg["data"]["batch_size"])
    ema_momentum: float = ov.get("ema_momentum", cfg["model"]["ema_momentum"])
    predictor_hidden_dim: int = ov.get(
        "predictor_hidden_dim", cfg["model"]["predictor_hidden_dim"]
    )
    jepa_cfg = cfg.get("jepa", {})
    use_variance_reg: bool = ov.get(
        "use_variance_reg", jepa_cfg.get("use_variance_reg", False)
    )
    use_covariance_reg: bool = ov.get(
        "use_covariance_reg", jepa_cfg.get("use_covariance_reg", False)
    )
    var_reg_weight: float = ov.get("var_reg_weight", jepa_cfg.get("var_reg_weight", 1.0))
    cov_reg_weight: float = ov.get("cov_reg_weight", jepa_cfg.get("cov_reg_weight", 0.04))
    epochs: int = num_epochs if num_epochs is not None else cfg["training"]["num_epochs"]

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg, batch_size=batch_size, dummy=dummy)

    if setup_type == "pixel":
        model = PixelWorldModel(embed_dim=embed_dim)
    else:
        model = JEPAWorldModel(
            embed_dim=embed_dim,
            ema_momentum=ema_momentum,
            predictor_hidden_dim=predictor_hidden_dim,
        )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {type(model).__name__}  trainable params: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = str(Path("experiments") / f"{setup_type}_{timestamp}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        setup_type=setup_type,
        use_variance_reg=use_variance_reg,
        use_covariance_reg=use_covariance_reg,
        var_reg_weight=var_reg_weight,
        cov_reg_weight=cov_reg_weight,
        log_dir=log_dir,
        log_interval=cfg["training"]["log_interval"],
        checkpoint_interval=cfg["training"]["checkpoint_interval"],
    )

    print(f"Starting {setup_type} training for {epochs} epochs  →  {log_dir}")
    val_loss = 0.0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate(epoch)
        elapsed = time.time() - t0
        print(
            f"  Epoch [{epoch + 1}/{epochs}]  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  ({elapsed:.1f}s)"
        )

        # Optuna: report intermediate value so the study can prune bad trials early
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                trainer.writer.close()
                import optuna
                raise optuna.exceptions.TrialPruned()

    trainer._save_checkpoint(epochs, val_loss)
    trainer.writer.close()
    return val_loss

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Lenia world model.")
    parser.add_argument(
        "--setup",
        required=True,
        choices=["pixel", "jepa"],
        help="Training paradigm: 'pixel' for pixel-prediction, 'jepa' for JEPA.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config file. Default: config.yaml",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use randomly generated in-memory data instead of loading from disk. "
             "Useful for a quick smoke-test without real Lenia data.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")
    if args.dummy:
        print("Running with dummy (randomly generated) data.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str(Path("experiments") / f"{args.setup}_{timestamp}")

    train_trial(
        setup_type=args.setup,
        cfg=cfg,
        log_dir=log_dir,
        dummy=args.dummy,
    )

if __name__ == "__main__":
    main()
