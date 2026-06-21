"""Comparison metrics between a model's post-intervention rollout and the
ground-truth Lenia rollout from the same intervened state."""
import numpy as np


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def mass(frame: np.ndarray) -> float:
    return float(frame.mean())


def mse_per_step(preds: np.ndarray, trues: np.ndarray) -> list[float]:
    """preds, trues: (n_steps, H, W) arrays of equal length."""
    return [mse(p, t) for p, t in zip(preds, trues)]


def time_to_divergence(preds: np.ndarray, trues: np.ndarray, threshold: float = 0.05) -> int | None:
    """First step index where MSE(pred, true) exceeds threshold, or None if it never does."""
    for i, (p, t) in enumerate(zip(preds, trues)):
        if mse(p, t) > threshold:
            return i
    return None
