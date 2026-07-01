"""Hand-crafted edits applied to a single Lenia frame to create an intervened
state — i.e. a do(s_t) operation, applied before continuing a rollout.

Each function takes a (H, W) float32 array in [0, 1] and returns an edited
copy of the same shape/range.
"""
import numpy as np


def inject_blob(frame: np.ndarray, center=None, radius=6, density=0.6, rng=None) -> np.ndarray:
    """Paste a new Gaussian-blob organism seed into the frame."""
    rng = rng or np.random.default_rng()
    size = frame.shape[-1]
    cx, cy = center if center is not None else rng.integers(8, size - 8, size=2)
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    blob = density * np.exp(-dist**2 / (2 * radius**2))
    return np.clip(frame + blob, 0, 1).astype(np.float32)


def zero_region(frame: np.ndarray, center=None, radius=10, rng=None) -> np.ndarray:
    """Kill an existing region — zero out a circular patch."""
    rng = rng or np.random.default_rng()
    size = frame.shape[-1]
    cx, cy = center if center is not None else rng.integers(8, size - 8, size=2)
    y, x = np.ogrid[:size, :size]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    out = frame.copy()
    out[mask] = 0.0
    return out.astype(np.float32)


def mirror_patch(frame: np.ndarray, center=None, half_size=12, rng=None) -> np.ndarray:
    """Mirror a square patch of the frame left-right, in place."""
    rng = rng or np.random.default_rng()
    size = frame.shape[-1]
    cx, cy = center if center is not None else rng.integers(half_size, size - half_size, size=2)
    x0, x1 = cx - half_size, cx + half_size
    y0, y1 = cy - half_size, cy + half_size
    out = frame.copy()
    out[y0:y1, x0:x1] = frame[y0:y1, x0:x1][:, ::-1]
    return out.astype(np.float32)


def scale_density(frame: np.ndarray, factor=1.5) -> np.ndarray:
    """Globally scale the frame's density up (factor>1) or down (factor<1)."""
    return np.clip(frame * factor, 0, 1).astype(np.float32)


def add_noise(frame: np.ndarray, sigma=0.05, rng=None) -> np.ndarray:
    """Perturb every cell with Gaussian noise."""
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, sigma, size=frame.shape).astype(np.float32)
    return np.clip(frame + noise, 0, 1).astype(np.float32)


INTERVENTIONS = {
    "inject_blob": inject_blob,
    "zero_region": zero_region,
    "mirror_patch": mirror_patch,
    "scale_density": scale_density,
    "add_noise": add_noise,
}
