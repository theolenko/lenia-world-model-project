"""Core Lenia physics — the single source of truth for the update rule.

Shared by data_gen/data_gen_grey_scale.ipynb (dataset generation) and
interventions/ (ground-truth rollouts from intervened states), so the kernel
and growth-function math only need to be tuned in one place.

Update rule: A_{t+1} = clip(A_t + (1/T) * G(K*A_t), 0, 1)
where K is the kernel (neighbourhood weighting) and G is the growth function.
"""
import numpy as np

SIZE = 64
R = 13       # kernel radius in px — larger = farther-reaching neighbourhood,
             # slower/bigger structures. 10-20 is sensible on a 64x64 grid.
T_SIM = 10.0  # time resolution, dt = 1/T_SIM per step. Larger = finer steps,
              # more stable dynamics, but slower evolution per frame.
M = 0.135    # growth bell center — neighbourhood sums near M grow, far from it shrink.
S = 0.015    # growth bell width — small = sharp selection, large = tolerant/blurrier dynamics.
M_K = 0.5    # kernel bell peak position as a fraction of R. 0.5 = ring kernel, 0 = disc kernel.
S_K = 0.15   # kernel bell width — small = thin ring, large = broad ring.


def bell(x: np.ndarray, m: float, s: float) -> np.ndarray:
    return np.exp(-((x - m) / s) ** 2 / 2)


def make_fK(R: float = R, size: int = SIZE, m_k: float = M_K, s_k: float = S_K):
    """Build the real-space kernel K and its FFT (the second return value is
    what lenia_step actually consumes)."""
    mid = size // 2
    D = np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / R
    K = (D < 1) * bell(D, m_k, s_k)
    K /= K.sum()
    return K, np.fft.rfft2(np.fft.fftshift(K))


def lenia_step(A: np.ndarray, fK: np.ndarray, m: float = M, s: float = S) -> np.ndarray:
    """Advance grid A by one Lenia step under kernel fK and growth params (m, s)."""
    U = np.fft.irfft2(fK * np.fft.rfft2(A), s=A.shape)
    return np.clip(A + (1.0 / T_SIM) * (bell(U, m, s) * 2 - 1), 0, 1)


def rollout(A0: np.ndarray, n_steps: int, fK: np.ndarray = None) -> np.ndarray:
    """Roll the simulator forward n_steps from an arbitrary starting grid A0.

    Used to get the ground-truth continuation from a state that was never part
    of the original dataset (e.g. an intervened/edited frame).

    Args:
        A0: Starting grid, shape (H, W), values in [0, 1].
        n_steps: Number of steps to simulate.
        fK: Precomputed FFT kernel from make_fK(). Built with defaults if omitted.

    Returns:
        Array of shape (n_steps + 1, H, W) — A0 followed by each subsequent step.
    """
    if fK is None:
        _, fK = make_fK()
    A = A0.astype(np.float32)
    frames = [A]
    for _ in range(n_steps):
        A = lenia_step(A, fK)
        frames.append(A)
    return np.stack(frames)
