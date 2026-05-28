"""Step 1: Data validation

Checks both HDF5 files against spec, visualizes sample frames, and computes
the identity baseline loss (MSE between consecutive frames). Run from project root:

    source lenia-wm/bin/activate
    python scripts/check_data.py
"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "lenia_train.h5"
VAL_PATH   = ROOT / "data" / "lenia_val.h5"
OUT_DIR    = ROOT / "experiments" / "smoke_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)



EXPECTED = {
    "ndim":  4,
    "dtype": np.float32,
    "T":     200,
    "H":     64,
    "W":     64,
    "min":   0.0,
    "max":   1.0,
}
TRAIN_N_EXPECTED = 1800
VAL_N_EXPECTED   = 200


def check_file(path: Path, expected_N: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  File : {path.name}  ({path.stat().st_size / 1e6:.1f} MB on disk)")

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"  Keys : {keys}")
        assert "frames" in keys, f"Missing 'frames' key – found: {keys}"

        ds    = f["frames"]
        shape = ds.shape
        dtype = ds.dtype

        print(f"  Shape: {shape}   Dtype: {dtype}")

        ok = True
        def check(label, got, want):
            nonlocal ok
            status = "OK" if got == want else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"    [{status}] {label}: got {got}, expected {want}")

        assert len(shape) == EXPECTED["ndim"], \
            f"Expected {EXPECTED['ndim']}D tensor, got {len(shape)}D"
        N, T, H, W = shape
        check("N_trajectories", N, expected_N)
        check("T_frames",       T, EXPECTED["T"])
        check("H",              H, EXPECTED["H"])
        check("W",              W, EXPECTED["W"])
        check("dtype",          np.dtype(dtype), np.dtype(EXPECTED["dtype"]))

        # Value range – read only a small sample to avoid RAM spike
        sample = ds[:20]       # 20 trajectories
        v_min  = float(sample.min())
        v_max  = float(sample.max())
        v_mean = float(sample.mean())
        print(f"  Values (sample, first 20 trajs): "
              f"min={v_min:.4f}  max={v_max:.4f}  mean={v_mean:.4f}")
        if v_min < EXPECTED["min"] - 1e-4:
            print(f"    [FAIL] min value {v_min:.6f} < 0.0"); ok = False
        if v_max > EXPECTED["max"] + 1e-4:
            print(f"    [FAIL] max value {v_max:.6f} > 1.0"); ok = False
        if ok:
            print("  Spec check: PASSED")
        else:
            print("  Spec check: FAILED (see above)")

    return {"shape": shape, "dtype": dtype, "min": v_min, "max": v_max, "mean": v_mean}


def visualize_trajectory(path: Path, out_path: Path, traj_idx: int = 0) -> None:
    """Save 6 evenly-spaced frames from one trajectory as a PNG."""
    timesteps = [0, 25, 50, 100, 150, 199]
    with h5py.File(path, "r") as f:
        traj = f["frames"][traj_idx]            # (T, H, W)

    fig, axes = plt.subplots(1, len(timesteps), figsize=(3 * len(timesteps), 3.2))
    for ax, t in zip(axes, timesteps):
        ax.imshow(traj[t], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"t={t}", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{path.name}  –  trajectory {traj_idx}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(ROOT)}")


def compute_identity_baseline(path: Path, n_trajs: int = 50) -> float:
    """Mean MSE(frame_t, frame_{t+1}) over the first n_trajs trajectories."""
    total_mse = 0.0
    n = 0
    with h5py.File(path, "r") as f:
        ds  = f["frames"]
        lim = min(n_trajs, ds.shape[0])
        for i in range(lim):
            traj = ds[i].astype(np.float32)          # (T, H, W)
            ft   = torch.from_numpy(traj[:-1])        # (T-1, H, W)
            ft1  = torch.from_numpy(traj[1:])         # (T-1, H, W)
            total_mse += F.mse_loss(ft, ft1).item()
            n += 1
    return total_mse / n


def main() -> None:
    print("=" * 60)
    print("  Lenia Data Validation")
    print("=" * 60)

    train_info = check_file(TRAIN_PATH, TRAIN_N_EXPECTED)
    val_info   = check_file(VAL_PATH,   VAL_N_EXPECTED)

    print(f"\n--- Trajectory visualisations ---")
    visualize_trajectory(TRAIN_PATH, OUT_DIR / "check_train_traj0.png", traj_idx=0)
    visualize_trajectory(VAL_PATH,   OUT_DIR / "check_val_traj0.png",   traj_idx=0)

    print(f"\n--- Identity baseline loss (first 50 trajectories each) ---")
    train_baseline = compute_identity_baseline(TRAIN_PATH, n_trajs=50)
    val_baseline   = compute_identity_baseline(VAL_PATH,   n_trajs=50)
    print(f"  Train identity MSE : {train_baseline:.6f}")
    print(f"  Val   identity MSE : {val_baseline:.6f}")
    print(f"  => A model with loss > {train_baseline:.6f} is no better than copying the input")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  train shape : {train_info['shape']}  min={train_info['min']:.4f}  max={train_info['max']:.4f}")
    print(f"  val   shape : {val_info['shape']}   min={val_info['min']:.4f}  max={val_info['max']:.4f}")
    print(f"  identity baseline  train={train_baseline:.6f}  val={val_baseline:.6f}")
    print(f"  visualisations saved to: experiments/smoke_test/")


if __name__ == "__main__":
    main()
