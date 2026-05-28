"""Re-chunk Lenia HDF5 files from per-trajectory to per-frame chunks.

The original files use chunk layout (1, 200, 64, 64) — one 3.12 MB chunk per
trajectory.  Random per-frame access decompresses the full chunk every time,
making shuffled DataLoader access ~18 min/epoch on CPU.

Re-chunking to (1, 1, 64, 64) decompresses only the 16 KB frame that is
actually needed, giving a >10x speedup for random access.

Usage:
    python scripts/rechunk_data.py --input data/lenia_train.h5 --output data/lenia_train_chunked.h5
    python scripts/rechunk_data.py --input data/lenia_val.h5   --output data/lenia_val_chunked.h5
    python scripts/rechunk_data.py --input data/lenia_train.h5 --output data/lenia_train_nocomp.h5 --compression none
"""
import argparse
import random
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

SEED = 42
N_VERIFY_SAMPLES = 5


def rechunk(
    input_path: str,
    output_path: str,
    compression: str | None,
) -> dict:
    """Copy an HDF5 file with a new chunk layout, one trajectory at a time."""
    src_path = Path(input_path)
    dst_path = Path(output_path)

    assert src_path.exists(), f"Input not found: {src_path}"
    assert not dst_path.exists(), (
        f"Output already exists: {dst_path}\n"
        "Delete it first or choose a different output path."
    )

    with h5py.File(src_path, "r") as f:
        ds = f["frames"]
        N, T, H, W = ds.shape
        dtype = ds.dtype
        src_chunks = ds.chunks
        src_comp = ds.compression

    new_chunks = (1, 1, H, W)
    comp_kwargs = {"compression": "lzf"} if compression == "lzf" else {}

    print(f"\nInput  : {src_path.name}  ({src_path.stat().st_size / 1e9:.3f} GB)")
    print(f"Shape  : {(N, T, H, W)}  dtype={dtype}")
    print(f"Chunks : {src_chunks}  compression={src_comp}")
    print(f"\nOutput : {dst_path}")
    print(f"Chunks : {new_chunks}  compression={compression or 'none'}")

    t0 = time.time()

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        src_ds = src["frames"]
        dst_ds = dst.create_dataset(
            "frames",
            shape=(N, T, H, W),
            dtype=dtype,
            chunks=new_chunks,
            **comp_kwargs,
        )

        # Copy one trajectory at a time (~3 MB RAM per iteration)
        for n in tqdm(range(N), desc="Rechunking", unit="traj"):
            dst_ds[n] = src_ds[n]

    elapsed = time.time() - t0
    in_size  = src_path.stat().st_size
    out_size = dst_path.stat().st_size

    print(
        f"\nDone in {elapsed:.1f}s  |  "
        f"{in_size / 1e9:.3f} GB → {out_size / 1e9:.3f} GB  "
        f"(ratio {out_size / in_size:.3f}×)"
    )

    return {
        "input_path": str(src_path),
        "output_path": str(dst_path),
        "input_size_gb": in_size / 1e9,
        "output_size_gb": out_size / 1e9,
        "size_ratio": out_size / in_size,
        "elapsed_s": elapsed,
    }


def verify(
    input_path: str,
    output_path: str,
    n_samples: int = N_VERIFY_SAMPLES,
) -> None:
    """Assert that n_samples random frames are bit-identical in both files."""
    rng = random.Random(SEED)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "r") as dst:
        N, T = src["frames"].shape[:2]
        samples = [(rng.randint(0, N - 1), rng.randint(0, T - 1)) for _ in range(n_samples)]

        print(f"\nVerifying {n_samples} random frames (seed={SEED})…")
        for n, t in samples:
            a = src["frames"][n, t]
            b = dst["frames"][n, t]
            if not np.array_equal(a, b):
                max_diff = float(np.abs(a - b).max())
                raise AssertionError(
                    f"Value mismatch at traj={n}, t={t}! max|diff|={max_diff:.3e}"
                )
            print(f"  [OK] traj={n:4d}  t={t:3d}  "
                  f"max={a.max():.4f}  mean={a.mean():.4f}")

    print(f"All {n_samples} frames match — data integrity verified. ✓")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-chunk Lenia HDF5 to per-frame (1,1,64,64) chunks."
    )
    p.add_argument("--input",  required=True, help="Source .h5 file path")
    p.add_argument("--output", required=True, help="Destination .h5 file path (must not exist)")
    p.add_argument(
        "--compression",
        choices=["lzf", "none"],
        default="lzf",
        help="Compression codec for output chunks. Default: lzf",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    comp = None if args.compression == "none" else args.compression
    rechunk(args.input, args.output, comp)
    verify(args.input, args.output)


if __name__ == "__main__":
    main()
