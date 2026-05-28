"""Benchmark HDF5 DataLoader throughput for different chunk layouts.

Measures the average milliseconds-per-batch for N batches with shuffle=True,
comparing the original per-trajectory chunks against the re-chunked
per-frame layout (and optionally a no-compression variant).

Usage:
    python scripts/benchmark_loading.py \\
        --old  data/lenia_train.h5 \\
        --new  data/lenia_train_chunked.h5 \\
        [--nocomp data/lenia_train_nocomp.h5] \\
        [--n-batches 100] \\
        [--batch-size 32] \\
        [--num-workers 0]
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_single import LeniaDataset


def _chunk_info(path: str) -> str:
    with h5py.File(path, "r") as f:
        chunks = f["frames"].chunks
        comp   = f["frames"].compression or "none"
        return f"chunks={chunks}  compression={comp}"


def benchmark_one(
    path: str,
    n_batches: int,
    batch_size: int,
    num_workers: int,
) -> dict:
    """Return timing stats for n_batches batches from path."""
    info = _chunk_info(path)
    print(f"\n  File    : {Path(path).name}")
    print(f"  Layout  : {info}")

    # Fresh dataset so the h5 file is not pre-warmed from a previous call
    ds = LeniaDataset(path)
    print(f"  Pairs   : {len(ds):,}  batch_size={batch_size}  workers={num_workers}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # One warm-up batch so OS page cache is primed and h5py file is open
    it = iter(loader)
    next(it)

    # Timed run: iterate through at most n_batches more batches
    t0      = time.perf_counter()
    counted = 0
    for _ in loader:
        counted += 1
        if counted >= n_batches:
            break
    elapsed = time.perf_counter() - t0

    ms_per_batch = elapsed / counted * 1000
    print(f"  Time    : {elapsed:.2f}s / {counted} batches  →  {ms_per_batch:.1f} ms/batch")

    return {
        "path": path,
        "info": info,
        "n_batches": counted,
        "elapsed_s": elapsed,
        "ms_per_batch": ms_per_batch,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark HDF5 DataLoader throughput.")
    p.add_argument("--old",         required=True,       help="Original .h5 (large chunks)")
    p.add_argument("--new",         required=True,       help="Re-chunked .h5 (frame-level)")
    p.add_argument("--nocomp",      default=None,        help="No-compression re-chunked .h5 (optional)")
    p.add_argument("--n-batches",   type=int, default=100, help="Batches to time per file. Default: 100")
    p.add_argument("--batch-size",  type=int, default=32,  help="Batch size. Default: 32")
    p.add_argument("--num-workers", type=int, default=0,   help="DataLoader workers. Default: 0")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  HDF5 DataLoader Loading Benchmark")
    print(f"  n_batches={args.n_batches}  batch_size={args.batch_size}  "
          f"num_workers={args.num_workers}")
    print("=" * 60)

    results: dict[str, dict] = {}

    print("\n[1] Original (per-trajectory chunks):")
    results["old"] = benchmark_one(
        args.old, args.n_batches, args.batch_size, args.num_workers
    )

    print("\n[2] Re-chunked (per-frame, LZF):")
    results["new"] = benchmark_one(
        args.new, args.n_batches, args.batch_size, args.num_workers
    )

    if args.nocomp and Path(args.nocomp).exists():
        print("\n[3] Re-chunked (per-frame, no compression):")
        results["nocomp"] = benchmark_one(
            args.nocomp, args.n_batches, args.batch_size, args.num_workers
        )

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    old_ms = results["old"]["ms_per_batch"]
    new_ms = results["new"]["ms_per_batch"]
    speedup_new = old_ms / new_ms

    print(f"  [old]    {old_ms:7.1f} ms/batch  (baseline)")
    print(f"  [new]    {new_ms:7.1f} ms/batch  ({speedup_new:.1f}× faster)")

    if "nocomp" in results:
        nc_ms      = results["nocomp"]["ms_per_batch"]
        speedup_nc = old_ms / nc_ms
        print(f"  [nocomp] {nc_ms:7.1f} ms/batch  ({speedup_nc:.1f}× faster)")

    print()
    if speedup_new >= 5:
        print(f"  ✓ {speedup_new:.1f}× speedup — rechunking is effective.")
    else:
        print(f"  ⚠  Only {speedup_new:.1f}× speedup. Consider no-compression or "
              "a trajectory-batch sampler as alternatives.")


if __name__ == "__main__":
    main()
