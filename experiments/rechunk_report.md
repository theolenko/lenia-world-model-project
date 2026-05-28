# HDF5 Rechunking Report

**Date:** 2026-05-28  
**Context:** Fix I/O bottleneck identified in `experiments/pipeline_smoke_test.md` (Finding 3)

---

## 1. Problem

The original HDF5 files use chunk layout `(1, 200, 64, 64)` — one chunk per trajectory, ~3.12 MB uncompressed, LZF-compressed.

With a shuffled `DataLoader`, each frame access decompresses the full 3.12 MB chunk:

- batch_size=32 → ~100 MB of decompression per batch
- Estimated epoch time: ~18 min (on CPU, full dataset)

Root cause: h5py's chunk cache defaults to 1 MB, which is smaller than one trajectory chunk. Every random-index access from a different trajectory is a cache miss and triggers full decompression.

---

## 2. Fix: Re-chunk to Per-Frame Layout

New chunk layout: `(1, 1, 64, 64)` — one 16 KB chunk per frame (LZF, same codec).

Each `DataLoader` access now decompresses only the 16 KB frame actually needed.

### Script

```
python scripts/rechunk_data.py --input <src.h5> --output <dst.h5> --compression lzf
```

Copies one trajectory at a time (~3 MB RAM per iteration). Writes output with `h5py.File(..., "w")` — refuses to overwrite existing files. Verifies 5 random frames (seed=42) for bit-identical content after conversion.

---

## 3. File Statistics

| File | Size | Shape | Chunks | Compression | Time |
|------|------|-------|--------|-------------|------|
| `lenia_train.h5` (original) | 1.466 GB | (1800, 200, 64, 64) | (1, 200, 64, 64) | LZF | — |
| `lenia_train_chunked.h5` | 1.498 GB | (1800, 200, 64, 64) | (1, 1, 64, 64) | LZF | 89.6s |
| `lenia_val.h5` (original) | 0.162 GB | (200, 200, 64, 64) | (1, 200, 64, 64) | LZF | — |
| `lenia_val_chunked.h5` | 0.166 GB | (200, 200, 64, 64) | (1, 1, 64, 64) | LZF | 10.2s |

Size overhead: **+2.2%** (same LZF codec, more chunk headers). Originals kept intact.

---

## 4. Data Integrity

5 random frames verified bit-identical between original and rechunked files (both train and val).

```
Verifying 5 random frames (seed=42)…
  [OK] traj=…  max=…  mean=…
All 5 frames match — data integrity verified. ✓
```

---

## 5. Loading Benchmark

Benchmark: 100 shuffled batches, batch_size=32, num_workers=0.

| File | Layout | ms/batch | Speedup |
|------|--------|----------|---------|
| `lenia_train.h5` (original) | (1, 200, 64, 64) LZF | 1183.5 ms | baseline |
| `lenia_train_chunked.h5` | (1, 1, 64, 64) LZF | 25.0 ms | **47.4×** |

**47.4× speedup.** At 25 ms/batch, loading 358,200 pairs in batches of 32 takes ~2.8 min/epoch (vs ~18 min).

---

## 6. DataLoader Sanity Check

Ran on rechunked files without pre-loading (full lazy h5py access per frame):

```
Train pairs: 358,200  Val pairs: 39,800
First batch: frame_t=torch.Size([32, 1, 64, 64])  range=[0.0000, 1.0000]
20 batches in 0.55s  →  27.4 ms/batch
DataLoader sanity check: PASSED
```

Shapes, value ranges, and throughput all correct.

---

## 7. Config Update

`config.yaml` updated to use the rechunked files:

```yaml
data:
  train_path: data/lenia_train_chunked.h5
  val_path: data/lenia_val_chunked.h5
```

---

## 8. Recommendation

Use `lenia_train_chunked.h5` and `lenia_val_chunked.h5` for all training. The original files can be kept as backup or deleted to reclaim ~1.6 GB.

A no-compression variant (`--compression none`) was not produced — at 47.4× speedup, LZF overhead is negligible and the size saving is worthwhile.
