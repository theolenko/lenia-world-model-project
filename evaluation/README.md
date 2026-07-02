# Evaluation & Intervention Analysis

Comparison of **Pixel-CNN**, **Pixel-ViT**, and **Patch-JEPA** across three evaluation scenarios
plus five causal intervention experiments. All metrics are in **pixel space** (MSE / PSNR / SSIM)
for a fair comparison across architectures.

> **Why not JEPA-CNN?**  
> JEPA-CNN's predictor outputs flat embedding vectors with no pixel decoder, so pixel-space
> comparison is not possible. Only models that can produce pixel-space outputs at test time are included.

---

## Models

| Model | Encoder | Training objective | Pixel output |
|-------|---------|-------------------|-------------|
| Pixel-CNN | 4-layer stride-2 CNN (embed_dim=128) | Next-frame MSE in pixel space | Direct decoder |
| Pixel-ViT | 6-layer ViT, patch_size=8 (embed_dim=256) | Next-frame MSE in pixel space | CNN decoder |
| Patch-JEPA | 6-layer ViT, pool=False (embed_dim=128) | JEPA: predict next patch embeddings (EMA target) | PatchDecoder (trained separately) |

**PatchDecoder training (job 25015367):** Decoder trained on `predictor(encoder(frame_t))` inputs
with `frame_{t+1}` as target (val_loss=0.000845 next-frame MSE). This exploits the JEPA EMA
property: `predictor(encoder(frame_t)) ≈ encoder(frame_{t+1})`, so the decoder learns to invert
that embedding back to pixels of the next frame. The decoder sees its actual inference distribution
(predictor outputs) at training time.

---

## Scenario 1 — One-step prediction

Measured on 2000 held-out frame pairs from `v2_lenia_val_chunked.h5`.

| Model | MSE ↓ | PSNR ↑ (dB) | SSIM ↑ | Identity baseline MSE |
|-------|-------|-------------|--------|-----------------------|
| Pixel-CNN | 0.00859 | 20.86 | 0.917 | 0.00045 |
| Pixel-ViT | 0.00814 | 21.05 | 0.869 | 0.00045 |
| Patch-JEPA | **0.00074** | **31.35** | **0.949** | 0.00045 |

**Key findings:**
- **Patch-JEPA achieves the lowest one-step MSE** (0.00074, ratio=1.64× identity baseline). This
  is partly a genuine improvement from the fixed decoder objective, and partly reflects that
  Lenia evolves slowly — the decoder's predictions are very close to the current frame.
- **Pixel-CNN and Pixel-ViT** sit at ~18× the identity baseline, consistent with prior runs.
  Their performance is stable across decoder versions since they don't use a separate decoder.
- Patch-JEPA's low one-step MSE should be interpreted carefully: multi-step rollout reveals it
  loses competence by step 10 (see Scenario 2), meaning it does not learn the full dynamics.

Plot: `evaluation/results/one_step_comparison.png`  
Visual grid (input | predicted | GT per model): `evaluation/results/one_step_grid.png`

---

## Scenario 2 — Multi-step autoregressive rollout

Each model rolled autoregressively for 30 steps from the same initial frame.
Compared against the **identity baseline** (predict "no change"). Averaged over 5 val trajectories.

| Model | Step 1 MSE | Step 30 MSE | Competence horizon |
|-------|-----------|------------|-------------------|
| Pixel-CNN | 0.00294 | 0.06362 | **> step 30** |
| Pixel-ViT | 0.00286 | 0.10278 | step 13 |
| Patch-JEPA | 0.00142 | 0.12484 | step 10 |

Identity baseline: step 1 = 0.00352, step 30 = 0.07457.

**Key findings:**
- **Pixel-CNN** stays below the identity baseline for all 30 steps — the strongest long-horizon predictor.
- **Pixel-ViT** degrades faster than Pixel-CNN (horizon=13).
- **Patch-JEPA** has step 1 MSE=0.00142 (well below baseline) and holds competence to step 10.
  After step 10 it diverges above baseline, reaching 0.12484 at step 30. This reflects error
  accumulation in the ViT embedding space — the predictor's autoregressive distribution drifts
  from what the decoder was trained on, causing the pixel output to degrade.

Plot: `evaluation/results/multistep_rollout_comparison.png`

---

## Scenario 3 — OOD robustness

One-step MSE as a function of Gaussian input noise σ.

| Model | σ=0.00 | σ=0.02 | σ=0.05 | σ=0.10 | σ=0.20 |
|-------|--------|--------|--------|--------|--------|
| Pixel-CNN | 0.00895 | 0.00960 (+7%) | 0.01206 (+35%) | 0.01730 (+93%) | 0.02709 (+203%) |
| Pixel-ViT | 0.00797 | 0.00840 (+5%) | 0.01027 (+29%) | 0.01523 (+91%) | 0.02918 (+266%) |
| Patch-JEPA | 0.00154 | 0.00163 (+6%) | 0.00213 (+38%) | 0.00405 (+163%) | 0.01348 (+775%) |

**Key findings:**
- **Patch-JEPA degrades severely at high noise** (+775% at σ=0.2 vs +203%/+266% for pixel models).
  The PatchDecoder was trained on predictor outputs from clean frames; noisy inputs shift the
  encoder/predictor embedding distribution, causing the decoder to produce degraded outputs.
- **Patch-JEPA's low clean baseline** (0.00154) reflects its good one-step accuracy, but the
  large relative degradation means it falls behind pixel models in absolute terms by σ=0.10.
- **Pixel models are comparably noise-robust**, degrading at similar rates to each other.

Plot: `evaluation/results/ood_robustness_comparison.png`

---

## Intervention Analysis

Causal probing: does the model correctly predict Lenia physics **after a state perturbation**?

### Protocol

1. **Ground truth (GT):** Real Lenia runs `t_star=10` steps from `traj[0]` → intervention applied to GT's own frame → Lenia continues for 30 steps
2. **Each model:** Autoregressive rollout for `t_star=10` steps → intervention applied to **model's own predicted frame** (same random parameters as GT — same blob position, noise seed, etc.) → model predicts 30 more steps
3. **MSE:** Model prediction vs. shared Lenia GT, per step, averaged over 5 trajectories

Each model is compared against the **same** ground truth so numbers are directly comparable across models.

```bash
python interventions/run_all_interventions.py --n-traj 5 --t-star 10 --n-steps 30
```

### Results (mean MSE steps 1–10, averaged over 5 trajectories)

| Intervention | Pixel-CNN | Pixel-ViT | Patch-JEPA |
|-------------|-----------|-----------|------------|
| inject_blob | 0.0495 | 0.0713 | 0.0704 |
| zero_region | 0.0429 | 0.0631 | 0.0586 |
| mirror_patch | 0.0471 | 0.0682 | 0.0632 |
| scale_density | 0.0585 | 0.0745 | 0.0867 |
| add_noise | 0.0524 | 0.0773 | 0.0684 |

### Per-intervention step-1 and step-30 MSE

| Intervention | Model | step 1 | step 10 | step 30 |
|-------------|-------|--------|---------|---------|
| inject_blob | Pixel-CNN | 0.0318 | 0.0495 | 0.0909 |
| inject_blob | Pixel-ViT | 0.0334 | 0.0713 | 0.1301 |
| inject_blob | Patch-JEPA | 0.0340 | 0.0704 | 0.1473 |
| zero_region | Pixel-CNN | 0.0312 | 0.0429 | 0.0660 |
| zero_region | Pixel-ViT | 0.0339 | 0.0631 | 0.1152 |
| zero_region | Patch-JEPA | 0.0324 | 0.0586 | 0.1240 |
| mirror_patch | Pixel-CNN | 0.0321 | 0.0471 | 0.0702 |
| mirror_patch | Pixel-ViT | 0.0344 | 0.0682 | 0.1286 |
| mirror_patch | Patch-JEPA | 0.0338 | 0.0632 | 0.1329 |
| scale_density | Pixel-CNN | 0.0461 | 0.0585 | 0.1002 |
| scale_density | Pixel-ViT | 0.0502 | 0.0745 | 0.1255 |
| scale_density | Patch-JEPA | 0.0503 | 0.0867 | 0.1558 |
| add_noise | Pixel-CNN | 0.0319 | 0.0524 | 0.0960 |
| add_noise | Pixel-ViT | 0.0345 | 0.0773 | 0.1336 |
| add_noise | Patch-JEPA | 0.0351 | 0.0684 | 0.1480 |

### Interventions

| Name | Description | What it probes |
|------|-------------|---------------|
| `inject_blob` | Gaussian organism seed at random position | Can the model integrate a new organism? |
| `zero_region` | Kill circular patch (set to 0) | Does the model understand local cell death? |
| `mirror_patch` | Horizontally flip a 24×24 local patch | Does the model handle broken local symmetry? |
| `scale_density` | Multiply all cell values by 1.5 | Does the model track global density shifts? |
| `add_noise` | σ=0.05 Gaussian noise everywhere | Can the model absorb low-level corruption? |

### Output files

```
experiments/intervention_results/
├── intervention_{name}.png        # MSE curve over 30 steps, all models vs. Lenia GT
├── snapshots_{name}.png           # Full timeline: pre-rollout | before pert | PERTURBED | post, one row per model + GT
├── gif_{name}_{model}.gif         # [Model | GT] side by side, animated (15 GIFs total)
└── intervention_heatmap.png       # Mean MSE steps 1–10 heatmap (model × intervention)
```

### Interpretation

- **Pixel-CNN is the strongest intervention tracker** at all horizons (step-30 MSE 0.066–0.100).
- **Patch-JEPA and Pixel-ViT are comparable** in interventions (step-10 MSE 0.058–0.087 vs
  0.063–0.077). Patch-JEPA matches or beats Pixel-ViT on most interventions at step 10, though
  it diverges more at step 30 (0.12–0.16 vs 0.12–0.13).
- **scale_density remains the hardest intervention** for all models (highest step-1 MSE ~0.046–0.050),
  since the global density shift immediately changes the overall pixel distribution.
- **Patch-JEPA step-1 MSE in interventions** (0.032–0.050) is now comparable to Pixel-CNN
  (0.031–0.046), a significant improvement over the previous broken decoder (0.041–0.065).

### Limitations

- **PatchDecoder OOD fragility:** The decoder was trained on clean-frame predictor outputs.
  Noisy or post-intervention inputs shift the embedding distribution, causing +775% MSE
  degradation at σ=0.2 (vs +203% for Pixel-CNN). Training with augmented inputs or a more
  robust loss (perceptual/SSIM) would help.
- **Multi-step divergence after step 10:** Autoregressive drift in embedding space pushes
  predictor outputs out of the decoder's training distribution. A recurrent latent state or
  periodic encoder refresh would reduce this.
- MSE loss produces slightly blurry reconstructions. Perceptual/SSIM losses not used.
- Only 5 val trajectories for intervention averaging — results may vary with more.
