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

**PatchDecoder training note (job 24911376):** The decoder is trained on `predictor(encoder(frame_t))`
inputs with `frame_t` as target (100% predictor embeddings, no distribution shift at inference).
However, this training objective teaches the decoder to reconstruct the *current* frame from the
predictor output, not the *next* frame. Since `predictor(encoder(frame_t)) ≈ encoder(frame_{t+1})`
(JEPA EMA property), the correct objective would use `frame_{t+1}` as target — this is a known
limitation (see Limitations section). The result is that Patch-JEPA acts as a near-identity
predictor: excellent one-step MSE on clean data, but poor multi-step and OOD performance.

---

## Scenario 1 — One-step prediction

Measured on 2000 held-out frame pairs from `v2_lenia_val_chunked.h5`.

| Model | MSE ↓ | PSNR ↑ (dB) | SSIM ↑ | Identity baseline MSE |
|-------|-------|-------------|--------|-----------------------|
| Pixel-CNN | 0.00822 | 21.02 | 0.919 | 0.00045 |
| Pixel-ViT | 0.00810 | 21.12 | **0.869** | 0.00045 |
| Patch-JEPA | **0.00085** | **30.74** | **0.933** | 0.00045 |

**Key findings:**
- **Patch-JEPA's one-step metrics are misleadingly good.** MSE=0.00085 is only 1.9× the identity
  baseline (vs ~18× for pixel models), and SSIM=0.933 exceeds pixel models. This is the
  *identity-predictor effect*: the PatchDecoder was trained to output `frame_t` from
  `predictor(encoder(frame_t))`, so at inference it reproduces the current frame. Lenia evolves
  slowly, so the current frame is already close to the next frame — hence low MSE and high SSIM.
  The model is not learning dynamics; it is learning to copy its input.
- **Pixel-CNN and Pixel-ViT are nearly equal** in MSE/PSNR at ~0.008; Pixel-ViT slightly edges
  out on PSNR, Pixel-CNN on SSIM.

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
| Patch-JEPA | 0.00331 | 0.12547 | step 2 |

Identity baseline: step 1 = 0.00352, step 30 = 0.07457.

**Key findings:**
- **Pixel-CNN** stays below the identity baseline for all 30 steps — the strongest long-horizon predictor.
- **Pixel-ViT** degrades faster than Pixel-CNN (horizon=13): ViT attention is sensitive to
  accumulated input drift, compounding errors in autoregressive mode.
- **Patch-JEPA's multi-step performance is poor despite a low step-1 MSE.** Step 1 (0.00331) is
  just below the identity baseline, but by step 30 (0.12547) it has diverged well above baseline.
  This confirms the identity-predictor diagnosis: the decoder outputs approximately the input frame
  at each step, so the predicted trajectory stalls rather than tracking actual Lenia dynamics.
  Errors compound as the "frozen" prediction drifts further from the true evolving state.

Plot: `evaluation/results/multistep_rollout_comparison.png`

---

## Scenario 3 — OOD robustness

One-step MSE as a function of Gaussian input noise σ.

| Model | σ=0.00 | σ=0.02 | σ=0.05 | σ=0.10 | σ=0.20 |
|-------|--------|--------|--------|--------|--------|
| Pixel-CNN | 0.00895 | 0.00957 (+7%) | 0.01209 (+35%) | 0.01747 (+95%) | 0.02715 (+203%) |
| Pixel-ViT | 0.00797 | 0.00841 (+5%) | 0.01028 (+29%) | 0.01510 (+89%) | 0.02918 (+266%) |
| Patch-JEPA | 0.00192 | 0.00201 (+5%) | 0.00248 (+29%) | 0.00464 (+142%) | 0.01542 (+703%) |

**Key findings:**
- **Patch-JEPA degrades catastrophically at high noise** (+703% at σ=0.2 vs +203%/+266% for
  pixel models). The PatchDecoder was trained on predictor outputs from clean frames; noisy
  inputs produce out-of-distribution predictor embeddings that break the decoder badly.
- **Pixel models are comparably noise-robust** to each other. The relative ranking at σ=0.2
  (Pixel-CNN < Pixel-ViT) is consistent with Pixel-CNN's stronger spatial pooling.
- The previous decoder (trained on encoder embeddings) showed Patch-JEPA at only +41% at
  σ=0.2, because the ViT encoder's natural low-pass filtering was not disrupted by the
  decoder's training objective. The current decoder does not benefit from that robustness.

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
| inject_blob | 0.0495 | 0.0713 | 0.0834 |
| zero_region | 0.0429 | 0.0631 | 0.0681 |
| mirror_patch | 0.0471 | 0.0682 | 0.0749 |
| scale_density | 0.0585 | 0.0745 | 0.1036 |
| add_noise | 0.0524 | 0.0773 | 0.0783 |

### Per-intervention step-1 and step-30 MSE

| Intervention | Model | step 1 | step 10 | step 30 |
|-------------|-------|--------|---------|---------|
| inject_blob | Pixel-CNN | 0.0318 | 0.0495 | 0.0909 |
| inject_blob | Pixel-ViT | 0.0334 | 0.0713 | 0.1301 |
| inject_blob | Patch-JEPA | 0.0450 | 0.0834 | 0.1672 |
| zero_region | Pixel-CNN | 0.0312 | 0.0429 | 0.0660 |
| zero_region | Pixel-ViT | 0.0339 | 0.0631 | 0.1152 |
| zero_region | Patch-JEPA | 0.0410 | 0.0681 | 0.1395 |
| mirror_patch | Pixel-CNN | 0.0321 | 0.0471 | 0.0702 |
| mirror_patch | Pixel-ViT | 0.0344 | 0.0682 | 0.1286 |
| mirror_patch | Patch-JEPA | 0.0436 | 0.0749 | 0.1523 |
| scale_density | Pixel-CNN | 0.0461 | 0.0585 | 0.1002 |
| scale_density | Pixel-ViT | 0.0502 | 0.0745 | 0.1256 |
| scale_density | Patch-JEPA | 0.0649 | 0.1036 | 0.1841 |
| add_noise | Pixel-CNN | 0.0319 | 0.0524 | 0.0960 |
| add_noise | Pixel-ViT | 0.0345 | 0.0773 | 0.1336 |
| add_noise | Patch-JEPA | 0.0445 | 0.0783 | 0.1655 |

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

- **Pixel-CNN is the best intervention tracker** at all horizons. Its step-30 MSE (0.066–0.100)
  is consistently lowest, and it stays below the identity baseline throughout.
- **Pixel-ViT diverges at step 30** (~0.11–0.13), consistent with its autoregressive instability
  in Scenario 2.
- **Patch-JEPA is worst in interventions** (step-30 MSE 0.14–0.18). The identity-predictor
  behaviour from the PatchDecoder compounds here: the model was already drifting during the t*=10
  autonomous steps, and after the intervention it continues to output near-static frames rather
  than tracking the post-intervention Lenia physics.
- **scale_density is the hardest intervention** for all models (highest step-1 MSE), because the
  global density shift changes the overall pixel distribution immediately.

### Limitations

- **PatchDecoder training objective is incorrect for next-frame prediction.** The decoder is
  trained on `predictor(encoder(frame_t)) → frame_t` (current frame as target). The correct
  objective for a predictive decoder is `predictor(encoder(frame_t)) → frame_{t+1}` (next frame
  as target). This would exploit the JEPA property that
  `predictor(encoder(frame_t)) ≈ encoder(frame_{t+1})` and train the decoder to actually invert
  that embedding into a pixel frame. Fixing this is the primary recommended next step.
- MSE loss in PatchDecoder training produces blurry reconstructions. A perceptual loss or SSIM
  loss would improve sharpness but is not used.
- Step-1 MSE (~0.031–0.046 in interventions) is partially dominated by 10-step autonomous drift
  before the intervention. A shorter `t_star` would isolate the intervention effect more cleanly.
- Only 5 val trajectories are used for averaging — results may vary with more trajectories.
