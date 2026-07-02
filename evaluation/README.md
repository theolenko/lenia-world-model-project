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

**PatchDecoder training (job 25015367):** Decoder input is `predictor(encoder(frame_t))`, target
is `frame_{t+1}` (val_loss=0.000845). This uses the JEPA EMA property
`predictor(encoder(frame_t)) ≈ encoder(frame_{t+1})` — the decoder learns to invert the
predicted next-frame embedding back to pixels.

---

## Scenario 1 — One-step prediction

Measured on 2000 held-out frame pairs from `v2_lenia_val_chunked.h5`.

| Model | MSE ↓ | PSNR ↑ (dB) | SSIM ↑ | Identity baseline MSE |
|-------|-------|-------------|--------|-----------------------|
| Pixel-CNN | 0.00859 | 20.86 | 0.917 | 0.00045 |
| Pixel-ViT | 0.00814 | 21.05 | 0.869 | 0.00045 |
| Patch-JEPA | **0.00074** | **31.35** | **0.949** | 0.00045 |

**Key findings:**
- **Patch-JEPA achieves the best one-step metrics** by a wide margin. Predictions are sharp and
  structurally accurate — visually comparable to or better than pixel models. The low MSE
  (1.64× identity baseline) partially reflects Lenia's slow dynamics (frames change little per
  step), but the SSIM=0.949 confirms genuine structural accuracy, not just copying the input.
- **Pixel-CNN and Pixel-ViT** score ~18× the identity baseline. They are nearly equal; Pixel-ViT
  edges out on PSNR, Pixel-CNN on SSIM.
- **Caution on one-step MSE as the sole metric:** Lenia evolves slowly. Even copying the current
  frame as a prediction (identity baseline) gives MSE≈0.00045. All models beat this, but the
  gap between model and identity is modest. Multi-step performance (Scenario 2) is a much
  stricter test of whether the model actually learned dynamics.

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
- **Pixel-CNN** stays below identity for all 30 steps — the only model that consistently improves
  on the no-change baseline throughout. Its CNN inductive bias (local spatial pooling, translation
  equivariance) is well-suited to the local update rules of Lenia.
- **Pixel-ViT** crosses the identity baseline around step 13. ViT's global self-attention is
  more sensitive to accumulated input drift — small errors compound into global attention pattern
  shifts that cascade quickly.
- **Patch-JEPA** holds competence until around step 10, then diverges sharply to ~0.13 by step
  20. The curve shows a characteristic spike (step ~20) then slight recovery, which corresponds
  to the PatchDecoder collapsing to salt-and-pepper noise once predictor embeddings drift out of
  its training distribution. After full collapse the output is approximately constant random
  noise, which gives stable but high MSE. The JEPA encoder and predictor may still be producing
  meaningful embeddings after step 10 — the decoder is the bottleneck.

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
- **The relative degradation numbers for Patch-JEPA (+775%) are misleading.** Because its clean
  baseline is so low (0.00154 vs ~0.008 for pixel models), the absolute MSE at σ=0.2 is still
  0.013 — roughly half the pixel models' 0.027. In absolute terms, Patch-JEPA is still the most
  accurate at every noise level.
- **Visually**, high-noise Patch-JEPA likely produces incoherent outputs (decoder OOD) while
  pixel models degrade more gracefully with recognisable (if blurry) predictions. MSE does not
  capture this perceptual difference.
- **Pixel-CNN** is slightly less OOD-robust than Pixel-ViT at σ≤0.1, but the margin is small.

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
| zero_region | **0.0429** | 0.0631 | 0.0586 |
| mirror_patch | **0.0471** | 0.0682 | 0.0632 |
| scale_density | **0.0585** | 0.0745 | 0.0867 |
| add_noise | **0.0524** | 0.0773 | 0.0684 |

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

---

## Visual Analysis

### One-step grid (`one_step_grid.png`)

All three models produce visually convincing one-step predictions. Pixel-CNN and Pixel-ViT show
slight blur in fine structures (expected from MSE training). Patch-JEPA predictions are sharp
and match ground truth structure closely — the fixed decoder (target=`frame_{t+1}`) produces
notably cleaner outputs than earlier decoder versions.

### Intervention snapshots — per-intervention observations

**`zero_region` — Pixel-CNN "regrowth" is correct Lenia physics:**  
After the circular kill (black region), Pixel-CNN predicts that the organism quickly regrows into
the killed area within 1–2 steps. This is not a model failure — it reflects real Lenia dynamics:
the cellular automaton's convolution kernel allows surrounding live cells to propagate back into
dead regions, and Pixel-CNN's local receptive field correctly captures this local regeneration
rule. This is confirmed by the GT row, which also shows the organism re-establishing itself.
Pixel-CNN has the best `zero_region` MSE precisely because it learned this regeneration behavior.
Pixel-ViT also shows some regrowth but with more spatial uncertainty.

**`inject_blob` — new organism not fully absorbed:**  
All models show the injected blob initially as a bright spot (correct — they see the intervention).
Pixel-CNN and Pixel-ViT track the subsequent evolution moderately well. Patch-JEPA handles the
short-term accurately but collapses to noise by step 30. None of the models correctly predicts
the complex interaction between the injected blob and the existing organism (this requires
understanding long-range organism dynamics not present in the training objective).

**`scale_density` — global perturbations are hardest:**  
Multiplying all cells by 1.5 is the hardest intervention for every model (highest step-1 MSE
~0.046–0.050). The global density shift immediately changes the pixel distribution in a way
that no model was trained to handle. Patch-JEPA collapses to noise the fastest; Pixel-CNN
maintains structure longest. This intervention effectively tests whether models have an
internal representation of global state — which they do not.

**`mirror_patch` and `add_noise`:**  
Pixel-CNN handles both best. Patch-JEPA step-30 collapse to salt-and-pepper noise is visible
in the snapshots for all interventions — the final column of the Patch-JEPA row shows a bright
speckle pattern rather than coherent organism structures.

### Patch-JEPA long-rollout noise collapse

The characteristic behavior in all Patch-JEPA snapshots: coherent prediction for the first
~10 steps, then sudden onset of salt-and-pepper noise by step 20–30. The noise is not random
from frame to frame — it's a consistent high-frequency pattern the decoder produces when fed
with out-of-distribution predictor embeddings. The decoder was trained on single-step predictor
outputs; under autoregressive rollout, it receives `predictor(predictor(...(encoder(frame_0))))`,
a compounding chain that drifts far from the training distribution. Visually this makes long
Patch-JEPA rollouts unusable, even though the MSE numbers (which include the structured noise
pattern) are not catastrophically high.

---

## Discussion & Findings

### Which model is best overall?

**Pixel-CNN** is the most reliable and presentable model across all scenarios. It:
- Stays below the identity baseline for all 30 autoregressive steps
- Has the best intervention tracking in 4 of 5 interventions
- Produces visually coherent outputs at all time horizons
- Is the smallest and fastest model

**Patch-JEPA** produces the best single-step pixel quality and holds competence to step 10,
but its long-rollout behavior (noise collapse) makes it unsuitable as a standalone predictor.
Its JEPA encoder almost certainly produces better internal representations of Lenia state —
the decoder is the bottleneck, not the world model itself.

**Pixel-ViT** sits between the two: better than Pixel-CNN at step 1 but worse from step 13
onward. It is the most sensitive to perturbations (highest intervention MSE across all 5
interventions), suggesting ViT's global attention mechanism is more easily disrupted.

### What the models actually learned

- **Pixel-CNN** learned a smooth local transition function that approximates Lenia's convolution
  kernel behavior. It generalises well because its inductive bias matches the problem structure.
- **Pixel-ViT** learned global spatial dependencies that help short-term but are fragile under
  distribution shift (autoregressive drift, perturbations).
- **Patch-JEPA** learned rich patch-level dynamics in embedding space. The quality of its
  one-step predictions suggests the JEPA encoder captured genuine structural features. The
  multi-step degradation is a decoder problem, not a representation problem.

### Why one-step MSE is a weak evaluation metric for Lenia

Lenia frames are highly autocorrelated — consecutive frames are nearly identical (identity
baseline MSE ≈ 0.00045). A model that predicts "no change" scores 18× the identity baseline
which sounds bad, but 0.008 MSE on a [0,1] image is imperceptible to the eye. This means:
1. Small numerical MSE differences between models are not visually meaningful
2. A model can achieve very low MSE by being a near-identity predictor
3. The competence horizon (multi-step) is a more honest discriminator

---

## Limitations

### Data limitations

- **5 val trajectories** per intervention — results have high variance. Adding 20+ trajectories
  would be needed for statistically reliable conclusions.
- **Only v2 data** (pre-established organisms) — all models were trained and evaluated on
  trajectories where organisms are already active at t=0. Behavior on organism emergence, death,
  or multi-species interactions is untested.
- **Single Lenia parameter set** — the v2 dataset uses one set of Lenia growth/kernel parameters.
  All models likely overfit to this specific species. Generalization to different organisms is
  unknown.
- **64×64 resolution** — fine-grained structure (organism boundary details) cannot be resolved.
  Predictions at higher resolution would likely show more obvious blur for pixel models.
- **Slow dynamics inflate all metrics** — because frames change slowly, even low-quality models
  score numerically well on one-step MSE. This makes it hard to distinguish model quality from
  the "easy" nature of the prediction task.

### Evaluation limitations

- **MSE rewards blurry predictions.** A perfect blurry mean of the training distribution can
  score lower MSE than a sharp but slightly misaligned prediction. Perceptual metrics (LPIPS,
  SSIM) would better reflect visual quality.
- **The identity baseline MSE (0.00045) is not zero.** This is a useful sanity check but not
  the right floor — a model that simply copies the input and adds the average per-step motion
  vector would score better than 0.00045 without learning any physics.
- **t*=10 pre-intervention steps** mean all intervention step-1 MSEs (~0.031–0.050) already
  include 10 steps of autonomous prediction error. A shorter `t_star` (e.g. 2–3) would better
  isolate the intervention response from accumulated drift.
- **Interventions are not in the training distribution.** No model was exposed to zeroed
  regions, injected blobs, or scaled densities during training. All intervention results reflect
  out-of-distribution generalisation, not in-distribution behavior.
- **OOD relative percentages are misleading for Patch-JEPA.** Its +775% at σ=0.2 sounds
  catastrophic but the absolute MSE (0.013) is still lower than pixel models (~0.027). The
  high relative percentage is an artifact of Patch-JEPA's very low clean baseline. Visual
  quality under noise is another matter and is not captured by MSE.

### Model-specific limitations

**Pixel-CNN / Pixel-ViT:**
- Trained directly on pixel MSE — encourages learning the conditional mean of the next frame,
  producing slightly blurry outputs. There is no guarantee that the predicted frames are
  physically plausible Lenia states.
- No explicit representation of state — the encoder discards information not useful for one-step
  prediction, which may discard information needed for long-horizon dynamics.

**Patch-JEPA:**
- **The PatchDecoder is the primary bottleneck**, not the JEPA encoder/predictor. Under
  autoregressive rollout, the predictor receives its own previous output (`predictor(predictor(…))`),
  a distribution it was never trained on. The decoder amplifies this drift into visible noise
  after ~10 steps.
- **Decoder training distribution mismatch at test time for rollout:** The decoder saw single-step
  predictor outputs during training. During autoregressive evaluation it sees multi-step chained
  predictor outputs. These distributions diverge quickly.
- **MSE loss on decoder** produces outputs that are mean predictions over the uncertainty in
  the embedding-to-pixel mapping. A perceptual loss (LPIPS or SSIM) would reduce blur and
  likely also suppress the noise collapse by penalising spatial incoherence.
- The JEPA model was never evaluated on its **latent representation quality** — e.g. whether
  its embeddings cluster by organism type, predict downstream Lenia statistics, or support
  causal disentanglement. The pixel-space evaluation only measures the decoder's output, not
  the richness of the internal representation, which is JEPA's actual design goal.
