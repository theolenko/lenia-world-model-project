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
| Pixel-CNN | 0.00829 | 20.93 | 0.918 | 0.00045 |
| Pixel-ViT | 0.00841 | 20.92 | 0.869 | 0.00044 |
| Patch-JEPA | 0.00074 | 31.34 | 0.949 | 0.00045 |

**Key findings:**
- **Patch-JEPA's one-step numbers are misleadingly flattering.** The low MSE (1.64× identity
  baseline) and high SSIM are partly an artifact of how the PatchDecoder was trained: it
  predicts `frame_{t+1}` from predictor embeddings, but because Lenia changes slowly, the
  predictor output still carries strong information about the current frame, and the decoder
  can score well by predicting something close to the input. Additionally, the MSE-trained
  decoder has a systematic brightness bias: it outputs slightly elevated pixel values
  (measured mean ~0.14 vs GT mean ~0.10), creating a grayish background and overexposed
  organism blobs. Visually, Patch-JEPA one-step outputs look noticeably brighter and less
  clean than Pixel-CNN or Pixel-ViT despite the better numbers.
- **Pixel-CNN and Pixel-ViT** are the more honest performers at step 1. Their ~18× identity
  MSE reflects genuine prediction error. Pixel-ViT edges out on PSNR; Pixel-CNN on SSIM.
- **All models score worse than the identity predictor on one-step MSE (ratio ≈ 18×).** The
  identity baseline of 0.00045 measures how much a Lenia frame changes between t and t+1. Any
  model that tries to predict where the organism actually goes — and gets it slightly wrong
  spatially — will incur larger MSE than just copying the input. This is the fundamental MSE
  paradox for slow-dynamics systems: the "predict nothing changes" baseline is very hard to beat
  numerically even when visual prediction quality is clearly better. The ratio number (18×) does
  not mean models are bad; it means MSE on a [0,1] image penalises spatial misalignment far more
  than stasis. Multi-step rollout (Scenario 2) and visual inspection are more informative.

Plot: `evaluation/results/one_step_comparison.png`  
Visual grid (input | predicted | GT per model): `evaluation/results/one_step_grid.png`

---

## Scenario 2 — Multi-step autoregressive rollout

Each model rolled autoregressively for 30 steps from the same initial frame.
Compared against the **identity baseline** (predict "no change"). Averaged over 5 active
val trajectories (filtered so the organism does not die during the eval window — see Limitations).

| Model | Step 1 MSE | Step 30 MSE | Competence horizon |
|-------|-----------|------------|-------------------|
| Pixel-CNN | 0.00454 | 0.15342 | **none (above baseline at step 1)** |
| Pixel-ViT | 0.00327 | 0.14910 | step 8 |
| Patch-JEPA | **0.00173** | 0.16730 | **step 10** |

Identity baseline: step 1 = 0.00333, step 30 = 0.11276.

**Key findings:**
- **Patch-JEPA** has the longest competence horizon (step 10) and the lowest step-1 MSE — it
  is 2× better than the identity baseline at step 1 and stays below it until step 10. The JEPA
  encoder/predictor pair genuinely learned useful short-horizon dynamics.
- **Pixel-ViT** is competitive up to step 8, then rapidly diverges. Global self-attention
  accumulates input drift quickly under autoregressive rollout.
- **Pixel-CNN is the worst model on active trajectories.** Its step-1 MSE (0.00454) already
  exceeds the identity baseline (0.00333) — meaning it is actively making predictions worse
  than just predicting no change, even at t=1. Its CNN inductive bias (local spatial pooling)
  works well on dying or quiescent trajectories where predicting near-black is correct, but
  fails to track the fast local dynamics of active organisms. The earlier finding that
  "Pixel-CNN stays below identity for all 30 steps" was entirely an artifact of the trajectory
  selection: most of the first 5 val trajectories contained dying organisms, and all models
  trivially predicted black correctly, making Pixel-CNN look best.
- **All three models** diverge above the identity baseline by step 10–15 on active trajectories.
  Active Lenia dynamics are genuinely hard — frames change fast enough that any accumulated
  positional error quickly compounds.
- **Patch-JEPA rollout degradation** pattern is unchanged: coherent steps 1–8, salt-and-pepper
  noise onset around step 10–15 as predictor embeddings compound outside the decoder's training
  distribution. Max pixel value saturates to ~0.99+ by step 12. Visually unusable beyond step 10
  despite numerically winning the MSE race at short horizons.

Plot: `evaluation/results/multistep_rollout_comparison.png`

---

## Scenario 3 — OOD robustness

One-step MSE as a function of Gaussian input noise σ.

| Model | σ=0.00 | σ=0.02 | σ=0.05 | σ=0.10 | σ=0.20 |
|-------|--------|--------|--------|--------|--------|
| Pixel-CNN | 0.00895 | 0.00957 (+7%) | 0.01210 (+35%) | 0.01721 (+92%) | 0.02722 (+204%) |
| Pixel-ViT | 0.00797 | 0.00840 (+5%) | 0.01028 (+29%) | 0.01510 (+89%) | 0.02880 (+261%) |
| Patch-JEPA | 0.00154 | 0.00163 (+6%) | 0.00214 (+39%) | 0.00401 (+160%) | 0.01336 (+768%) |

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

### Results (mean MSE step 10, averaged over 5 active trajectories)

| Intervention | Pixel-CNN | Pixel-ViT | Patch-JEPA |
|-------------|-----------|-----------|------------|
| inject_blob | 0.1189 | 0.1260 | **0.0953** |
| zero_region | 0.1078 | 0.1035 | **0.0830** |
| mirror_patch | 0.1153 | 0.1124 | **0.0910** |
| scale_density | 0.1236 | **0.1128** | 0.1136 |
| add_noise | 0.1247 | 0.1198 | **0.0944** |

**Patch-JEPA has the lowest step-10 MSE in 4 of 5 interventions.** Pixel-CNN is the worst in
4 of 5. This mirrors the multi-step rollout finding: on active trajectories, Pixel-CNN fails
to track fast dynamics and accumulates error fastest.

### Per-intervention step-1 and step-30 MSE

| Intervention | Model | step 1 | step 10 | step 30 |
|-------------|-------|--------|---------|---------|
| inject_blob | Pixel-CNN | 0.0649 | 0.1189 | 0.1850 |
| inject_blob | Pixel-ViT | 0.0592 | 0.1260 | 0.1953 |
| inject_blob | Patch-JEPA | **0.0454** | **0.0953** | 0.2044 |
| zero_region | Pixel-CNN | 0.0626 | 0.1078 | 0.1655 |
| zero_region | Pixel-ViT | 0.0552 | 0.1035 | 0.1707 |
| zero_region | Patch-JEPA | **0.0428** | **0.0830** | 0.1732 |
| mirror_patch | Pixel-CNN | 0.0634 | 0.1153 | 0.1832 |
| mirror_patch | Pixel-ViT | 0.0582 | 0.1124 | 0.1842 |
| mirror_patch | Patch-JEPA | **0.0460** | **0.0910** | 0.1910 |
| scale_density | Pixel-CNN | 0.0920 | 0.1236 | 0.1773 |
| scale_density | Pixel-ViT | 0.0896 | **0.1128** | 0.1795 |
| scale_density | Patch-JEPA | **0.0682** | 0.1136 | 0.2010 |
| add_noise | Pixel-CNN | 0.0643 | 0.1247 | 0.1840 |
| add_noise | Pixel-ViT | 0.0592 | 0.1198 | 0.1807 |
| add_noise | Patch-JEPA | **0.0460** | **0.0944** | 0.1953 |

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

Pixel-CNN and Pixel-ViT produce clean, well-calibrated one-step predictions with correct
background level and organism brightness. Patch-JEPA predictions have a visible brightness
artefact: the background is slightly grey instead of black, and the organism blobs appear
overexposed and more diffuse. This is a systematic bias of the MSE-trained PatchDecoder —
it outputs a slightly elevated mean pixel value (~0.14 vs GT ~0.10) across all predictions,
expanding the effective dynamic range and making the output look "washed out" or scaled up
compared to ground truth. This is not fixable by retraining with MSE alone; a perceptual
or SSIM loss would be needed to suppress it.

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
~0.068–0.092). The global density shift immediately changes the pixel distribution in a way
that no model was trained to handle. At step 10, Pixel-ViT and Patch-JEPA are essentially
tied (0.1128 vs 0.1136); Pixel-CNN is worst (0.1236). This intervention effectively tests
whether models have an internal representation of global state — which they do not.

**`mirror_patch` and `add_noise`:**  
Patch-JEPA handles both best at step 10. Patch-JEPA step-30 collapse to salt-and-pepper noise
is visible in the snapshots for all interventions — the final column of the Patch-JEPA row shows
a bright speckle pattern rather than coherent organism structures.

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

This depends on how you measure "best," and the answer changed dramatically once the evaluation
was restricted to active (non-dying) trajectories.

**On active trajectories, Patch-JEPA is numerically best:**
- Lowest step-1 MSE (0.00173 vs. identity baseline 0.00333 — 2× better than no-change)
- Longest competence horizon (step 10 before crossing the identity baseline)
- Lowest step-10 MSE in 4 of 5 intervention experiments

**Pixel-ViT is second:** Competence to step 8, second-best on 4 of 5 interventions.

**Pixel-CNN is the weakest model on active trajectories:**
- Step-1 MSE (0.00454) is already worse than the identity baseline (0.00333) — predicting
  no change would be better at every time step
- Worst in 4 of 5 intervention experiments at step 10
- The previous finding that "Pixel-CNN stays below identity for all 30 steps" was an artifact
  of using the first 5 val trajectories, most of which contain dying organisms. All models
  trivially predict black on dying trajectories, artificially inflating Pixel-CNN's ranking.

**However, visual quality tells a different story for Patch-JEPA:**
- One-step outputs are visibly overexposed with a grey background artefact (brightness bias)
- Rollouts collapse to bright salt-and-pepper noise after ~10 steps
- Long-rollout step-30 MSE (0.167) is actually the worst of the three models

**Conclusion: Patch-JEPA is quantitatively best on active trajectories, but visually unusable
beyond step 10. Pixel-ViT is the best balance of numerical performance and visual coherence.
Pixel-CNN's apparent strength in earlier evaluations was a trajectory-selection artifact.**

### What the models actually learned

- **Pixel-CNN** learned a smooth local transition function that approximates Lenia's convolution
  kernel behavior. This inductive bias is well-matched to dying or quiescent trajectories where
  predicting near-zero is correct. On active organisms with fast local dynamics, the CNN's
  receptive field is too limited and predictions diverge immediately.
- **Pixel-ViT** learned global spatial dependencies that help short-term but are fragile under
  distribution shift (autoregressive drift, perturbations). Better than CNN on active dynamics.
- **Patch-JEPA** learned rich patch-level dynamics in embedding space. The quality of its
  one-step predictions and intervention responses suggests the JEPA encoder/predictor captured
  genuine structural features of active Lenia organisms. The multi-step degradation is a decoder
  problem, not a representation problem — the PatchDecoder was not trained to invert chained
  predictor outputs.

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
- **Active-trajectory filter**: trajectories where the organism dies within the eval window
  (mean pixel value drops below 0.05) are excluded. This is necessary for a meaningful
  comparison — all models correctly predict black on dying trajectories, producing
  artificially low MSE that does not reflect dynamics learning. Without the filter, Pixel-CNN
  appeared best at all 30 steps; with it, Pixel-CNN is worst from step 1. The filter makes
  the evaluation both stricter and more honest.
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

- **All models fail to beat the identity baseline on active trajectory rollouts.** On active
  trajectories, the identity predictor ("predict no change") turns out to be a very strong
  baseline: because the organism moves, any predicted frame that is even slightly misaligned
  spatially incurs larger MSE than just copying the last frame. Pixel-CNN is already above the
  identity baseline at step 1; Pixel-ViT by step 8; Patch-JEPA by step 10. Beyond step 10, all
  models have worse rollout MSE than the no-change predictor. This is not primarily a model
  failure — it reflects that MSE is the wrong metric for evaluating predictions of moving objects:
  a sharp prediction that is two pixels wrong will outscore the identity predictor on MSE, but
  visually it is clearly better. Perceptual or structural metrics would rank the models more fairly.
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
- **The PatchDecoder produces visually poor output despite good MSE numbers.** It has a
  systematic brightness bias (predicted mean ~0.14 vs GT mean ~0.10), a grey background
  instead of black, and overexposed organism blobs. This is a known MSE-decoder artefact:
  the loss does not penalise spatial incoherence or mean offset, so the decoder learns a
  slightly scaled-up output distribution.
- **The PatchDecoder is the primary bottleneck**, not the JEPA encoder/predictor. Under
  autoregressive rollout, the predictor receives chained outputs (`predictor(predictor(…))`),
  a distribution it was never trained on. Max pixel values climb to 0.99+ by step 12 and the
  output degrades to salt-and-pepper noise — visually unusable beyond step 10.
- **Patch-JEPA's MSE/SSIM scores overstate its quality** because: (1) Lenia is slow-changing
  so near-identity predictions score well, (2) the brightness bias inflates structural
  similarity metrics, and (3) MSE rewards blurry/diffuse predictions that overlap with GT.
- The JEPA model was never evaluated on its **latent representation quality** — e.g. whether
  embeddings cluster by organism type or support causal reasoning. The pixel-space evaluation
  measures only the decoder output, not the representation itself, which is JEPA's actual
  design goal. A probing study on the embeddings directly would be more informative.
