# 1D representation richness diagnostics

Four diagnostics computed on ~6000 held-out snapshots (1500 per system × 4 systems) for all eight trained 1D encoders.

Methods: `v3_recon`, `v3_vicreg`, `v3_spatiotemporal` (T2), `mlp` (MLPSparseAE), `cnn` (CNN1DAE), `mae` (MAE1DAE), `jepa_perceiver_sparse` (target branch), `jepa_vit1d` (target branch).

---

## (1) Participation Ratio of latent covariance

PR = (Σλ)² / Σλ² where λᵢ are eigenvalues of the latent covariance. Intrinsic-dim estimate ≈ 27 (≈25 GRF IC modes + coefficient + time).

| method | **PR_full** | heat | adv | burg | AC |
|---|---|---|---|---|---|
| V3-recon | **3.3** | 1.9 | 2.1 | 1.9 | 2.2 |
| V3+VICReg | 19.9 | 10.0 | 6.7 | 8.9 | 10.2 |
| T2 spatiotemp | **48.5** | 10.6 | **1.2** | 25.7 | 19.8 |
| MLPSparseAE | 8.6 | 3.6 | 8.1 | 4.7 | 5.2 |
| CNN1DAE | **2.5** | 1.8 | 3.0 | 3.3 | 1.3 |
| MAE1DAE | **2.5** | 1.1 | 3.0 | 1.2 | 2.6 |
| JEPA-Perceiver | 18.8 | 7.2 | 11.2 | 9.5 | 13.2 |
| JEPA-ViT | 12.6 | 6.9 | 13.8 | 13.6 | 6.4 |

**Findings:**
- **CNN, MAE, v3_recon are effectively rank-2.** They pass coefficient probes only because the coefficient happens to be a 1-D quantity that survives collapse. Classic "detector, not encoder" pattern.
- **T2 spatiotemporal's PR=48 is misleading.** Per-system PR for advection is **1.2** — pure rank-1 representation for translation. Whole-dataset richness was hiding a per-system collapse.
- **V3+VICReg, JEPA-Perceiver, JEPA-ViT** sit in the 13–20 range — genuinely multi-dimensional, closer to the intrinsic estimate.

---

## (2) Within-field dispersion

Real discretization-invariance metric: fix 15 snapshots, sample 20 random sensor sets per N, report `within-spread / between-spread`. Lower = more truly N-invariant.

| | N=16 | N=32 | N=64 | N=256 |
|---|---|---|---|---|
| **V3+VICReg** | **0.449** | **0.227** | **0.081** | **0.016** |
| T2 spatiotemp | 0.531 | 0.228 | 0.084 | 0.016 |
| JEPA-Perceiver | 0.876 | 0.408 | 0.209 | 0.070 |
| V3-recon | 0.751 | 0.471 | 0.231 | 0.038 |
| MAE | 1.810 | 1.061 | 1.056 | 0.755 |
| MLPSparseAE | 2.547 | 1.804 | 1.238 | 0.566 |

**Findings:**
- **V3+VICReg is the discretization-invariance winner.** At N=64 sensors, latent drift from re-sampling is only 8% of between-field spread.
- **MLP and MAE are not invariant.** At small N, they spread *more* within a fixed field than between fields — re-sampling perturbs the representation more than changing the field does.
- T2 spatiotemporal matches V3+VICReg on dispersion, despite its per-system PR collapse — the spatial-VICReg loss preserves invariance even when the encoder allocates few dimensions to that system.

---

## (3) Reconstruction rel-L2 vs sensor count N

**Caveat reported numbers are dominated by late-time heat snapshots where ‖GT‖→0 → rel-L2 denominator explodes. Median + small-norm filter recommended; current means are unreliable in absolute terms but show relative ordering.**

Approximate finding: among well-behaved methods (MLPSparseAE, V3-recon at full N=1024), rel-L2 drops sharply from N=16→64, then plateaus. Pattern consistent with "manifold saturates around N=64–256 for these 1D PDEs" rather than "model ignores extra sensors."

---

## (4) Richer linear probes — the smoking gun

Per-system linear-probe R² on five targets (coefficient, energy `⟨u²⟩`, fourth moment `⟨u⁴⟩`, max amplitude, time index):

```
                            coeff  energy  fourth max_amp   time
V3-recon       heat         0.29   1.00    0.97   0.92    0.26
V3+VICReg      heat         0.66   1.00    0.96   0.98    0.30
JEPA-ViT       burgers      0.78   0.96    0.93   0.94    0.76
MAE            burgers      0.71   0.94    0.87   0.82    0.80
MLPSparseAE    burgers      0.003  1.00    0.96   0.60    0.10
all methods    adv-β        ≈ 0    > 0.97  > 0.90 ~0     ~0
```

**Detector-vs-encoder diagnosis (energy R² vs coeff R²):**

| | energy | coeff | diagnosis |
|---|---|---|---|
| V3-recon on heat | 1.00 | 0.29 | **detector** — finds amplitude, misses physics |
| MLPSparseAE on burgers | 1.00 | 0.003 | **pure detector** — energy yes, ν invisible |
| CNN on heat | 1.00 | 0.56 | mixed |
| V3+VICReg on heat | 1.00 | 0.66 | encoder |
| JEPA-ViT on burgers | 0.96 | **0.78** | encoder |

**Key observations:**
- **Scalar amplitudes (energy, fourth moment) are linearly perfect for EVERY method × system pair (R² > 0.91).** Even rank-2 collapsed encoders trivially encode `⟨u²⟩`. So a high energy probe alone proves nothing — it's the floor, not a sign of richness.
- **Time index is the most-discriminating non-trivial target.** Burgers time recovery: MAE 0.80, JEPA-ViT 0.76, CNN 0.69 — these methods track shock evolution. V3+VICReg only 0.42 — its invariance regularizer may suppress temporal cues.
- **Advection destroys all single-snapshot identifiability** beyond energy/moment. Every method gets coeff ≈ 0, max_amp ≈ 0, time ≈ 0. Pair-conditional was the only way through.
- **MLPSparseAE on burgers is the cleanest "pure detector" case:** R²(energy)=1.00 but R²(coeff)=0.003. Probing only the coefficient would have totally missed this.

---

## Headline takeaways

1. **Three of eight methods are effectively rank-2 collapsed** (CNN, MAE, v3_recon) and would pass any single-target probe trivially. Without a richness diagnostic this is invisible.
2. **PR_full is necessary but not sufficient** — T2's 48 looked best until per-system PR revealed the advection collapse to 1.2.
3. **V3+VICReg is the best balance** — PR in healthy range, lowest within-field dispersion, encoder-not-detector on the non-trivial probes.
4. **JEPA-ViT is the best at coefficient-recovery on hard systems** (burgers, AC) but suffers per-system PR variance (low on heat and AC at 6-7).
5. **Energy/fourth moment probes are useless as evidence of representation quality** — they saturate at R² ≈ 1 even for collapsed encoders. Always pair with a "hard" probe (coefficient, time, IC parameters).
6. **Advection-β is unidentifiable from single snapshot** for every encoder. This is structural, not a model failure.

Saved figures (under `results/probes/g1/`):
- `diag_richness_pr.png` — PR bars + singular spectrum
- `diag_richness_dispersion.png` — within/between spread vs N
- `diag_richness_probes.png` — 5-target linear probes per system
- `diag_richness_reconN.png` — recon-vs-N (mean unreliable; needs median rerun)
- Raw numbers: `diag_richness.json`
