# Results — turbulent_radiative_layer_2D (cooling-time estimation)

Task: estimate log10(`t_cool`) from a turbulent radiative cooling layer.
Protocol: self-supervised pretrain -> **frozen** encoder -> probe. PR =
participation ratio of the probed representation (collapse check; healthy >~10).

## Corrected result (2026-06-14) — NO valid FAE win; FAE collapses here

| method | pretrain | linear probe R² | nonlinear probe R² | PR |
|---|---|---|---|---|
| **JEPA authentic** | 40 ep | **0.78** | **0.81** | **45.8** (rich) |
| FAE+VICReg (best of 7 anti-collapse configs) | 40–60 ep | 0.87–0.94 | — | **≤ 5.9 (COLLAPSED)** |

**The FAE numbers are invalid** — its latent is collapsed (PR ≤ 6), so the high
probe R² is a "detector" artifact (latent collapsed onto the t_cool axis), not a
representation-quality win. Confirmed by t-SNE (`tsne_trl2d_fae_vs_jepa.png`) and
PR across all 7 sweep configs (snapshot + spatiotemporal): 2.5–5.9, none > 8.

## What was wrong with the earlier (retracted) "FAE 0.88 vs JEPA 0.48"

1. **JEPA was undertrained.** At 6 epochs its linear probe was 0.48; trained
   authentically (40 ep, val loss 18.6->10.2) it is **0.78** with PR 45.8 — its
   features are linearly accessible. The "JEPA needs a nonlinear head" claim is
   withdrawn.
2. **FAE was collapsed.** PR 1.49 — the 0.88 was a detector artifact.

Both legs were flawed (h/t the instinct to train JEPA authentically AND check
FAE for collapse — both were necessary).

## Anti-collapse sweep (snapshot + spatiotemporal), all COLLAPSED

| config | PR | probe (contaminated) |
|---|---|---|
| recononly (no VICReg) | 2.49 | 0.902 |
| lowsim (sim 1) | 5.58 | 0.874 |
| strongvar (std 100, cov 10) | 5.16 | 0.872 |
| highrec (lam_rec 10) | 4.19 | 0.890 |
| combo (n_freq 32) | 5.89 | 0.884 |
| st_strongvar (spatiotemporal) | 3.54 | 0.945 |
| st_combo (spatiotemporal) | 5.49 | 0.941 |

Mechanism: recon-only gives PR 2.5; VICReg variants reach ~6 (so VICReg *helps*
spread, it is not the cause of collapse). The encoder underuses its capacity on
2D turbulent fields — open problem.

## Standing conclusions

- **G1 (1D) results stand** — FAE+VICReg has PR ~20 there (genuinely rich); the
  probe/invariance/dimension findings are real. FAE works in 1D.
- **FAE+VICReg does NOT transfer to 2D turbulence as-is** — it collapses (PR ≤ 6
  even after anti-collapse tuning). A genuine limitation.
- **JEPA authentic is a strong, healthy baseline** on this benchmark (PR 45.8,
  linear 0.78) — not the weak baseline the undertrained run suggested.

Artifacts: `logs/night_fae_*.log`, `logs/night_jepa_auth_*.log`,
`results/probes/g1/tsne_trl2d_fae_vs_jepa.png`.
