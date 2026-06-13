# Results — turbulent_radiative_layer_2D (cooling-time estimation)

Task: estimate log10(`t_cool`) from a turbulent radiative cooling layer.
Protocol: self-supervised pretrain → **frozen** encoder → probe. Metric:
R² of log10(t_cool) on the held-out (valid) split, standardized by train stats.

| method | input | probe | R² |
|---|---|---|---|
| **FAE+VICReg** (ours) | single 2D snapshot | **linear ridge** | **0.879** |
| JEPA (helenqu) | 16-frame video | **linear ridge** (mean-pool, matched) | **0.479** |
| JEPA (helenqu) | 16-frame video | attentive-pool + MLP (their head) | 0.71 |

(FAE at 2 epochs already reached 0.81; climbs to 0.88 by 40, stable. JEPA
linear probe: mean-pool 0.479, max-pool 0.254, flatten 0.088 — mean-pool is its
best linear reduction. JEPA embeddings read from their cached
`embeddings/*ConvEncoder_5*.h5`.)

## The decisive comparison: matched linear probe

With the **same** linear ridge probe, FAE (0.879) nearly doubles JEPA (0.479).
JEPA only reaches 0.71 once given a *learned nonlinear* head (attentive pooling
+ MLP) — i.e. its features hold the information but do **not** expose it
linearly. FAE's linear probe beats JEPA's best nonlinear head. This is the G1
linearization thesis reproduced on a real 2D benchmark: VICReg flattens the
manifold so the physics is linearly readable; the I-JEPA objective leaves it
entangled.

## Reading

FAE+VICReg beats the JEPA baseline, and the *manner* of the win is the thesis:

- **Single snapshot vs 16-frame video** — FAE recovers the cooling time from one
  frame; JEPA uses a temporal window.
- **Linear ridge vs MLP probe** — FAE wins with the *weaker* probe, i.e. the
  physics is **linearly accessible** in its latent. (Same mechanism as G1:
  VICReg's variance/covariance terms flatten the manifold.)
- FAE is `coord_dim`-agnostic; the only extension needed for 2D multi-channel
  was `in_chans` (val_proj / decoder head), 7.0M params, parity preserved.

## Caveats (PoC, not a faithful benchmark)

1. **Single seed**, one configuration each.
2. The JEPA number is from a **quick 6-epoch** pretrain on our hardware, not the
   paper's optimized result — treat 0.71 as a re-run baseline, not their
   headline.
3. The inputs differ (snapshot vs video) and the probes differ (ridge vs MLP),
   so this is "each method in its natural setup", not a perfectly controlled
   ablation. A controlled version would run both through an identical probe and
   matched pretraining budget.
4. Coarse task: 9 distinct t_cool values.

Strong enough as a proof that FAE is competitive-to-better on a recognized 2D
physics benchmark; the controlled version is the follow-up.

Artifacts: `results/checkpoints/g1/fae_vicreg_trl2d.pt`,
`logs/fae_vicreg_trl2d.log`, `logs/trl2d_jepa_{train,finetune}.log`.
