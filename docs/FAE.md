# FAE — Function AutoEncoder

The Function AutoEncoder is our PDE-field representation learner. It takes any
number of sparse coordinate–value pairs as input, summarizes them into a fixed
set of M learned latent tokens via Perceiver-style cross-attention, and decodes
back to per-coordinate values via cross-attention readout at arbitrary query
coordinates. The deterministic core training recipe is multi-view SSL with
VICReg regularizers + multi-count sensor sampling; the probabilistic variant
(FAE-NP) replaces the token latent with a single global Gaussian z trained
with a Neural-Process ELBO.

Code: `src/models/fae.py` (deterministic), `src/models/fae_np.py` (NP variant).

## Architecture (G1 configuration, ~7M params)

| Component | Detail |
|---|---|
| Input | sensor values `u ∈ R^{B×N×1}` + coordinates `coords ∈ [0,1)^{B×N×1}`; N arbitrary (16–1024+ tested) |
| Coord features | `fourier_features`: linear-spaced, Nyquist-capped; `freqs = linspace(1, max_freq/2, n_freq)`, sin/cos. `n_freq=32` → 64 dims per coord |
| Token embedding | `coord_proj: Linear(64→288)` ⊕ `val_proj: Linear(1→32)` → `emb_dim = 320` |
| Latent set | M = 128 learned tokens, init N(0, 0.02²) clamped to ±2σ |
| Encoder (`FAEEncoder`) | Senseiver pattern: distinct `layer_1` + shared `layer_n` reused (num_iter−1)=3 times. Each layer = 1 cross-attn (4 heads, latents→tokens) + 4 self-attn blocks (8 heads), all residual pre-LN with LN-D-D MLPs. Cost O(M·N) — linear in N |
| Decoder (`SenseiverDecoder`, default) | query = proj(concat[γ(x_q), learned output buffer]); 1 cross-attn block vs latent tokens; `Linear(320→1)` head, no terminal LN |
| Decoder (`CViTDecoder`, option) | num_blocks × [cross-attn → MLP(2×) → self-attn(queries) → MLP(2×)] |
| Pooled representation | `tokens.mean(dim=1)` ∈ R^320 — used by every probe/diagnostic |

## Training — fae_vicreg (`scripts/train_fae.py`)

Two random views A and B of the same field per batch step:

```
n_A, n_B ~ Uniform({64, 128, 256, 512, 1024})          # multicount
idx_A, idx_B ~ independent random sensor subsets
q_idx        ~ RandomPerm(1024)[:512]                   # shared query points

(pred_A, tokens_A) = FAE(u_A, coords_A, q_coords)
(pred_B, tokens_B) = FAE(u_B, coords_B, q_coords)

L_rec   = ½·[MSE(pred_A, target_q) + MSE(pred_B, target_q)]
xz, yz  = projector(pool_A), projector(pool_B)          # 320→8192→8192→8192
L_align = MSE(xz, yz)
L_var   = VICReg variance hinge on xz, yz
L_cov   = VICReg covariance penalty on xz, yz

L_total = 1·L_rec + 25·L_align + 25·L_var + 1·L_cov
```

AdamW(lr 5e-4, wd 1e-4), 2-epoch warmup → cosine, batch 32, grad-clip 1.0,
20 epochs. The projector is training-only and discarded. The twin/Siamese
structure (two forward passes through the shared encoder) is training-only;
inference is one pass.

`fae_recon` is the same model with L_rec only (the ablation that revealed
rank-collapse — see docs/results/RICHNESS_DIAGNOSTICS.md).

## FAE-NP (`scripts/train_fae_np.py`)

- `FAEEncoder` tokens → mean-pool → `GaussianLatentHead` → (μ, logσ²) of
  dim 256; logvar clamped to (−6, 1).
- z → `LatentToContext` (pure projection — a FiLM-on-bank design collapses
  the encoder) → 64 context tokens → `HeteroscedasticCViTDecoder` →
  per-query (μ_y, logσ²_y).
- Loss: recon (MSE for v1; heteroscedastic NLL once collapse is controlled)
  + β·KL(q(z|C,T)‖q(z|C)) with per-dim free bits; β warmup after a recon-only
  phase. Per-batch: sensors split into context C and sensor-targets, plus
  off-context query positions to force inference.
- Why a single global z: a per-token Gaussian feeding a cross-attention
  decoder lets the decoder average away noise across L=128 tokens
  (effective σ/√L), so the encoder wins with wide-uninformative posteriors.

**Open issue (2026-06-09 runs)**: with β ∈ {1e-4, 1e-3} the posterior logvar
saturates at the clamp upper bound (σ_active_frac = 0) and the *linear*
coefficient probe trails FAE+VICReg badly (heat: 0.42 vs 0.97) while the MLP
probe is comparable (0.95) — information is present but not linearized.

## Lineage

| Element | Senseiver (Lim et al.) | Token-OPERA (ours, prior) | **FAE (current)** |
|---|---|---|---|
| Encoder | M=8–32 latents, cross+self attn | self-attn over all N tokens | M=128 latents, cross+self attn |
| Encoder cost | O(N) | O(N²) (OOM at large N) | O(N) |
| Decoder | coord cross-attn to latents | coord cross-attn to input tokens | coord cross-attn to latents |
| Loss | recon only | recon + SSL alignment + VICReg | recon + alignment + VICReg / NP ELBO |
