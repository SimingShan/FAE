# fjepa: single-frame functional dynamics-prediction SSL — summary for analysis

## 1. Goal & evaluation protocol

Learn a self-supervised representation of 2D PDE fields (The Well `shear_flow`:
4 channels, here 224×224) such that a **frozen encoder** linearly encodes the
governing physical parameters. Probe targets: `logRe = log10(Reynolds)` and raw
`Sc` (Schmidt). Metric: ridge **linear probe → R²** (per param), plus
**participation ratio (PR)** as a collapse guard. A trivial baseline (random
projection / channel means) is always run first — its floor is logRe ≈ 0.25,
Sc ≈ 0.075. Param-matched to ViT-Tiny (~5.5M); our encoder ~7M.

Reference points (same data, established earlier):
- **Trivial floor:** logRe 0.25.
- **Multi-frame (temporal) encoder:** logRe ~0.55 with 16 input frames.
- **Frames-curve (input frames → probe):** nf1 0.40, nf2 0.39, nf8 0.48, nf16 0.52→0.55.
  (More input frames → better Re. Treated as a near-given, NOT the contribution.)

## 2. The idea

A **single-frame, observation-agnostic, dynamics-aware** functional representation,
trained **without reconstruction**.

- **Observation-agnostic:** the encoder reads a *sparse set* of (coordinate, value)
  samples of the field — any number, any placement — and maps it to a fixed latent
  token set. (Perceiver-style: learned latent queries cross-attend to the samples.)
- **Single-frame input (deliberate):** encode ONE snapshot. This preserves
  inference flexibility — you can probe a single frame, unlike video encoders that
  need the whole clip at inference. This is the core thesis and is non-negotiable.
- **No reconstruction:** avoid turning the method into a sparse-pixel-accuracy
  contest (which would collapse into MAE/AE under another name).
- **Dynamics via latent prediction:** make the representation "dynamics-aware" by
  predicting the *future latent* (not future pixels). A Δt-conditioned predictor
  maps L(t) → L̂(t+Δ); match it to L(t+Δ). The predictor is **discardable** at
  eval (only the encoder is the representation) but is itself an approximate latent
  evolution operator.

## 3. Structure

- **Encoder (FAEEncoder):** Perceiver, emb_dim 320, 128 latent tokens, 4 iterations
  of cross+self attention. Input: sparse {(x,y), u(x,y)} tokens with Fourier
  coordinate features. Output: 128 latent tokens. `represent()` = mean-pool over
  the 128 tokens → the probed vector.
- **Predictor (TokenPredictor):** Δt-conditioned (Fourier embed of Δ → added to
  tokens), self-attention over the 128 tokens. Discardable.
- **Loss = VICReg, adapted for time (the JEPA-flavored reframing):**
  - Two views are **different times** t and t+Δ (each sparsely, independently sampled).
  - **invariance:** MSE(proj(L̂b), proj(Lb)) with **stop-grad on Lb** — i.e. the
    future must be *predictable*, NOT "L(t) ≈ L(t+Δ)" directly (that would force
    time-invariance = collapse on purpose).
  - **variance + covariance:** on proj(La) only (the online rep we probe),
    **across the batch** — prevents sample-axis collapse / redundant dims.
  - Sparsity differs between the two views → also enforces observation-invariance.

## 4. The elegance (why it's appealing)

- One coherent object: a coordinate-set latent state, invariant to sensor
  placement, whose evolution is predictable in latent space.
- No decoder, no pixel loss — the claim is about the *representation*, not reconstruction.
- Single-frame → maximal inference flexibility.
- The predictor falls away cleanly, leaving just an encoder.
- Anti-collapse (VICReg) is load-bearing and well-understood.

## 5. What we tried, and the results (single-frame, linear probe logRe)

| variant | logRe | PR | note |
|---|---|---|---|
| trivial floor | 0.25 | — | baseline to beat |
| **static-VICReg** (two sparse views of SAME frame, NO predictor) | **0.364** (5-seed); 0.325 (1-seed) | 5–9 | the no-prediction control |
| temporal latent-prediction (pooled target) | 0.340 (5-seed); 0.337 (1-seed) | 2–6 | ≤ static |
| token-level prediction (match per-token, not pooled) | ~0.33 (3-seed) | 3–6 | no better |
| EMA target (BYOL-style) | 0.12–0.14 | ~1.0 | **fully collapses** |
| temporal-variance anti-collapse term (τ>0) | 0.00–0.25 (↓ with τ) | →1.0 | **actively destroys probe** |

Secondary findings: short horizon best (dt16 → 0.23); density 256↔1024 within
noise; VICReg essential (EMA/raw collapse). **Nothing single-frame robustly clears
~0.36, and prediction never beats no-prediction invariance.**

## 6. Mechanism / diagnosis (why it fails)

We instrumented forecast-vs-persistence on held-out (t, t+Δ) pairs:
- **Raw cosine** of L(t) vs L(t+Δ) ≈ 1.0 — but this was a **DC-offset artifact**
  (a large shared constant component dominates cosine).
- **DC-removed:** centered cosine ≈ 0.79; **temporal/sample variance ratio**:
  pooled-pred encoder **0.43** (d_time/d_sim; ≈0.32 in variance) — so the latent
  DOES move in time. Token-pred encoder collapses to **0.12** (≈0.056 var).
- The predictor never beats persistence (forecast skill ≤ persistence, often negative).

**Interpretation:** because Re/Sc are **time-invariant labels**, the optimal
representation for the probe is itself **time-invariant**. The latent-prediction
objective is satisfiable by a (near-)time-constant latent — so gradient descent
goes there; latent-prediction even *drives* temporal collapse (token-pred ratio
0.12 < pooled 0.43 < static), making the probe mildly worse, not better. When we
*forced* temporal variance (a VICReg-style anti-collapse on the TIME axis, the term
standard VICReg lacks), the variance ratio rose as intended BUT the probe collapsed
(PR→1, logRe→0): the time-varying part of the field (advecting structure / small
scales) is **not** what sets Re/Sc, so encoding it crowds out the conserved-quantity
signal the probe needs. The only thing that reliably raises the probe is **more
input frames** — i.e. letting the encoder *observe* the evolution (0.40→0.52→0.55) —
which the single-frame thesis deliberately forgoes.

## 7. The crux / open question

Given the two hard constraints — **(a) single-frame input** (flexibility) and
**(b) no reconstruction** — is there ANY self-supervised objective that makes a
single-frame functional encoder **exceed plain same-frame invariance (~0.36)** for
probing time-invariant physical parameters?

Or is ~0.36 a genuine **information ceiling**: one snapshot underdetermines a
parameter that is most cleanly read from the *evolution*, so "dynamics-aware"
single-frame SSL cannot, even in principle, beat static invariance — and the
"dynamics-aware" framing is fundamentally at odds with probing time-invariant labels?

Candidate directions not yet tried: spatially-localized / coordinate-anchored latent
tokens (vs global learned queries); predicting a future *summary statistic* rather
than the full latent; a different downstream target that is itself dynamical rather
than a constant parameter.
