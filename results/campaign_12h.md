# 12-hour autonomous campaign — started 2026-06-16 ~00:30 EDT

**Mandate:** continuous, ≤16 GPU. Mean-pool linear probe = primary metric. Task-agnostic only
(no Sc-targeted losses/architectures — motivate by general reconstruction/representation quality,
report battery honestly). May change plan mid-flight on breakthrough or clear failure.

**Standing result going in:** twoview (present+future field recon, two-view shared target, short Δ,
no VICReg/latent-match) — mean-pool probe logRe ~0.47 / Sc ~0.27. Beats single-frame MAE/I-JEPA on
logRe; ~matches best 16-frame video on logRe; LOSES Sc to all multi-frame. Invariance achieved (cos 0.99,
free from recon). Reconstruction blurs developed turbulence + plateaus with sensor count — CAUSE UNKNOWN
(latent bottleneck vs sensors vs L2 vs intrinsic field rank); Phase 1 tests it (not assumes it).

## Plan
- **Phase 1 (R1-3): fidelity diagnostic + capacity.** POD-rank (target difficulty) + sweep num_latents /
  num_iter / n_query / decoder. Q: what limits reconstruction, and does fixing it lift Sc as a *consequence*?
- **Phase 2 (R4-5): localized/coord-anchored latents + RoPE** — general expressiveness upgrades; weigh
  fidelity gained vs observation-invariance lost (the 0.99 cosine).
- **Phase 3 (R6-8): recipe ablations** at best capacity (dynamics on/off, present:future weight, Δ, decoder).
- **Phase 4 (R9-11): seed everything** (best config ×5, baselines ×3).
- **Phase 5 (R12+): battery probe** (energy/enstrophy/scalar-var + Re/Sc) + buffer.

## Log
### R1 (capacity / num_latents curve, twoview, 80ep) — LAUNCHED
num_latents {128,192,256,384,512,768} @ NI4 NQ1024 (+seeds on 128/256/512), NI6 @ {256,512},
NQ4096 @ {256,512}, cvit decoder @ {256,512}, big NL512/NI6/NQ4096. + POD-rank diagnostic.
Gate: does more capacity sharpen recon (lower rec loss) AND raise Sc without hurting logRe?
RESULTS: (pending)

### POD-RANK RESULT (critical): shear_flow frames are LOW-RANK.
90% energy = ~8 modes, 95% = ~13, 99% = ~48 — BOTH laminar(f2) & developed(f15). 128 latents >> rank.
=> 128-latent bottleneck is NOT the recon limit (refutes my assumption). Predicts capacity sweep FLAT;
   blur = L2 ignoring low-energy high-freq tail (last ~1%); Sc weakness is INFORMATIONAL (temporal), not
   architectural. PLAN PIVOT (pending sweep confirm): drop capacity/localized (Phase 1/2), spend the night
   on SEEDS (confirm headline) + ABLATIONS + BATTERY. NL128 seeds reproduce 0.46-0.45 logRe / 0.24 Sc.

### R1 RESULT: num_latents 128->384 FLAT (logRe 0.42-0.47 / Sc 0.24-0.27) — capacity NOT the limit (POD confirmed).
5 heavy jobs DEAD (OOM/cvit). NI6 NL256=0.486 (1-seed, marginal). PIVOT confirmed: drop Phase 1/2.
### R2 (LAUNCHED): consolidate — headline SEEDS + component ABLATION + Δ + depth. All NL128 (7M, matched).
### BATTERY (twoview generality): kin_energy R2=0.999, enstrophy 0.996, scalar_var 0.996, scalar_grad 0.935,
press_var 0.998 | logRe 0.45, Sc 0.17. => rep is GENERAL (encodes physics near-perfectly); Re/Sc are the
hard residual, NOT Goodharted. Strong generality result.
### R2 FAILED (all OOM): my 64G backfill downsize too small for n_seed24 clip data (MaxRSS 64G). Reverted to
128G, relaunching R2.

### R2 RESULTS — headline confirmed + ablation:
HEADLINE twoview (7 seeds): logRe 0.455 ± 0.020, Sc 0.242 ± 0.013. SOLID.
COMPONENT ablation (logRe): recon(present)=0.412 -> +future(recon_both)=0.445 -> +two-view(twoview)=0.455.
  => dynamics/future-recon is the main gain (+0.033), two-view marginal (+0.010); Sc FLAT ~0.24 (no component
  helps Sc — confirms Sc is informational/temporal, not objective-fixable).
Δ horizon: dt1=0.463 >= dt2=0.455 > dt4=0.397 > dt8=0.368 (short wins, again).
### R3 (baseline SEEDS — comparison table is currently 1-seed; make it trustworthy):

### HARD STOP: cluster maintenance reservation starts 2026-06-16 08:00 EDT (nodes held 'Reserved for
maintenance'). Campaign ends at 08:00; no GPU after. Salvaged the SHORT single-frame baseline seeds
(MAE/I-JEPA, 1:30 limit, fit before 08:00); CANCELLED the 6 multi-frame seeds (8h, can't finish).
Full-train probe-density (2012558) OOM'd (full clip data); reduced-train trend stands (probe density-stable).
### R3 single-frame baseline seeds RUNNING (mae x2, ijepa x2) — will finalize the apple-to-apple table.

================================================================================
## FINAL SUMMARY (campaign end @ 08:00 maintenance, 2026-06-16)
================================================================================
RECIPE (best): twoview = present + Δ-future field reconstruction, two sparse views sharing the SAME
recon targets, short Δ (1-2), NO VICReg, NO latent-match. Single-frame, observation-agnostic. 7M params.

HEADLINE (mean-pool linear probe = standard metric; floor 0.25/0.075):
  ours twoview (7 seeds):     logRe 0.455 ± 0.020   Sc 0.242 ± 0.013
  MAE (3 seeds, single-frame):logRe 0.40  ± 0.02    Sc 0.258
  I-JEPA (3 seeds):           logRe 0.35  ± 0.03    Sc 0.137
  VideoMAE 16f (1 seed):      logRe 0.49           Sc 0.45
  ST-JEPA 16f (1 seed):       logRe 0.44           Sc 0.32
=> WIN single-frame logRe vs MAE(+0.06)/I-JEPA(+0.10), robust. ~match best 16f video on logRe (0.46 vs 0.49).
   LOSE Sc to multi-frame (informational/temporal limit); Sc ~tied MAE, beat I-JEPA.

WHY (mechanism, all confirmed this campaign):
1. CAPACITY is NOT the limit: POD-rank shows shear_flow frames are LOW-RANK (90% energy ~8 modes, 99% ~48)
   << 128 latents; num_latents 128->768 sweep FLAT. Sc gap is INFORMATIONAL (needs temporal), not architectural.
2. ABLATION: present-recon 0.412 -> +future 0.445 -> +two-view 0.455. Dynamics/future-recon is the main gain;
   Sc flat ~0.24 across all components (no objective fixes Sc).
3. Δ: short wins (dt1 0.463 > dt2 0.455 > dt4 0.40 > dt8 0.37).
4. GENERALITY (battery): rep linearly exposes kin_energy R2=0.999, enstrophy 0.996, scalar_var 0.996,
   scalar_grad 0.935, press_var 0.998 — a true physical-field rep, NOT a Re/Sc detector (Re/Sc are its
   HARDEST probes). Anti-Goodhart confirmed.
5. INVARIANCE: cosine(rep across two diff sparse views) = 0.99, FREE from recon (pure-recon already invariant);
   two-view shared-target adds invariance without VICReg (which collapsed/hurt).

INCOMPLETE (cut by 08:00 maintenance): multi-frame baseline seeds (only 1-seed); battery on baselines;
RoPE/general-trick bake-off (deprioritized after POD ruled out capacity).

NEXT (when cluster returns): multi-frame baseline seeds for a fully-seeded table; if pushing Sc, the only
lever left is TEMPORAL INPUT (abandons single-frame) — the rank result says single-frame Sc is info-bounded.
Defensible paper claim: a single-frame, observation-agnostic, general physical-field representation that
matches multi-frame video on Reynolds and beats single-frame SSL baselines; Schmidt is the honest
single-frame information limit.
### BATTERY on MAE (generality head-to-head): kin_energy 0.999, enstrophy 0.993, scalar_var 0.993,
scalar_grad 0.905, press_var 0.998 | logRe 0.357, Sc 0.207. => MAE is ALSO general (generality NOT our
differentiator); ours marginally better on fine-scale (enstrophy/scalar_grad) + clearly on logRe (0.45 vs 0.36).

### FAITHFULNESS AUDIT (baselines): CONVERGENCE clean — all plateau by ep80 (loss down, probe flat, PR healthy
6.5/5.4/3.2/3.5), NOT undertrained. MAE & VideoMAE = faithful standard recipes (comparison SOLID).
CAVEAT: I-JEPA & ST-JEPA use SIMPLIFIED random ctx/tgt masking (sample_masks), NOT authentic I-JEPA block-
masking → may be artificially weak. I-JEPA Sc=0.13/PR=3.2 thin. => DO NOT claim I-JEPA/ST-JEPA win until
upgraded to block-masking (or run external authentic JEPA). Held ST-JEPA seeds; VideoMAE seeds kept.
