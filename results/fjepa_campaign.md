
## Round 1 (seed0,150ep) — attn>mlp(0.358/0.330), L1>L0(0.353/0.336), dt/depth neutral, no collapse.
   best attn_L1: 0.379 logRe (=FAE-single), Sc~0.29. fjepa matches recon baseline.

## Round 2 (seed0,150ep): EMA COLLAPSES (PR~1) -> vicreg only. short dt wins (dt16=0.228 bad).
   std>=75 hurts (R1 L1-win was noise). denser>sparse (sp64 0.304 < sp256 0.390). all 1-seed/noisy.
## Round 3: SEED-heavy. temporal-best vs static (5 seeds each) + density curve, recipe=vicreg attn dt4 std50 cov2.

## Round 3 (5 seeds,150ep) DECISIVE: static-VICReg 0.364 >= temporal 0.340 (PR also higher).
   Temporal PREDICTION does NOT beat no-prediction invariance at single-frame. 60ep "temporal>static" was noise.
   denser helps (1024~0.368), dt1~dt4. -> dynamics objective not validated single-frame; needs temporal ENCODER.
## Round 4: static density curve (winning recipe) w/ seeds + temporal@dense check. Round 5: forecast diagnostic.

## Round 4/5 (seeded + FORECAST DIAGNOSTIC): MECHANISM FOUND.
   persist(cos La,Lb pooled)=0.96-1.00 at ep150/healthy-PR => mean-pooled rep is TIME-INVARIANT.
   fcast<persist everywhere (predictor cant beat copy). => temporal collapses to static BY CONSTRUCTION (pooling).
   Density static 256~0.367 >= 1024~0.334 (noise plateau). Whole logRe landscape ~0.33-0.40 regardless.
   FIX: predict at TOKEN level (don't pool the target) so predictor must learn per-token dynamics.

## Round 5-6 (forecast diagnostic + token-level fix) — DECISIVE MECHANISM
   ep150, healthy PR 6-8: pooled persist=0.96-1.00 (rep is TIME-FLAT, not collapse).
   Token-level fix (--token_pred, match per-token no pooling): tpersist=1.000 at PR 4-6.5.
   => dynamics washed out BEFORE pooling. NOT pooling -- the Perceiver GLOBAL-QUERY bottleneck:
      128 learned queries each grab a conserved global mode (=> Re/Sc probe-able) => latent set
      is time-invariant token-by-token. fcast<=persist always (often negative). Probe stays ~0.32-0.37.
   CONCLUSION: at single-frame, latent-prediction SSL is degenerate (identity-optimal) -> temporal=static.
      Dynamics needs either temporal ENCODER (multi-frame, loses flexibility) or SPATIALLY-LOCALIZED
      tokens (coord-anchored latents that track advecting local state) -- an architecture change.
