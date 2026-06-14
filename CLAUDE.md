# CLAUDE.md — WFAE project instructions

Read `README.md` and `docs/MIGRATION.md` first. This file is the working brief.

## Current focus (everything else is archived in `arxiv/`)

Compare four self-supervised paradigms on **2D physics fields from The Well**,
by **frozen-encoder linear probe of physical parameters**, with the **trivial
baseline as the floor**:

| method | paradigm | where |
|---|---|---|
| **AE** | reconstruction | FAE recon-only (`train_fae_*` with sim=std=cov=0); ViT-AE in `benchmarks/mae` (mask 0) |
| **MAE** | masked reconstruction | `benchmarks/mae/mae.py` (faithful Kaiming port) |
| **JEPA** | latent prediction | spatio-temporal: `external/` helenqu (3D conv); single-frame: `benchmarks/jepa/ijepa2d.py` |
| **FAE+VICReg** | recon + invariance (ours) | `src/models/fae.py` |

Two experiments: **single-frame** (first) and **spatio-temporal**. JEPA was
originally single-frame (I-JEPA); the helenqu JEPA is genuine 3D-conv video
(downsamples time, frames ∈ {4,16}).

## Benchmarks

- **shear_flow** — THE benchmark. Discriminating: trivial baselines (random
  proj / channel means / PCA) give R² ≈ 0 for (Reynolds, Schmidt). Probe targets
  `logRe, logSc`. Pruned PoC grid: 4 Re × 3 Sc × 32 ICs.
- **trl_2D** — sandbox only, **saturated/trivial** (random-init encoder ~0.91 on
  t_cool). Do not draw method conclusions from it.

## Evaluation = linear probe + trivial baseline ONLY

`scripts/eval_linear_probe.py`: per method, frozen embedding → ridge probe of
(logRe, logSc) → R² and **MSE on standardized labels** (same standard as the
paper's Table 1; trivial predictor → MSE 1.0), plus **participation ratio**
(collapse guard). Richer evals (consistency, dimension, IC-mode, cross-config,
rollout) are archived in `arxiv/` — out of scope now.

## Hard rules (each cost us real time)

1. **Random/trivial baseline FIRST.** Never claim a probe result without it —
   trl_2D's t_cool was beaten by a *random* encoder (0.91). `eval_linear_probe`
   always prints the floor.
2. **PR collapse check.** A high probe on a low-PR latent is a "detector"
   artifact — unless PR ≈ the data's intrinsic dimension (compare to the data,
   not an arbitrary threshold; FAE matched data PR ~6 on trl_2D = not collapsed).
3. **Train baselines authentically.** A 6-epoch JEPA gave a false-weak 0.48;
   40-epoch gave 0.78. Match a fair budget before comparing.
4. **Param match** to ~ViT-Tiny (5.5M). FAE 7.0M / MAE 6.6M / I-JEPA enc 5.0M.
5. **Paper-comparable MSE** = standardized-label MSE averaged over params
   (their normalization: shear means [4.85,2.69] stds [0.61,3.38] compression
   [log, None]). Their shear-flow MSE: JEPA 0.38, VideoMAE 0.67, DISCO 0.13.

## Layout

```
src/      models/fae.py (FAE), data/well2d.py (Well 2D datasets),
          metrics/probes.py (lin_probe, r2_score)
benchmarks/  mae/mae.py (MAE+AE), jepa/ijepa2d.py (single-frame I-JEPA), smoke_test.py
scripts/  train_fae_shear.py, train_fae_trl2d.py (FAE; --temporal=coord_dim 3),
          eval_linear_probe.py
external/ (gitignored) physical-representation-learning (JEPA), mae, the_well_data
arxiv/    G1 1D line + rich evals (tracked, organized, not current)
```

The repo tracks **only algorithm + eval code** — results, plots, weights, data
are gitignored (regenerate; see docs/MIGRATION.md). `THE_WELL_DATA_DIR` env var
sets the Well data root.

## Standing state

G1 1D results stand (archived). 2D: trl_2D is saturated (dead end); shear_flow
is live — FAE single-frame linear-probe MSE ~0.69 (≈ VideoMAE, below JEPA 0.38),
**not competitive yet** and not same-standard (pruned data, linear vs MLP probe,
single-frame vs video). Real work remains: full matrix on the cluster
(single-frame {AE,MAE,I-JEPA,FAE} then spatio-temporal), each with PR + trivial
guards, ideally their MLP-finetune protocol for a same-standard MSE.
