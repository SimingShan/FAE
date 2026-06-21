# CLAUDE.md — WFAE project instructions

Read `docs/STRUCTURE.md` first (repo layout + how to run). This file is the working brief.

## Current focus — TWO evaluation categories (everything else is in `arxiv/`)

Encoders pretrained on **2D PDE fields from The Well**, evaluated two ways. The encoders under
comparison are the same in both categories:

| encoder | paradigm | where |
|---|---|---|
| **FAE** (ours) | functional / coordinate, dual-view temporal | `src/models/fae.py`, `scripts/train_fae.py` (`--mode twoview` default) |
| **MAE** | masked reconstruction | `benchmarks/mae/mae.py` (Kaiming port), `scripts/train_baseline.py` |
| **JEPA** (I-JEPA) | latent prediction | `benchmarks/jepa/ijepa2d.py`, `scripts/train_baseline.py` |
| **VICReg** | invariance (their SSLForPDEs) | `scripts/train_baseline.py` |

### Category 1 — REPA generation  (`scripts/generate.py`)
Pixel-space SiT flow-matching on PDE fields, **NO VAE**; REPA aligns SiT tokens to a frozen encoder's
per-patch features (`--align {none,fae,mae,jepa}`, `none` = pixel-DiT benchmark). Three modes:
- `uncond` — unconditional. metric `spectrum_dist`.
- `param` — physical parameter as a class → LabelEmbedder + AdaLN + **CFG** (REPA's class mechanism;
  NOT channel-concat). metric `spectrum_dist`.
- `sparse` — **FAE encodes scattered sensors → dense field guess → DiT refines** (ViTs can't ingest
  scattered points). metric `recon_relL2` + `spectrum_dist`. **This is the FAE-necessary regime.**

Core question: *which encoder is the best REPA target for PDE generation, and where is FAE necessary
(not merely preferable)?*

### Category 2 — Linear probe  (`scripts/eval_linear_probe.py`, `eval_ns_probe.py`)
Frozen embedding → ridge probe of physical parameters → R² and **MSE on standardized labels**
(paper Table-1 standard; trivial predictor → MSE 1.0), plus **participation ratio** (collapse guard).

## Benchmarks
- **shear_flow** — THE benchmark. 4-ch [tracer, pressure, vx, vy]; probe `logRe, logSc`. Trivial
  baselines give R² ≈ 0. Pruned PoC grid 4 Re × 3 Sc. The 3 smooth channels dilute pooled gen metrics
  → use **per-channel** spectrum for gen.
- **NS-2D-conditioned** (PDE-Arena buoyancy smoke) — 3-ch [smoke, vx, vy]; probe / condition on
  `buoyancy`. The conditional-generation benchmark.
- **trl_2D** — saturated sandbox, dead end. Don't draw conclusions from it.

## Hard rules (each cost us real time)
1. **Trivial/random baseline FIRST** (probe). Never claim a probe result without the floor;
   `eval_linear_probe` always prints it (trl_2D t_cool was beaten by a *random* encoder, 0.91).
2. **PR collapse check.** High probe on low-PR latent = detector artifact unless PR ≈ data intrinsic dim.
3. **`spectrum_dist` is NOISY** — single-batch ~0.03–0.1 (3-seed std ~0.025 even at 512 samples). A
   false "JEPA wins shear" came from one lucky batch. **Multi-seed / many-sample (≥1024) for ANY gen
   ranking.** This is the #1 generation gotcha.
4. **Train baselines authentically + param-match** (~ViT-Tiny 5.5M; FAE 7.0M / MAE 6.6M / I-JEPA 5.0M).
   MAE/JEPA are mature/tuned; tune FAE's neutral knobs (lr/epochs/lam/depth) equally — but **never
   task-couple the objective** (no Goodhart; FAE tuning was already near-optimal, no hidden win).
5. **Conditioning mechanism matters.** Present-FRAME channel-concat over-determines the field → FAE
   redundant. Parameter AdaLN+CFG (weak/global) → FAE's prior helps again. `train_fae.py` saves by
   default (the `--save` footgun cost a run).

## Layout (post REPA-pivot reorg)
```
scripts/  generate.py · eval_linear_probe.py · eval_ns_probe.py · train_fae.py · train_baseline.py
          viz_generate.py · download_ns.py
src/      models/fae.py · data/{well2d,ns,flowbench}.py · metrics/probes.py · utils/seed.py
benchmarks/ mae/mae.py · jepa/ijepa2d.py
external/ (gitignored) REPA (SiT) · physical-representation-learning (JEPA) · mae · the_well_data
arxiv/pre_repa_pivot/  ~80 archived files (rollout, flowbench, diagnostics, one-off evals, old trainers)
```
Repo tracks **only code** — results, plots, weights, data are gitignored (regenerate).
`THE_WELL_DATA_DIR` sets the Well data root.

## Standing state (REPA generation results; all single-seed unless noted — see rule #3)
- **Unconditional:** FAE best (NS 0.241 vs pixel 0.265) / tied-best (shear robust-3-seed 0.157 ≈ jepa
  0.163, vs pixel 0.187); **MAE consistently hurts**. FAE is the only target top-tier on *both*.
- **Param-cond (AdaLN+CFG):** shear FAE 0.128 **clearly beats** pixel 0.179; NS a tie.
- **Cond rollout/recons (present→future):** sharp + accurate (NS relL2 0.11, shear 0.05) but FAE no
  help (conditioning over-determines). Autoregressive: shear stable ~0.04, NS drifts 0.10→0.53.
- **Tuning flat:** alignment-hp + 200-ep FAE gave no gain → competitiveness is genuine, not under-tuned.
- **Open (Prof Lu): story = why FAE / why FAE-for-PDEs.** Answer lies in `sparse` + multi-resolution
  generation, where fixed-grid ViTs *can't compete* — pivot experiments there. See memory
  `repa-generation.md`.
