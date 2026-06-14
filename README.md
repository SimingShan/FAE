# WFAE — FAE vs self-supervised baselines on 2D PDE representation learning

Does **FAE** (Function AutoEncoder — a sparse, coordinate-set encoder with VICReg)
learn better PDE-field representations than the canonical self-supervised
paradigms? We compare four methods on 2D physics fields from
[The Well](https://github.com/PolymathicAI/the_well), by **frozen-encoder linear
probe of physical parameters**, with the **trivial baseline as the floor**.

| method | paradigm | location |
|---|---|---|
| **AE** | reconstruction | `src/models/fae.py` (recon-only) / `benchmarks/mae` (mask 0) |
| **MAE** | masked reconstruction | `benchmarks/mae/mae.py` (faithful Kaiming He port) |
| **JEPA** | latent prediction | `external/` (helenqu 3D-conv, spatio-temporal) / `benchmarks/jepa/ijepa2d.py` (single-frame) |
| **FAE+VICReg** (ours) | recon + invariance | `src/models/fae.py` |

Two experiments: **single-frame** and **spatio-temporal**.

## Benchmark

**shear_flow** (The Well) — discriminating: trivial baselines give R² ≈ 0 for
(Reynolds, Schmidt), so a probe genuinely measures representation quality.
(trl_2D is a saturated sandbox — a *random* encoder scores 0.91 there; not used
for conclusions.)

## Evaluation — one pipeline

```bash
python scripts/eval_linear_probe.py --method fae --ckpt <ckpt>   # R2 + MSE + PR
python scripts/eval_linear_probe.py --method trivial            # the floor
```
Reports R² and **MSE on standardized labels** (same standard as the source
paper's Table 1) and the **participation ratio** (collapse guard). A method
only counts if it clearly beats the trivial baseline.

## Quickstart

```bash
export THE_WELL_DATA_DIR=/path/to/the_well_data          # see docs/MIGRATION.md
python benchmarks/smoke_test.py                          # verify MAE / AE / I-JEPA
python scripts/train_fae_shear.py --epochs 60 --tag v1   # FAE+VICReg on shear (snapshot)
python scripts/train_fae_shear.py --temporal --tag v1t   # spatio-temporal (coord_dim=3)
```

## Layout

```
src/         models/fae.py (FAE), data/well2d.py (Well 2D datasets), metrics/probes.py
benchmarks/  mae/ (MAE+AE), jepa/ (single-frame I-JEPA), smoke_test.py, README.md
scripts/     train_fae_{shear,trl2d}.py, eval_linear_probe.py
docs/        MIGRATION.md (cluster setup), FAE.md, benchmarks/ (JEPA harness patch+configs)
arxiv/       archived G1 1D line + rich evaluation suite (tracked; see arxiv/README.md)
```

The repo tracks **only algorithm + evaluation code**. Results, plots, weights,
and data are gitignored — regenerate per `docs/MIGRATION.md`. See `CLAUDE.md`
for the working brief and the hard rules (trivial-baseline-first, PR collapse
check, authentic baselines, param matching).
