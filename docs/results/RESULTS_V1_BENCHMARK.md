> **Historical note (2026-06-10):** these numbers were computed on the *old* PDEBench-sourced G1 data, before the 2026-06-02 regeneration with continuous coefficients (`gen_g1_all.py`). Current checkpoints and all later reports use the new data; method names predate the v3→FAE rename. Kept for provenance.

# Overnight 1D v1 — G1 multi-PDE training + evaluation

**Date**: see file mtime.  Run by orchestrator `scripts/_pipeline/overnight_1d_v1.sh`.

## Setup

- **Dataset**: G1 = 4 PDE systems (heat, advection, burgers, diff-sorp), 5000 trajectories per system,
  each trajectory subsampled to T=100 frames at X=1024 spatial resolution.
  Combined into one balanced 4-class corpus (20000 trajectories total).
- **Coefficients available**: heat (ν), advection (β), burgers (ν); diff-sorp fixed.
- **Methods**: V3 recon-only, V3 + VICReg, MLP-sparse, CNN-1D, MAE-1D — all ~7M params.
- **Training**: 20 epochs, batch 32, lr 5e-4, multi-PDE classification + per-system regression.

## Results table

| Method | tier | params | Probe heat ν | Probe adv β | Probe burg ν | LogReg | kNN | AdvF1 | Consistency cos | Var-subsets |
|---|---|---|---|---|---|---|---|---|---|---|
| v3_recon | sparse | 6.98M | 0.692 | -0.016 | 0.267 | 0.944 | 0.924 | 0.825 | 0.999 | 33.403 |
| v3_vicreg | sparse | 6.98M | 0.738 | -0.021 | 0.442 | 0.959 | 0.953 | 0.898 | 1.000 | 2.924 |
| mlp | sparse | 6.99M | 0.608 | -0.023 | -0.010 | 0.674 | 0.918 | 0.818 | 0.824 | 0.001 |
| cnn | dense | 7.55M | 0.838 | -0.027 | 0.195 | 0.847 | 0.944 | 0.878 | n/a | n/a |
| mae | dense | 7.18M | 0.888 | 0.001 | 0.311 | 0.962 | 0.958 | 0.910 | n/a | n/a |
| random baseline | baseline | n/a | -0.019 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Plots

Saved under `results/probes/g1/`:
- `probe_bars.png` — probe + classification bar chart for all methods
- `tsne_grid.png` — per-method t-SNE of val embeddings, colored by PDE class
- `consistency.png` — two-subset latent agreement + variance for sparse methods

## Notes

- **MAE-1D and CNN-1D are dense-grid methods**: they cannot be evaluated on
  consistency-under-partial-observation or sparse-recon-vs-N axes. Marked 'n/a'.
- **Diff-sorp has no per-traj coefficient**: not included in the linear probe column.
- **Random baseline** uses Gaussian random features at the same dim as encoders;
  any encoder R² below this is meaningless.

## Files

- Checkpoints: `results/checkpoints/g1/{v3_recon,v3_vicreg,mlp,cnn,mae}.pt`
- Raw JSON: `results/probes/g1/g1_all.json`
- Embedded latents: `results/probes/g1/emb_{method}.npz`