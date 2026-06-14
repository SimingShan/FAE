# arxiv/ — archived, tracked, organized

Code and findings that are **not** part of the current comparison
(AE vs MAE vs JEPA vs FAE+VICReg on 2D Well data, evaluated by linear probe +
trivial baseline). Kept under version control (not gitignored) for provenance
and possible reuse — but out of the main `src/` / `scripts/` / `benchmarks/`.

## Layout

```
arxiv/
├── src/
│   ├── models/   fae_np.py (FAE-NP), jepa_vit.py (1D I-JEPA), baselines.py
│   │             (MLP/CNN/MAE-1D), zoo.py (G1 model registry)
│   ├── data/     g1.py (G1 1D multi-PDE loader)
│   └── metrics/  consistency, classification, cross_coefficient, intrinsic_dim,
│                 linear_probe, random_baseline, sparse_recon
├── scripts/      diag_* (richness / dimension / IC-mode / cross-config /
│                 off-grid / token-readout), eval_fae_np*, train_fae(_np/_temporal),
│                 train_jepa_*, evaluate.py (G1 8-method), viz_*
├── data_gen/     G1 1D generators (gen_g1_all.py)
└── docs/         G1 result reports (RESULTS_V1, RICHNESS_DIAGNOSTICS,
                  IC_MODES_AND_CROSS_CONFIG), DATA.md, CLEANUP_MANIFEST.md
```

## What it is

- **G1 (1D) line**: the multi-PDE benchmark (heat/advection/burgers/AC), the
  FAE-NP (Neural Process) variant, and the rich evaluation suite (participation
  ratio, within-field dispersion, intrinsic dimension, IC-mode spectral probes,
  cross-config transfer, consistency). These results stand (FAE+VICReg PR ~20 in
  1D) but are not the current 2D focus.

## Running archived code

These scripts import `from src.models import ...` etc. expecting the **main**
`src/` (which still has `fae.py`). The 1D-specific modules now live here, so to
run archived code, either copy the needed file back into `src/` temporarily or
add `arxiv/` to the path. The archive is for reference, not turn-key.
