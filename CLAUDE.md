# CLAUDE.md — WFAE project instructions

Read `README.md` first; it is accurate. This file adds the working rules and
context an agent needs. It replaces all earlier CLAUDE versions (the old
Stage-3 plan with burgers_shock/KS, P1/P2 and FINDINGS_v3 is dead — archived
in `../WFAE_attic/CLAUDE_v3_old.md`).

## What this project is

We build **FAE — Function AutoEncoder** for PDE fields, in two flavors:

1. **FAE + VICReg** (deterministic, the working core): `src/models/fae.py`,
   trained by `scripts/train_fae.py --method fae_vicreg`.
2. **FAE-NP** (probabilistic, functional Neural Process with a single global
   Gaussian z): `src/models/fae_np.py`, trained by `scripts/train_fae_np.py`.
   Known open issue: posterior logvar saturates at its clamp upper bound
   (σ uninformative) and the linear coefficient probe trails VICReg.

Everything is measured on the **G1 benchmark**: the four 1D PDEs — heat,
advection, burgers, reaction_diffusion (called **AC** in tables/discussion;
it is Fisher-KPP, see `data_gen/gen_g1_all.py`). The paper framing is an
*evaluation framework* for PDE representations: coefficient probes, class
structure, consistency under partial observation, richness diagnostics
(participation ratio, dispersion, IC-mode spectral probes), cross-config
transfer. Reconstruction accuracy alone is treated as a weak proxy.

## Hard rules (each one has burned us before)

1. **Fair-comparison rule.** Never compare reconstruction accuracy across
   methods trained with different objectives. Latent-space metrics (probes,
   classification, consistency, richness) are the only cross-method currency.
2. **Shuffle before split.** Use `src.data.g1.train_val_split`. An ordered
   split once produced probe R² ≈ -1e15 and cost a day.
3. **Per-snapshot training/eval** at the mid-frame; native grid 1024.
4. **~7M parameter parity** (≤10%) for any method added to the benchmark.
5. **No zero-fill adapters** to give dense methods (CNN/MAE/JEPA-ViT) sparse
   metrics in the headline tables — they get n/a. (Zero-fill appears only,
   explicitly caveated, in the cross-config diagnostic.)
6. **State-dict compatibility.** Model class names were modernized in the
   2026-06 cleanup but module *attribute* names must not change, or the
   trained checkpoints in `results/checkpoints/g1/` stop loading.

## How things are wired

- `src/models/zoo.py` is the single place that knows how to load and encode
  every benchmark method. Scripts iterate `zoo.METHODS`; do not re-implement
  per-method loaders in scripts.
- Checkpoint files are named `<method>.pt` under `results/checkpoints/g1/`:
  fae_recon, fae_vicreg, fae_spatiotemporal (T2), mlp, cnn, mae,
  jepa_perceiver, jepa_vit, plus fae_np_b1e-4 / fae_np_b1e-3.
- The pooled representation is `tokens.mean(dim=1)` for FAE-family models,
  `mu` of q(z|C) for FAE-NP, target-branch mean-pool for JEPA models.
- Results JSONs/figures land in `results/probes/g1/`; written findings go to
  `docs/results/`.

## Naming history (for reading old artifacts)

The 2026-06-10 cleanup renamed v3→fae and v4→fae_np everywhere:
`v3_recon→fae_recon`, `v3_vicreg→fae_vicreg`,
`v3_spatiotemporal→fae_spatiotemporal`, `jepa_perceiver_sparse→jepa_perceiver`,
`jepa_vit1d→jepa_vit`, `v4_np*→fae_np*`. Old JSONs and the reports in
`docs/results/` still use labels "V3-recon", "V3+VICReg", "T2 spatiotemp" —
read them as FAE-recon / FAE+VICReg / FAE-T2. Class aliases (`V3 = FAE`,
`V4 = FAENP`) exist in `src/models` for old notebooks.

Caution: `docs/results/RESULTS_V1_BENCHMARK.md` was computed on the *old*
PDEBench-sourced data (pre 2026-06-02 regeneration); every other report and
all current checkpoints use the regenerated continuous-coefficient data.

Everything removed in the cleanup (old stages, 2D/KS work, third-party
benchmark repos, raw PDEBench files, superseded checkpoints) lives in
`../WFAE_attic/` — see `docs/CLEANUP_MANIFEST.md` for the full mapping.

## Stage-1/2 conventions that still hold

PyTorch; AdamW + cosine with warmup; honest reporting of negative results
(the advection-β unidentifiability finding is a feature, not a bug).
