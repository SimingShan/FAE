# Cleanup manifest — 2026-06-10 deep reorganization

Nothing was hard-deleted. Everything removed from the repo was **moved to
`/mnt/crucial/shansiming/project/WFAE_attic/`** (~69 GB), preserving relative
paths. Once you have sanity-checked the clean repo, the attic can be removed
with `rm -rf /mnt/crucial/shansiming/project/WFAE_attic`.

## Renames (kept, new names)

### Code modules
| old | new |
|---|---|
| `src/models/v3.py` (`PerceiverSparseAEV3`, `PerceiverEncoderV3`, `DecoderV3`, `DecoderCViT`) | `src/models/fae.py` (`FAE`, `FAEEncoder`, `SenseiverDecoder`, `CViTDecoder`) — aliases for old names kept |
| `src/models/v4.py` (`V4`, `HeteroscedasticDecoderCViT`) | `src/models/fae_np.py` (`FAENP`, `HeteroscedasticCViTDecoder`) — aliases kept |
| JEPA-ViT classes inline in `scripts/train_jepa_vit1d.py` | `src/models/jepa_vit.py` |
| per-script model loaders/encoders (duplicated ×3) | `src/models/zoo.py` (single registry; verified to reproduce diag_richness PR_full exactly) |
| — | `src/metrics/probes.py` (shared lin/MLP probe + rel-L2 helpers) |

State-dict layouts are untouched: all pre-cleanup checkpoints load strict.

### Scripts
| old | new |
|---|---|
| `train_1d.py` | `train_fae.py` (methods `v3_recon/v3_vicreg` → `fae_recon/fae_vicreg`; `v3_vicreg_asym` dropped) |
| `train_v4_np.py` | `train_fae_np.py` |
| `train_v3_temporal.py` | `train_fae_temporal.py` |
| `train_v3_jepa_perceiver.py` | `train_jepa_perceiver.py` |
| `train_jepa_vit1d.py` | `train_jepa_vit.py` |
| `evaluate_1d.py` | `evaluate.py` (now covers all 8 methods via the zoo + AC coefficient probe) |
| `diag_richness_g1.py` | `diag_richness.py` |
| `pair_probe_v2.py` | `diag_pair_probe.py` |
| `eval_v4_np.py` / `eval_v4_ensemble.py` / `viz_v4_ensemble.py` | `eval_fae_np.py` / `eval_fae_np_ensemble.py` / `viz_fae_np_ensemble.py` (`--v4_ckpt` → `--np_ckpt`) |

### Checkpoints (`results/checkpoints/g1/`)
`v3_recon.pt→fae_recon.pt`, `v3_vicreg.pt→fae_vicreg.pt`,
`v3_spatiotemporal.pt→fae_spatiotemporal.pt`,
`jepa_perceiver_sparse.pt→jepa_perceiver.pt`, `jepa_vit1d.pt→jepa_vit.pt`,
`v4_np_b1e-4.pt→fae_np_b1e-4.pt`, `v4_np_b1e-3.pt→fae_np_b1e-3.pt`.
(`mlp/cnn/mae.pt` unchanged.)

### Docs
`overnight_1d_v1.md → docs/results/RESULTS_V1_BENCHMARK.md` (old-data caveat
added); `results/probes/g1/{RICHNESS_DIAGNOSTICS, IC_MODES_AND_CROSS_CONFIG}.md
→ docs/results/`; `FAE_schematic.md` rewritten as `docs/FAE.md`; `DATA.md`,
`README.md`, `CLAUDE.md` rewritten.

Old result JSONs/embeddings in `results/probes/g1/` keep their historical
keys/names (`v3_recon`, `emb_v3_*` …); re-running `evaluate.py`/diags writes
the new names.

## Archived (→ WFAE_attic/, same relative paths)

- **Third-party reference repos**: `benchmarks/` (cvit, fundiff, mae_pde,
  senseiver), `external/ijepa` (672 MB)
- **Stage-1/2 configs**: `configs/` (v2 + v3 incl. E01–E12, _heat2d/_heat3d/
  _kalman/_nor/_opera/_symssl)
- **Old plan/handoff docs**: `CLAUDE_v3_old.md` (the Stage-3 plan),
  `HANDOFF_G1.md`, `README_old.md`, `docs/_audit_*.md`, `DATA_old.md`,
  `FAE_schematic_old.md`
- **Old src modules**: `src/training/` (v2-era loops), `src/generation/`,
  `src/viz/`, `src/data/family_1d.py`, `src/data/g2.py` (KS/2D)
- **Stale scripts** (~45): old `train.py`/`evaluate.py`/`probe*.py`, all 2D/G2
  (`*_2d.py`, `*_g2.py`), rollout/latent-dynamics exploration
  (`eval_rollout_*`, `eval_latent_dynamics`, `eval_query_propagator`,
  `viz_rollout_*`, `viz_latent_dynamics`, `viz_query_prop`), fixed-N ablation
  drivers, `shortcut_check*`, `pair_probe.py` (v1), `train_v3_jepa.py`,
  jepa-decoder experiments, one-off viz, `write_overnight_report.py`,
  `scripts/_pipeline/` orchestrators
- **Superseded data generators**: `combine_g1.py`, `download_pdebench.py`,
  `gen_heat_1d.py`, `gen_advection_1d.py`, `gen_reaction_diffusion_1d.py`,
  `gen_ks_1d.py`, `gen_g2_all.py`, `visualize_g2.py` (current data comes from
  `gen_g1_all.py` alone)
- **Data** (~60 GB): raw PDEBench hdf5 (advection/burgers), old heat splits
  (`heat_train/valid/ood.h5`), `data/2d/`, empty placeholder dirs
- **Checkpoints** (~7.5 GB): `g1_v1_olddata/`, `g2/`, all `senseiver_*`,
  `v3_7M/11M_*`, `mae_pde_11M.pt`; from g1: `*_fixed512.pt`,
  `v3_vicreg_asym.pt`, `v3_temporal{,_pure,_long}.pt`, `decoder_on_jepa_vit.pt`
- **Old probes/logs**: `results/probes/{g2, smoke_test}`,
  `probe_v3_11M_recon.json`, all of `logs/` (fresh empty `logs/` recreated)

## Verification performed

1. `py_compile` over all kept Python.
2. All 10 checkpoints load through `zoo.load_method` / `FAENP` with
   `strict=True`; finite forward passes (encode + decode).
3. Numeric regression: `zoo.encode(fae_vicreg)` PR_full over the standard
   6000-snapshot set = **19.867**, identical to the pre-refactor
   `diag_richness.json` value.
4. `--help` parses for all five training entry points.
