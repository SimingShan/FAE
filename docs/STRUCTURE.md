# Repo structure (post REPA-pivot reorg, 2026-06-20)

Prepared for inspection before the large experiment sweep. The repo now has **two evaluation
categories** and nothing else in `scripts/`.

## Two evaluation categories

### 1. REPA generation — `scripts/generate.py`
Pixel-space SiT flow-matching on PDE fields (no VAE); REPA aligns the SiT tokens to a frozen encoder.
```
python scripts/generate.py --mode {uncond,param,sparse} --align {none,fae,mae,jepa} --dataset {ns,shear}
```
| `--mode` | conditioning | metric |
|---|---|---|
| `uncond` | none | `spectrum_dist` |
| `param`  | physical parameter as a class → LabelEmbedder + AdaLN + **CFG** (REPA's class mechanism) | `spectrum_dist` |
| `sparse` | **FAE encodes scattered sensors → dense field guess** (ViTs can't); DiT refines it | `recon_relL2` + `spectrum_dist` |

`--align`: `none` = pixel-DiT benchmark; `fae` = ours; `mae`/`jepa` = SSL benchmarks (need `--enc_ckpt`).
The **sparse** mode is the one where the FAE is *necessary*, not merely preferable.

### 2. Linear probe — `scripts/eval_linear_probe.py` (+ `eval_ns_probe.py`)
Frozen encoder → ridge probe of physical parameters, with the trivial-baseline floor (CLAUDE.md rule #1).

## Encoders (targets for both categories)
- `scripts/train_fae.py` — FAE. **Default `--mode twoview` = dual-view temporal** (two sparsity views,
  shared recon targets, + future prediction). Saves by default (`--no-save` to skip).
- `scripts/train_baseline.py` — MAE / JEPA / VICReg, `--dataset {shear,flowbench,ns}`.

## Layout
```
scripts/   generate.py · eval_linear_probe.py · eval_ns_probe.py · train_fae.py · train_baseline.py
           viz_generate.py · download_ns.py
src/       models/fae.py · data/{well2d,ns,flowbench}.py · metrics/probes.py · utils/seed.py
benchmarks/ mae/mae.py · jepa/ijepa2d.py
arxiv/pre_repa_pivot/   ~80 archived files (rollout, flowbench, diagnostics, one-off evals,
                        superseded trainers, gen_dit/gen_dit_param now folded into generate.py)
```

## What was archived (for inspection, in `arxiv/pre_repa_pivot/`)
- **Rollout / spatio-temporal generation** (Lu: not a convincing application): `rollout_*`, `gen_dit_cond`,
  `gen_dit_st`, `viz_dit_{cond,st}`, `src/cond_unet/`.
- **FlowBench side-experiment**: `eval_flowbench_*`, `viz_flowbench_gif`, `viz_fpo_recon`.
- **Diagnostics / one-off evals**: `diag_*`, `eval_ns_{diag,floor,theirs,residual}`, `eval_{attentive,battery,
  recon,residual,tsne,views,readout,lowshot,pod_rank,probe_density,rb_floor,shear_perchannel}*`.
- **Superseded trainers**: `train_fae_{shear,trl2d}`, `train_fjepa`, `train_supervised`, `train_vicreg_fpo`,
  `models/fjepa.py`, `mae/videomae.py`, `jepa/stjepa.py`; all old `run_*.slurm`.
