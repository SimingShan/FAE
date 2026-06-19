# NS / SSLForPDEs experiment suite

Self-supervised representations of 2D Navier–Stokes, evaluated **head-to-head against
[SSLForPDEs](https://arxiv.org/abs/2307.05432) (Mialon et al., NeurIPS 2023)** on *their* data and
*their* protocol: (1) **buoyancy linear-probe** and (2) **rollout / time-stepping conditioning** (their Table 2).

- **Ours** — FAE (reconstruction + Δ-future prediction, sparse-sensor coordinate encoder).
- **Theirs** — VICReg + Lie-point-symmetry augmentations + ResNet-18 (`external/SSLForPDEs`).
- **Floor** — trivial baselines (channel mean+std), the only honest yardstick.

## Setup (any cluster)

```bash
git clone git@github.com:SimingShan/FAE.git && cd FAE
conda create -n wfae python=3.11 -y && conda activate wfae
pip install -e .                 # core deps
pip install -e ".[vicreg]"       # + submitit/tensorboard, only to run THEIR baseline
export NS_DATA_ROOT=$HOME/scratch/ns_data    # where the dataset lives

# THEIR VICReg baseline is a third-party repo (gitignored). Only needed for train-vicreg / the theirs probe:
git clone https://github.com/facebookresearch/SSLForPDEs external/SSLForPDEs
```
(The PDE-Arena conditioned UNet used by the rollout is already **vendored** in `src/cond_unet/` — no extra clone needed.)

## Data

```bash
python scripts/run_ns.py download                 # full train/valid/test (~77 GB), verified
python scripts/run_ns.py download --subset 8       # 8 files/split, for a quick smoke
```
Fields `u (smoke) / vx / vy`, 128², 56 frames; buoyancy in the filename (≈U(0.2,0.5)).

> ⚠️ **Data-scale caveat.** This released HF dataset is a **~10× subset** of the paper's full
> pretraining set (26,624 trajectories — see paper App. F.3; we have ~2,500). Our FAE (recon-based)
> trains fine on it, but **their VICReg collapses** here (loss flat, representation ≈ random) — so
> reproducing *their* paper number needs the full dataset. Our wins below are stated on this data/protocol.

## Run everything (seed-controlled)

```bash
# pretraining
python scripts/run_ns.py train-fae    --seed 0            # ours -> results/checkpoints/g1/faep_twoview_fae_ns_s0.pt
python scripts/run_ns.py train-fae    --seed 0 --grad     # + gradient loss variant
python scripts/run_ns.py train-vicreg --seed 0            # theirs (collapses on subset; see caveat)

# eval 1 — buoyancy linear probe (valid->test ridge, standardized; floor + ours + theirs)
python scripts/run_ns.py probe --seed 0

# eval 2 — rollout / time-stepping (one-step val MSE x1e3); run all three rows
python scripts/run_ns.py rollout --cond time     --seed 0
python scripts/run_ns.py rollout --cond buoyancy --seed 0
python scripts/run_ns.py rollout --cond rep      --seed 0
```
On SLURM, wrap any stage with the templates in `slurm/` (`sbatch slurm/ns_rollout.slurm rep`).

## Results (this data + protocol, seed 0)

**Eval 1 — buoyancy linear probe** (valid→test, ridge, standardized R²; trivial floor on the same split):

| representation | R²(buoyancy) | note |
|---|---|---|
| FLOOR (channel mean+std) | −0.42 | trivial features fail valid→test |
| **OURS (FAE)** | **0.67–0.71** | robust across ridge α (RidgeCV 0.71); max\|corr\| 0.59 |
| THEIRS (VICReg, this data) | −0.61 | **untrained — collapsed on the 10× subset, not a fair number** |

**Eval 2 — rollout / time-stepping** (one-step val MSE ×1e3, UNetmod-64 + AdaGN, 20 ep):

| conditioning | one-step MSE | |
|---|---|---|
| time-only | 0.250 | baseline |
| **+ ours (FAE rep)** | **0.243** | recovers ~21% of the gap to the ceiling |
| + true buoyancy | 0.217 | upper bound |

Our representation carries usable buoyancy info that helps the forecaster — the central
"amortized system-identification" result, and it **does not depend on their (collapsed) pretraining**.

## Layout

```
scripts/run_ns.py          unified CLI (download / train-fae / train-vicreg / probe / rollout)
scripts/download_ns.py     dataset download + integrity check
scripts/train_fae_predict.py   our FAE trainer (--dataset ns)
scripts/eval_ns_probe.py   buoyancy probe: ours + floor (valid->test ridge)
scripts/eval_ns_theirs.py  their VICReg backbone, same ridge probe (apples-to-apples)
scripts/rollout_ns.py      time-stepping: time / buoyancy / rep conditioning
src/data/ns.py             NS datasets (probe clips + rollout pairs)
src/cond_unet/             PDE-Arena conditioned UNet (UNetmod-64 + AdaGN), vendored
src/utils/seed.py          central set_seed (call at every entry point)
external/SSLForPDEs/       their method (VICReg+Lie); submitit bypassed
```

## Reproducibility

Every entry point calls `set_seed(args.seed)` (python/numpy/torch + seeded DataLoader workers).
For bitwise determinism use `set_seed(s, deterministic=True)` (forces cuDNN determinism, slower).
Results/weights/data are gitignored — regenerate with the commands above.
