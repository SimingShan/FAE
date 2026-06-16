# Cluster migration guide

What moves, what regenerates, and how to stand the project back up on a cluster.
Nothing here is required to live at a fixed path anymore — all data roots are
env-driven (`THE_WELL_DATA_DIR`); fallbacks point at the old dev box only.

## 1. Repo

```bash
git clone https://github.com/SimingShan/FAE.git WFAE && cd WFAE
```

NOTE: 2 commits may be unpushed from the dev box (GitHub push there kept
stalling on a dead VS Code auth socket). Verify `git log` matches what you
expect; if commits are missing, push them from the dev box first
(`! git push origin main`).

## 2. Python environment

`requirements_cluster.txt` is a frozen snapshot of the working env. Core:
torch 2.4 (cu124), the_well 1.2, hydra-core, timm, einops, omegaconf, h5py,
scikit-learn, matplotlib. The helenqu harness also needs these (it imports
`the_well` and `hydra`).

```bash
pip install -r requirements_cluster.txt    # or recreate with your cluster's torch build
pip install -e .                           # if you want `import src...` without sys.path hacks
```

## 3. Data — set THE_WELL_DATA_DIR, then fetch

```bash
export THE_WELL_DATA_DIR=/cluster/path/to/the_well_data    # used by src/data/well2d.py AND the harness
```

We focus on **shear_flow, FULL data** — all 28 (Reynolds × Schmidt) combinations
× 40 ICs (4 Re × 7 Sc), matching the source paper's Table 1 setup so our MSE is
same-standard on the data axis. **Do NOT prune** (the earlier 189 GB
Schmidt-subset was a PoC only). Budget ~440 GB of disk for it.

| dataset | size | how to get it on the cluster |
|---|---|---|
| **The Well `shear_flow` (FULL — primary)** | **~440 GB** | `huggingface_hub.snapshot_download("polymathic-ai/shear_flow", repo_type="dataset", local_dir=$THE_WELL_DATA_DIR/shear_flow, allow_patterns=["data/*"])` — **all** files, no pruning. (HF's "114.7 GB ensemble" understates the HDF5 ~4×; real footprint ~440 GB.) |
| The Well `turbulent_radiative_layer_2D` | 7.4 GB | `snapshot_download("polymathic-ai/turbulent_radiative_layer_2D", ...)` — sandbox only |
| G1 1D (`data/1d/*_g1.h5`) | 6.5 GB | archived line; **regenerate** if needed: `python arxiv/data_gen/gen_g1_all.py` |

shear_flow is the **discriminating** benchmark (trivial baselines R² ~0 for
Re/Sc). trl_2D is **saturated/trivial** (random-init encoder ~0.91 on t_cool) —
sandbox only, no method conclusions. The data loaders
(`ShearFlowSnapshotDataset` / `ShearFlowWindowDataset`) glob all files in
`data/{train,valid,test}`, so full data needs **no code change** — just download
everything.

## 4. External harness (helenqu JEPA baseline)

The repo `external/physical-representation-learning/` is gitignored. On the
cluster:

```bash
mkdir -p external && cd external
git clone https://github.com/helenqu/physical-representation-learning
cd physical-representation-learning
git apply ../../docs/benchmarks/helenqu_trl2d_integration.patch      # our data.py/finetuner.py edits
cp ../../docs/benchmarks/*.yaml configs/  ;  cp ../../docs/benchmarks/turbulent_radiative_layer_2D.yaml configs/dataset/
pip install the_well hydra-core timm einops omegaconf
```

Harness gotchas (cost us time, don't rediscover):
- JEPA ConvEncoder supports **num_frames ∈ {4, 16} only** (not 8).
- shear_flow at 224² is memory-heavy: 16-frame OOMs on a 24 GB GPU shared with
  another job; use num_frames=4, batch ≤ 8.
- Their "finetune" = a FROZEN-encoder probe (encoder weights never update).

## 5. What to move vs regenerate

- **Re-download**: shear_flow FULL (~440 GB) — the primary dataset.
- **Don't move**: results/plots/weights (gitignored — retrain on the cluster),
  `../WFAE_attic/`, `logs/`, `__pycache__`, the pruned shear_flow subset.

## 6. The experiment matrix on the cluster (full shear_flow)

Single-frame first, then spatio-temporal. Every run reports linear-probe
R²/MSE + PR + trivial baseline via `scripts/eval_linear_probe.py`.

```bash
export THE_WELL_DATA_DIR=/cluster/path/to/the_well_data
# FAE+VICReg (ours) — snapshot, then spatio-temporal:
python scripts/train_fae_shear.py --epochs 60 --n_seed 32 --tag full
python scripts/train_fae_shear.py --temporal --n_frames 4 --tag full_t
# baselines (train, then eval_linear_probe --method {mae,ae,ijepa}):
#   MAE/AE: benchmarks/mae;  single-frame I-JEPA: benchmarks/jepa;
#   spatio-temporal JEPA: helenqu harness (configs/train_shear_auth.yaml,
#   num_frames 4 or 16 — cluster GPUs allow full 16-frame/224², which our
#   24 GB dev box could not).
python scripts/eval_linear_probe.py --method fae --ckpt results/checkpoints/g1/fae_vicreg_shear_full.pt
python scripts/eval_linear_probe.py --method trivial        # the floor
```

Harness gotchas (don't rediscover): JEPA ConvEncoder needs num_frames ∈ {4,16};
their "finetune" is a frozen-encoder probe; for a same-standard MSE, run their
MLP finetune (not just our linear ridge) and use their label normalization
(shear means [4.85, 2.69], stds [0.61, 3.38], compression [log, None]).

## Standing state at migration

Repo is algorithm + eval only (results/plots/weights gitignored, G1 1D line in
`arxiv/`). On the dev box, FAE shear on the PRUNED subset gave linear-probe
MSE ~0.69 (logRe R² 0.21 / logSc R² 0.41) — ≈ VideoMAE, below the paper's JEPA
(MSE 0.38), and NOT same-standard (pruned data, linear vs MLP probe, single
frame). The cluster job: the full matrix on FULL shear_flow, param-matched
(~ViT-Tiny 5.5M), with PR + trivial guards — that is the first real comparison.

## Running on a NEW cluster (portable — all slurm scripts source scripts/env.sh)
1. `git clone https://github.com/SimingShan/FAE.git WFAE && cd WFAE`
2. Set paths once — edit `scripts/env.sh` OR export in ~/.bashrc:
   `THE_WELL_DATA_DIR` (data root), `WFAE_ENV` (conda prefix or venv), `WFAE_REPO` (this repo).
   Adapt the `module load` line in env.sh to your cluster (`module avail`).
3. Env: `docs/cluster_setup.sh` (conda or venv) — pip: torch the_well timm einops h5py scikit-learn matplotlib huggingface_hub.
4. Data (~440GB, HF, resumable): `sbatch scripts/download_shear_flow.slurm` (export HF_TOKEN first for speed).
   Python reads via THE_WELL_DATA_DIR; no hardcoded paths.
5. Smoke: `python benchmarks/smoke_test.py`
6. Run best recipe: `sbatch scripts/run_fae_predict.slurm twoview 0 run1 "--dt_max 2 --dt_fixed 2 --num_latents 128 --batch 128"`
