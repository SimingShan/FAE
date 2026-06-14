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

| dataset | size | how to get it on the cluster |
|---|---|---|
| G1 1D (`data/1d/*_g1.h5`) | 6.5 GB | **regenerate**: `python data_gen/gen_g1_all.py` (GPU spectral solvers, deterministic) — do NOT rsync |
| The Well `turbulent_radiative_layer_2D` | 7.4 GB | `huggingface_hub.snapshot_download("polymathic-ai/turbulent_radiative_layer_2D", repo_type="dataset", local_dir=$THE_WELL_DATA_DIR/turbulent_radiative_layer_2D, allow_patterns=["data/*"])` |
| The Well `shear_flow` | ~440 GB full / **189 GB pruned** | same snapshot_download for `polymathic-ai/shear_flow`, then prune to the PoC grid (keep `Schmidt_{1e-1,1e0,1e1}`, all 4 Reynolds) — see `scripts/_queue/` history or just delete other Schmidt files |

trl_2D is a known **saturated/trivial** probe (random-init encoder ~0.91) — keep
it only as a sandbox. shear_flow is the **discriminating** benchmark (random
baselines ~0). See `docs/benchmarks/RESULTS_TRL2D.md`.

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

- **Regenerate** (cheap, deterministic): G1 1D data.
- **Re-download**: The Well datasets (HF).
- **Move only if you want to skip retraining**: `results/checkpoints/g1/` (3.5 GB,
  gitignored) — FAE/JEPA checkpoints. Otherwise retrain from scratch.
- **Don't move**: `../WFAE_attic/` (old material), `logs/`, `__pycache__`.

## 6. Reproduce the current results

```bash
python scripts/train_fae_shear.py --epochs 60 --tag v1          # FAE on shear (GPU, ~20 min)
# JEPA on shear (their harness, 10 ep ~ 8h on one GPU at 224²):
cd external/physical-representation-learning
THE_WELL_DATA_DIR=$THE_WELL_DATA_DIR CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m physics_jepa.train_jepa configs/train_shear_auth.yaml
```

Standing state at migration: FAE shear done (PR 6.3, logRe 0.21, logSc 0.41 vs
random ~0); JEPA shear was mid-run (10-epoch config) when we stopped to migrate.
G1 1D results are final and stand.
```
