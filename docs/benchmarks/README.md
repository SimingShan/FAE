# External 2D benchmark — The Well via the helenqu harness

We evaluate FAE against the JEPA baseline from *Representation Learning for
Spatiotemporal Physical Systems* (helenqu/physical-representation-learning,
arXiv 2603.13227) on **The Well** datasets, using their exact
self-supervised-pretrain → frozen-encoder-probe protocol.

## Datasets (downloaded to `../the_well_data/`, gitignored)

- `turbulent_radiative_layer_2D` — 6.9 GB, full. Parameter: cooling time
  `t_cool` (9-value grid). The PoC sandbox.
- `shear_flow` — pruned PoC subset: 4 Reynolds × 3 Schmidt × 32 ICs
  (~190 GB; full download was ~471 GB — the HF "ensemble size" understates the
  HDF5 by ~4×). Parameters: (Reynolds, Schmidt). We dropped 4 of 7 Schmidt
  values (PoC, not a faithful reproduction — preserves IC + Reynolds variation).

## "Finetune" = frozen-encoder probe

Their `finetune` step does **not** update the encoder: it caches embeddings
from the frozen pretrained encoder, then trains a small regression head
(attentive-pool + MLP) to predict the physical parameter. Mechanically a
linear/shallow probe on frozen features — the same evaluation philosophy as
our G1 coefficient probes.

## Reproducing the harness side (their JEPA baseline)

The helenqu repo lives in `../external/physical-representation-learning/`
(gitignored). Our modifications to wire in `turbulent_radiative_layer_2D` are
captured in `helenqu_trl2d_integration.patch` (applies to `physics_jepa/`)
plus the two config files here. To run:

```bash
cd external/physical-representation-learning
git apply ../../docs/benchmarks/helenqu_trl2d_integration.patch  # if fresh clone
cp ../../docs/benchmarks/turbulent_radiative_layer_2D.yaml configs/dataset/
cp ../../docs/benchmarks/train_trl2d_small.yaml configs/
export THE_WELL_DATA_DIR=$(realpath ../../../the_well_data)
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --standalone \
    -m physics_jepa.train_jepa configs/train_trl2d_small.yaml          # pretrain
CUDA_VISIBLE_DEVICES=1 python -m physics_jepa.finetune \
    configs/train_trl2d_small.yaml --trained_model_path <ckpt>/ConvEncoder_5.pth  # probe
```

Integration specifics: trl_2D is 128×384 (non-square) — we crop the middle
128² (matching how they handle shear_flow/rayleigh_benard); the `tcool` label
normalization (log10, mean −0.4984, std 0.6456) was added to
`finetuner.py`'s `STATS`.

## Our side (FAE)

`scripts/train_fae_trl2d.py` — FAE+VICReg (coord_dim=2, in_chans=4) on 2D
snapshots via `src/data/well2d.py`, then a frozen ridge probe of
log10(t_cool). Same frozen-encoder → probe protocol.

See `RESULTS_TRL2D.md` for numbers.
