# Baselines for the FAE comparison

Clean, faithful ports of the canonical self-supervised methods, adapted to our
4-channel 2D physics fields (128², in_chans=4) at ~7M-param parity with FAE.
Each is the *original algorithm*, not a re-implementation.

| baseline | paradigm | source | location |
|---|---|---|---|
| **MAE** | masked reconstruction | facebookresearch/mae (Kaiming He et al.) | `benchmarks/mae/mae.py` |
| **AE** | reconstruction (full) | MAE backbone, `mask_ratio=0` | `benchmarks/mae/mae.py` (`ae_physics`) |
| **JEPA** (spatio-temporal) | latent prediction (3D conv) | helenqu/physical-representation-learning | `external/physical-representation-learning/` (see `docs/benchmarks/`) |
| **I-JEPA** (single-frame) | latent prediction (2D ViT) | original I-JEPA recipe, our 2D port | `benchmarks/jepa/ijepa2d.py` |
| FAE+VICReg | recon + invariance | ours | `src/models/fae.py` |

Single-frame experiment uses MAE / AE / **I-JEPA** (2D) / FAE; spatio-temporal
uses the helenqu 3D-conv JEPA (and FAE `--temporal`, coord_dim=3, via
`ShearFlowWindowDataset`). Note the two JEPAs differ by design: helenqu does
genuine 3D convolution (time downsampled with Conv3d, frames ∈ {4,16}); the
single-frame I-JEPA is the original image recipe.

## MAE / AE

`benchmarks/mae/mae.py` is a faithful port of MAE's `models_mae.py` +
`util/pos_embed.py` (original cloned to `external/mae`). The masking, encoder,
decoder, and patch-MSE loss are unchanged. Three adaptations, all documented in
the module header:
1. `in_chans` generalized (original hard-codes 3) → 4-channel fields.
2. timm 1.0.25 `Block` no longer takes `qk_scale` → dropped.
3. AE mode: `mask_ratio=0` → no masking, loss over all patches (the original
   divides by `mask.sum()`, which is 0 with nothing masked). So AE-vs-MAE
   isolates the masking signal, sharing one backbone.

```python
from benchmarks.mae.mae import mae_physics, ae_physics
mae = mae_physics()                 # 6.61M params
loss, pred, mask = mae(x, mask_ratio=0.75)   # MAE pretraining step
ae  = ae_physics()
loss, pred, mask = ae(x, mask_ratio=0.0)     # AE pretraining step
z = mae.encode(x)                   # frozen representation for the probe (mean patch token)
```

Verify with `python benchmarks/smoke_test.py`.

## JEPA / I-JEPA

- **Spatio-temporal JEPA**: helenqu's 3D-conv model, cloned to `external/`, wired
  via `docs/benchmarks/`. Genuine spatio-temporal — `nn.Conv3d` throughout,
  downsampling time→1 (num_frames ∈ {4,16}, hardcoded schedule).
- **Single-frame I-JEPA** (`benchmarks/jepa/ijepa2d.py`): the original image
  recipe — ViT over 2D patches, EMA target encoder sees the full image, context
  encoder + predictor fill target patches, smooth-L1 on LayerNorm'd target
  features, no pixel recon. Same recipe as our faithful 1D port
  (`src/models/jepa_vit.py`), extended to 2D. `encoder=5.0M` (parity).

```python
from benchmarks.jepa.ijepa2d import ijepa2d_physics, sample_masks
m = ijepa2d_physics()
ctx, tgt = sample_masks(B, m.num_patches, n_ctx=40, n_tgt=12, device)
pred, target = m(imgs, ctx, tgt)         # smooth_l1_loss(pred, target)
m.update_target(tau)                     # per-iter EMA
z = m.encode(imgs)                       # frozen probe representation
```
