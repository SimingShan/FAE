# Baselines for the FAE comparison

Clean, faithful ports of the canonical self-supervised methods, adapted to our
4-channel 2D physics fields (128², in_chans=4) at ~7M-param parity with FAE.
Each is the *original algorithm*, not a re-implementation.

| baseline | paradigm | source | location |
|---|---|---|---|
| **MAE** | masked reconstruction | facebookresearch/mae (Kaiming He et al.) | `benchmarks/mae/mae.py` |
| **AE** | reconstruction (full) | MAE backbone, `mask_ratio=0` | `benchmarks/mae/mae.py` (`ae_physics`) |
| **JEPA** | latent prediction | helenqu/physical-representation-learning | `external/physical-representation-learning/` (see `docs/benchmarks/`) |
| FAE+VICReg | recon + invariance | ours | `src/models/fae.py` |

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

## JEPA

Already handled the same way (original GitHub code): cloned to `external/`,
wired into our trl_2D/shear_flow harness via `docs/benchmarks/`. Its ConvEncoder
is spatio-temporal (num_frames ∈ {4,16}); a single-frame 2D I-JEPA still needs
building (1D template at `src/models/jepa_vit.py`).
