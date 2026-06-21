"""WFAE models (current: 2D AE/MAE/JEPA/FAE comparison).

- ``FAE``  Function AutoEncoder; ``--method fae_vicreg`` = FAE+VICReg,
  recon-only (sim=std=cov=0) = the AE baseline. coord_dim 2 (snapshot) or 3
  (spatiotemporal); in_chans for multi-channel fields.

MAE / I-JEPA baselines live in ``benchmarks/``. Older 1D-G1 models
(fae_np, jepa_vit, baselines, zoo) are archived under ``arxiv/``.
"""
from .fae import FAE, FAEEncoder, SenseiverDecoder, fourier_features

__all__ = ["FAE", "FAEEncoder", "SenseiverDecoder", "fourier_features"]
