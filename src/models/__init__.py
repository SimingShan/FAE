"""WFAE models.

- ``FAE``     deterministic Function AutoEncoder (Perceiver encoder + sparse decoder)
- ``FAENP``   probabilistic variant (functional Neural Process, global Gaussian z)
- baselines:  ``MLPSparseAE``, ``CNN1DAE``, ``MAE1DAE``, ``VisionTransformer1D``
- ``zoo``     checkpoint loading / encoding for all benchmark methods
"""
from .fae import FAE, FAEEncoder, SenseiverDecoder, CViTDecoder, fourier_features
from .fae_np import FAENP, gaussian_kl, het_gaussian_nll
from .baselines import MLPSparseAE, CNN1DAE, MAE1DAE
from .jepa_vit import VisionTransformer1D, VisionTransformerPredictor1D

# Backward-compat aliases (pre-cleanup names).
V3 = FAE
V4 = FAENP

__all__ = [
    "FAE", "FAEEncoder", "SenseiverDecoder", "CViTDecoder", "fourier_features",
    "FAENP", "gaussian_kl", "het_gaussian_nll",
    "MLPSparseAE", "CNN1DAE", "MAE1DAE",
    "VisionTransformer1D", "VisionTransformerPredictor1D",
    "V3", "V4",
]
