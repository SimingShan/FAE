"""Random-encoder baseline.

Generates random features for the same trajectories — the probe applied to
random features establishes a *floor* below which any encoder's probe score
is meaningless (and equally a check for over-permissive probes).
"""
from __future__ import annotations
import numpy as np


def random_features(n_traj: int, dim: int, seed: int = 0) -> np.ndarray:
    """Per-trajectory random features (unit-normal Gaussian)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_traj, dim)).astype(np.float32)


def random_baseline_probe(train_coeffs: dict, val_coeffs: dict,
                            n_train: int, n_val: int, dim: int = 320,
                            seed: int = 0, coeffs: list = None) -> dict:
    """Random-features → linear probe scores. The 'ceiling for fluke probes'."""
    from .linear_probe import probe_all_coefficients
    Zt = random_features(n_train, dim, seed=seed)
    Zv = random_features(n_val,   dim, seed=seed + 1)
    return probe_all_coefficients(Zt, train_coeffs, Zv, val_coeffs, coeffs=coeffs)
