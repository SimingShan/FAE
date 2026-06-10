"""Cross-coefficient probe transfer.

Train a linear probe on coefficient A; freeze it; evaluate on coefficient B.
If transfer is high, A and B are entangled in the latent. If low,
the latent has factorized those axes.

Returns the full pairwise transfer matrix.
"""
from __future__ import annotations
import numpy as np
from .linear_probe import lin_probe_r2, pca_reduce


def cross_probe_matrix(Z_train: np.ndarray, train_coeffs: dict,
                        Z_val: np.ndarray, val_coeffs: dict,
                        coeffs: list = None) -> dict:
    """Compute cross-probe matrix M[i, j] = R² of (probe trained on coeff_i,
    tested on coeff_j).

    Diagonal entries = standard same-coeff R².
    Off-diagonal = transfer (high → entangled).
    """
    if coeffs is None:
        coeffs = ["nu", "ax", "ay", "cx", "cy"]
    Zt, Zv = pca_reduce(Z_train, Z_val)
    out = {}
    for i, ci in enumerate(coeffs):
        out[ci] = {}
        for cj in coeffs:
            r2 = lin_probe_r2(Zt, train_coeffs[ci], Zv, val_coeffs[cj])
            out[ci][cj] = r2
    return out
