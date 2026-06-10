"""Linear regression probe (R²) on PDE coefficients.

Given encoder features Z (val/train) and target coefficients y,
fit a ridge regression Z → y and report val R².
"""
from __future__ import annotations
import numpy as np


def pca_reduce(Xt: np.ndarray, Xv: np.ndarray, k: int = 64):
    """PCA-reduce features to k dimensions if D > k."""
    if Xt.shape[1] <= k:
        return Xt, Xv
    mu = Xt.mean(0, keepdims=True)
    _, _, V = np.linalg.svd(Xt - mu, full_matrices=False)
    return (Xt - mu) @ V[:k].T, (Xv - mu) @ V[:k].T


def lin_probe_r2(Xt: np.ndarray, yt: np.ndarray, Xv: np.ndarray, yv: np.ndarray,
                  ridge: float = 1e-4) -> float:
    """Fit ridge regression on Xt → yt, score val R² on Xv → yv.

    Returns: R² (1 - SSE/SST). Higher is better; 1.0 is perfect.
    """
    mu = Xt.mean(0, keepdims=True); sd = Xt.std(0, keepdims=True).clip(1e-6)
    Xt = (Xt - mu) / sd; Xv = (Xv - mu) / sd
    Xt = np.hstack([Xt, np.ones((Xt.shape[0], 1))])
    Xv = np.hstack([Xv, np.ones((Xv.shape[0], 1))])
    y_mu = yt.mean(); y_sd = max(yt.std(), 1e-12)
    yt = (yt - y_mu) / y_sd; yv = (yv - y_mu) / y_sd
    w = np.linalg.solve(Xt.T @ Xt + ridge * np.eye(Xt.shape[1]), Xt.T @ yt)
    yp = Xv @ w
    return float(1 - ((yp - yv) ** 2).sum() / max(((yv - yv.mean()) ** 2).sum(), 1e-12))


def probe_all_coefficients(Z_train: np.ndarray, train_coeffs: dict,
                            Z_val: np.ndarray, val_coeffs: dict,
                            coeffs: list = None) -> dict:
    """Linear probe for each named coefficient + mean.

    Returns: {coeff_name: R²} plus 'mean'.
    """
    if coeffs is None:
        coeffs = ["nu", "ax", "ay", "cx", "cy"]
    Zt_p, Zv_p = pca_reduce(Z_train, Z_val)
    r = {c: lin_probe_r2(Zt_p, train_coeffs[c], Zv_p, val_coeffs[c]) for c in coeffs}
    r["mean"] = float(np.mean(list(r.values())))
    return r
