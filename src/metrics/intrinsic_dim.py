"""Intrinsic-dimension estimators.

Used by scripts/diag_dimension.py to compare the dimension of the data
manifold (known by construction on G1) with the dimension of each method's
latent manifold.

- ``twonn``  Facco et al. 2017 — nonlinear ID from the ratio of the two
             nearest-neighbor distances; MLE form with top-fraction discard.
- ``mle_id`` Levina & Bickel 2005 with the MacKay–Ghahramani correction
             (average inverse of per-point estimates).
- ``participation_ratio``  linear effective dimension (sum λ)² / sum λ².
- ``pca_dim``  number of PCA components to reach an energy fraction.

Nonlinear ID is invariant under any smooth, invertible re-encoding — a
faithful representation must preserve it. The linear measures are inflated
by manifold curvature; a representation that *flattens* the manifold has
linear dim ≈ nonlinear dim.
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors


def twonn(X, discard_frac: float = 0.1, seed: int = 0):
    """Facco et al. TwoNN estimator. X: (N, D). Returns float ID."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = r1 > 0
    mu = r2[ok] / r1[ok]
    mu = np.sort(mu)
    # discard the largest ratios (heavy tail from inhomogeneous density)
    keep = mu[: int(len(mu) * (1 - discard_frac))]
    keep = keep[keep > 1.0]
    if len(keep) == 0:
        return float("nan")
    return float(len(keep) / np.log(keep).sum())


def mle_id(X, k: int = 20):
    """Levina–Bickel MLE with MacKay–Ghahramani averaging. X: (N, D)."""
    X = np.asarray(X, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    dist = dist[:, 1:]                                   # drop self
    ok = dist[:, -1] > 0
    dist = dist[ok]
    with np.errstate(divide="ignore"):
        logr = np.log(dist[:, -1][:, None] / dist[:, :-1])
    inv_d = logr.sum(axis=1) / (k - 1)                   # 1 / d_hat per point
    inv_d = inv_d[np.isfinite(inv_d) & (inv_d > 0)]
    if len(inv_d) == 0:
        return float("nan")
    return float(1.0 / inv_d.mean())


def participation_ratio(X):
    """Linear effective dimension of the covariance. X: (N, D)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    eig = np.linalg.eigvalsh(Xc.T @ Xc / max(len(X) - 1, 1))
    eig = np.clip(eig, 0, None)
    return float(eig.sum() ** 2 / max((eig ** 2).sum(), 1e-30))


def pca_dim(X, energy: float = 0.95):
    """Number of PCA components needed to capture `energy` of the variance."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    eig = np.linalg.eigvalsh(Xc.T @ Xc / max(len(X) - 1, 1))[::-1]
    eig = np.clip(eig, 0, None)
    c = np.cumsum(eig) / max(eig.sum(), 1e-30)
    return int(np.searchsorted(c, energy) + 1)


def all_estimates(X, max_points: int = 2000, seed: int = 0):
    """All four estimators on (at most max_points of) X."""
    X = np.asarray(X)
    if len(X) > max_points:
        idx = np.random.default_rng(seed).choice(len(X), max_points, replace=False)
        X = X[idx]
    return {
        "twonn": twonn(X),
        "mle_k20": mle_id(X, k=20),
        "pr": participation_ratio(X),
        "pca95": pca_dim(X, 0.95),
    }
