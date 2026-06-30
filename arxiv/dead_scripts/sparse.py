"""Sparse-observation utilities: linear (Delaunay) interpolation of K sensors -> dense grid.
Weights are built once per sensor set (CPU Delaunay) and applied on GPU to all fields. This is the
architecture-axis baseline (grid ViTs cannot ingest scattered sensors; the FAE uses them directly).
"""
import numpy as np
import torch
from scipy.spatial import Delaunay


def linear_weights(idx, H, W, device):
    """Barycentric linear-interp weights for an HxW grid from sensor flat-indices `idx`. Built once per K.
    Returns (vertices (HW,3) int, weights (HW,3), outside_hull (HW,) bool) — apply with `apply_linear`."""
    ys, xs = (idx // W).cpu().numpy(), (idx % W).cpu().numpy()
    pts = np.stack([ys, xs], 1).astype(float)
    gy, gx = np.mgrid[0:H, 0:W]
    grid = np.stack([gy.ravel(), gx.ravel()], 1).astype(float)
    tri = Delaunay(pts)
    s = tri.find_simplex(grid)
    T = tri.transform[s]
    b = np.einsum("nij,nj->ni", T[:, :2], grid - T[:, 2])
    w = np.concatenate([b, 1 - b.sum(1, keepdims=True)], 1)
    v = tri.simplices[s].copy()
    out = s < 0
    v[out] = 0
    w[out] = 0.0
    return (torch.tensor(v, device=device),
            torch.tensor(w, dtype=torch.float32, device=device),
            torch.tensor(out, device=device))


def apply_linear(vals, vw):
    """vals (B,C,K) sensor values -> dense grid (B,C,HW) via precomputed weights; outside-hull -> field mean."""
    v, w, out = vw
    g = (vals[:, :, v] * w).sum(-1)
    return torch.where(out[None, None], vals.mean(-1, keepdim=True), g)
