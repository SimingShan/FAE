"""Consistency under partial observation — flagship novel metric.

The two-subset latent-agreement test: encode the same field with two
independent sensor subsets and measure whether the encoder produces
similar latents.

If the encoder properly represents the FIELD (not just the sensor pattern),
the two latents should agree, regardless of which sensors were used.

Also includes the near-degenerate-pairs test: fields that look identical
at sensors but differ in the unobserved part. A good encoder should
distinguish them according to true field difference, NOT according to
sensor agreement.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Callable
from scipy.stats import spearmanr


@torch.no_grad()
def two_subset_agreement(
    encode_fn: Callable,           # callable: (u_traj, sensor_idx, seed) → (N, D) np array
    u_traj: np.ndarray,            # (N, T, H, W) trajectories
    n_pix: int,
    n_sensors: int,
    seed_pair: tuple = (0, 1),
    device: str = "cuda:0",
):
    """Encode each trajectory with TWO independent random sensor subsets.

    Returns:
      Z_A, Z_B   : (N, D) encoder outputs from subsets A and B
      agreement  : mean cosine similarity between Z_A_i and Z_B_i (higher = more consistent)
      l2_diff    : mean L2 distance between Z_A_i and Z_B_i (lower = more consistent)
    """
    g1 = torch.Generator(); g1.manual_seed(seed_pair[0])
    g2 = torch.Generator(); g2.manual_seed(seed_pair[1])
    idx_A = torch.randperm(n_pix, generator=g1)[:n_sensors].sort().values
    idx_B = torch.randperm(n_pix, generator=g2)[:n_sensors].sort().values

    Z_A = encode_fn(u_traj, idx_A.numpy(), seed_pair[0])
    Z_B = encode_fn(u_traj, idx_B.numpy(), seed_pair[1])

    # Normalize and compute cos sim
    Za = Z_A / (np.linalg.norm(Z_A, axis=1, keepdims=True) + 1e-12)
    Zb = Z_B / (np.linalg.norm(Z_B, axis=1, keepdims=True) + 1e-12)
    cos = (Za * Zb).sum(axis=1)        # (N,)
    l2  = np.linalg.norm(Z_A - Z_B, axis=1)
    return Z_A, Z_B, float(cos.mean()), float(l2.mean())


def near_degenerate_pair_correlation(
    encode_fn: Callable,
    u_traj: np.ndarray,           # (N, H, W) — snapshots
    field_differences: np.ndarray, # (N_pairs,) ground-truth L2 field differences
    pair_indices: np.ndarray,     # (N_pairs, 2)  indices into u_traj of paired snapshots
    sensor_idx: np.ndarray,       # (M,) sensor positions
):
    """For paired snapshots (u_i, u_j) that agree at sensors but differ in field,
    measure whether the encoder's latent distance correlates with the true
    field distance.

    Spearman rank correlation: high (→1) means encoder DOES represent the field;
    low (~0) means encoder only encodes the sensor agreement.
    """
    Z = encode_fn(u_traj, sensor_idx, seed=0)
    z_diff = np.linalg.norm(Z[pair_indices[:, 0]] - Z[pair_indices[:, 1]], axis=1)
    rho, _ = spearmanr(z_diff, field_differences)
    return float(rho), z_diff


def variance_across_subsets(
    encode_fn: Callable,
    u_traj: np.ndarray,
    n_pix: int,
    n_sensors: int,
    n_subsets: int = 5,
):
    """Across n_subsets random sensor subsets, encode the same trajectories
    and measure encoder-output variance (lower = more consistent).

    Returns per-trajectory mean across-subset variance: (N,) array.
    """
    Zs = []
    for s in range(n_subsets):
        g = torch.Generator(); g.manual_seed(s)
        idx = torch.randperm(n_pix, generator=g)[:n_sensors].sort().values.numpy()
        Zs.append(encode_fn(u_traj, idx, seed=s))
    Zs = np.stack(Zs, axis=0)               # (S, N, D)
    var_per_traj = Zs.var(axis=0).mean(axis=1)   # (N,)
    return var_per_traj
