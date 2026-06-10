"""Sparse-sensor reconstruction quality at varying sensor counts.

Evaluates how well an encoder-decoder pair reconstructs a full field from
N sparse sensors, sweeping N.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Callable, Sequence


@torch.no_grad()
def sparse_recon_rel_l2(
    model_forward: Callable,           # callable: (u_field_at_sensors, sensor_coords, query_coords) → pred at queries
    fields: np.ndarray,                # (N_eval, *field_shape) — full ground-truth fields
    full_coords: torch.Tensor,         # (n_pix, D) — all coords on the canonical grid (D=1 or 2)
    n_sensors_list: Sequence[int] = (8, 16, 32, 64, 128, 256, 512, 1024),
    device: str = "cuda:0",
    seed: int = 0,
):
    """For each N in `n_sensors_list`, draw a random sensor subset, reconstruct,
    and compute per-trajectory rel-L2 = ||recon - gt|| / ||gt||.

    Returns: dict[N] = {"mean": float, "std": float, "per_traj": (N_eval,)}
    """
    n_pix = full_coords.shape[0]
    results = {}
    for n_s in n_sensors_list:
        if n_s > n_pix:
            continue
        g = torch.Generator(); g.manual_seed(seed)
        idx = torch.randperm(n_pix, generator=g)[:n_s].sort().values.to(device)
        coords_in = full_coords[idx]
        errs = []
        for i in range(fields.shape[0]):
            gt = torch.from_numpy(fields[i].reshape(-1).astype(np.float32)).to(device)
            u_in = gt[idx].unsqueeze(0).unsqueeze(-1)         # (1, n_s, 1)
            pred = model_forward(u_in, coords_in.unsqueeze(0),
                                   full_coords.unsqueeze(0))
            pred_v = pred.reshape(-1).cpu().numpy()
            gt_v = gt.cpu().numpy()
            rel = float(np.linalg.norm(pred_v - gt_v) / max(np.linalg.norm(gt_v), 1e-12))
            errs.append(rel)
        errs = np.array(errs)
        results[int(n_s)] = {"mean": float(errs.mean()),
                              "std":  float(errs.std()),
                              "per_traj": errs}
    return results
