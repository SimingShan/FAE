"""Latent Operators for the 3-jobs harness (mentor pivot) — a small module that maps latent -> latent
with the encoder/decoder FROZEN (the freeze is the evidence). Two flavors mirror the set-vs-grid contrast,
both exposing forward(z, dt) -> z' so the forecasting/inverse harness is identical across arms:
  SetOperator   — permutation-invariant set-transformer over the FAE's 128 tokens (ours; no positional emb)
  FlatOperator  — MLP regressor over a grid-AE's flat latent (= the L-DeepONet operator; the architecture axis)
"""
import math
import torch
import torch.nn as nn
from src.models.fae import TokenPredictor


class SetOperator(nn.Module):
    """FAE-latent operator: Δt-conditioned set->set over the M latent tokens. Reuses the FAE's
    TokenPredictor block (self-attention, NO positional embedding — the slots are an unordered set)."""
    def __init__(self, dim, depth=4, heads=8, dropout=0.0):
        super().__init__()
        self.net = TokenPredictor(dim, depth=depth, heads=heads, dropout=dropout)

    def forward(self, z, dt):                                  # z (B, M, D), dt (B,)
        return self.net(z, dt)


class FlatOperator(nn.Module):
    """Grid-AE-latent operator (= L-DeepONet's regressor): a residual MLP over the flat latent vector,
    Δt-conditioned via Fourier features. The architecture-axis baseline's time-stepper."""
    def __init__(self, dim, hidden=512, depth=4, dt_freq=16):
        super().__init__()
        self.dt_freq = dt_freq
        self.inp = nn.Linear(dim + 2 * dt_freq, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.GELU(),
                                                   nn.Linear(hidden, hidden)) for _ in range(depth)])
        self.out = nn.Linear(hidden, dim)

    def _dt(self, dt):
        f = torch.arange(1, self.dt_freq + 1, device=dt.device, dtype=dt.dtype)
        a = dt[:, None] * f[None, :] * math.pi
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

    def forward(self, z, dt):                                  # z (B, D) flat, dt (B,)
        h = self.inp(torch.cat([z, self._dt(dt)], dim=-1))
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)
