"""Shared probe helpers used by the evaluation and diagnostic scripts."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def r2_score(pred, y):
    """Coefficient of determination with a guarded denominator."""
    return float(1 - ((pred - y) ** 2).sum() /
                  max(((y - y.mean()) ** 2).sum(), 1e-8))


def lin_probe(Ztr, ytr, Zva, yva, alpha: float = 1.0):
    """Standardize -> ridge -> validation R^2."""
    sc = StandardScaler()
    Ztr = sc.fit_transform(Ztr)
    Zva = sc.transform(Zva)
    r = Ridge(alpha=alpha).fit(Ztr, ytr)
    return r2_score(r.predict(Zva), yva)


def lin_probe_split(Z, y, val_frac: float = 0.2, seed: int = 0, alpha: float = 1.0):
    """lin_probe with an internal shuffled train/val split."""
    n = len(y)
    nv = max(1, int(n * val_frac))
    perm = np.random.default_rng(seed).permutation(n)
    tr, va = perm[:-nv], perm[-nv:]
    return lin_probe(Z[tr], y[tr], Z[va], y[va], alpha=alpha)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_probe(Ztr, ytr, Zva, yva, epochs=200, lr=1e-3, batch=512, patience=20,
                device="cpu"):
    """Small MLP regression probe with early stopping; returns best val R^2."""
    sc = StandardScaler()
    Ztr = sc.fit_transform(Ztr)
    Zva = sc.transform(Zva)
    ym = ytr.mean(); ys = ytr.std() + 1e-8
    Ztr_t = torch.from_numpy(Ztr).float().to(device)
    Zva_t = torch.from_numpy(Zva).float().to(device)
    ytr_t = torch.from_numpy((ytr - ym) / ys).float().to(device)
    model = MLPProbe(Ztr_t.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = -1e9; pat = patience
    for _ in range(epochs):
        perm = torch.randperm(Ztr_t.shape[0], device=device)
        model.train()
        for i0 in range(0, Ztr_t.shape[0], batch):
            ix = perm[i0:i0+batch]
            loss = F.mse_loss(model(Ztr_t[ix]), ytr_t[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Zva_t).cpu().numpy() * ys + ym
        r2 = r2_score(pred, yva)
        if r2 > best:
            best = r2; pat = patience
        else:
            pat -= 1
            if pat <= 0:
                break
    return float(best)


def rel_l2(pred, gt):
    """Per-sample relative L2 over the last dim. Torch in, numpy out."""
    num = ((pred - gt) ** 2).sum(-1).sqrt()
    den = (gt ** 2).sum(-1).sqrt().clamp_min(1e-8)
    return (num / den).cpu().numpy()
