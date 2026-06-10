"""Pair-conditional probe variants for advection β recovery.

Tests three pair features per method:
  1. concat[z_t, z_{t+1}]   — linear probe   (already tested in pair_probe.py)
  2. z_{t+1} - z_t           — linear probe   (latent "velocity")
  3. concat[z_t, z_{t+1}]   — 2-layer MLP probe
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models import FAE as V3
from src.models.baselines import MLPSparseAE, CNN1DAE, MAE1DAE
from src.data.g1 import load_g1_system, PDE_NAMES

device = "cuda:0"
torch.set_num_threads(4)
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "checkpoints", "g1")


def make_coords(N=1024):
    return torch.linspace(0, 1 - 1.0/N, N, device=device).unsqueeze(-1)


def load(name):
    p = f"{CKPT}/{name}.pt"
    if not os.path.exists(p): return None, None
    ck = torch.load(p, map_location=device, weights_only=False)
    if name.startswith("v3"):
        m = V3(**ck["config"]).to(device).eval(); kind = "v3"
    elif name == "mlp":
        m = MLPSparseAE(coord_dim=1, latent_dim=320, enc_emb=640, dec_emb=640).to(device).eval()
        kind = "mlp"
    elif name == "cnn":
        m = CNN1DAE().to(device).eval(); kind = "cnn"
    elif name == "mae":
        m = MAE1DAE().to(device).eval(); kind = "mae"
    m.load_state_dict(ck["model"])
    for p_ in m.parameters(): p_.requires_grad_(False)
    return m, kind


@torch.no_grad()
def encode_batch(model, kind, u_batch, full_coords, N_sensors=256):
    X = full_coords.shape[0]
    if kind in ("v3", "mlp"):
        idx = torch.arange(0, X, X // N_sensors, device=device)[:N_sensors]
        coords_in = full_coords[idx]
    if kind == "v3":
        tok = model.encoder(u_batch[:, idx].unsqueeze(-1), coords_in)
        return tok.mean(dim=1)
    elif kind == "mlp":
        B = u_batch.size(0)
        return model.encoder(u_batch[:, idx].unsqueeze(-1),
                              coords_in.unsqueeze(0).expand(B, -1, -1))
    elif kind == "cnn":
        feats = model.encoder_conv(u_batch.unsqueeze(1))
        return model.latent_proj(feats.mean(dim=-1))
    elif kind == "mae":
        return model.encode_full(u_batch.unsqueeze(1))


def encode_pair_pool(model, kind, u_traj, full_coords, ts, batch=64):
    """For times `ts`, return list of (z_t, z_{t+1}) numpy arrays."""
    K = u_traj.shape[0]
    pair_z = []
    for t in ts:
        u_t = u_traj[:, t]
        u_tp = u_traj[:, t + 1]
        z_t_all = []; z_tp_all = []
        for i0 in range(0, K, batch):
            u_t_b = torch.from_numpy(u_t[i0:i0+batch]).to(device).float()
            u_tp_b = torch.from_numpy(u_tp[i0:i0+batch]).to(device).float()
            z_t = encode_batch(model, kind, u_t_b, full_coords)
            z_tp = encode_batch(model, kind, u_tp_b, full_coords)
            z_t_all.append(z_t.cpu().numpy())
            z_tp_all.append(z_tp.cpu().numpy())
        pair_z.append((np.concatenate(z_t_all), np.concatenate(z_tp_all)))
    return pair_z                                               # list of (z_t, z_tp), shape K×d each


def lin_probe(Ztr, ytr, Zva, yva):
    sc = StandardScaler(); Ztr = sc.fit_transform(Ztr); Zva = sc.transform(Zva)
    r = Ridge(alpha=10.0).fit(Ztr, ytr); pred = r.predict(Zva)
    return float(1 - ((pred - yva)**2).sum() / ((yva - yva.mean())**2).sum())


class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def mlp_probe_R2(Ztr, ytr, Zva, yva, epochs=300, lr=1e-3, batch=512, patience=30):
    sc = StandardScaler(); Ztr = sc.fit_transform(Ztr); Zva = sc.transform(Zva)
    ym = ytr.mean(); ys = ytr.std() + 1e-8
    ytr_n = (ytr - ym) / ys; yva_n = (yva - ym) / ys
    Ztr = torch.from_numpy(Ztr).float().to(device)
    Zva = torch.from_numpy(Zva).float().to(device)
    ytr_t = torch.from_numpy(ytr_n).float().to(device)
    model = MLPProbe(Ztr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = -float("inf"); pat = patience
    for ep in range(epochs):
        perm = torch.randperm(Ztr.shape[0], device=device)
        model.train()
        for i0 in range(0, Ztr.shape[0], batch):
            ix = perm[i0:i0+batch]
            pred = model(Ztr[ix])
            loss = F.mse_loss(pred, ytr_t[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Zva).cpu().numpy() * ys + ym
        r2 = 1 - ((pred - yva)**2).sum() / ((yva - yva.mean())**2).sum()
        if r2 > best:
            best = r2; pat = patience
        else:
            pat -= 1
            if pat <= 0: break
    return float(best)


def main():
    full_coords = make_coords()
    rng = np.random.default_rng(0)
    n_pairs = 10
    splits = {}
    for sys_name in PDE_NAMES:
        d = load_g1_system(sys_name)
        u = d["u"].astype(np.float32); c = d["coeff"].astype(np.float32)
        perm = rng.permutation(u.shape[0])
        u = u[perm]; c = c[perm]
        nv = u.shape[0] // 5
        T = u.shape[1]
        ts = rng.choice(T - 1, n_pairs, replace=False)
        splits[sys_name] = {
            "train_u": u[:-nv], "val_u": u[-nv:],
            "train_c": c[:-nv], "val_c": c[-nv:],
            "ts": ts,
        }

    print(f"{'method':12s} | {'system':22s} | concat·lin | diff·lin | concat·MLP")
    print("-" * 80)
    for tag in ["fae_recon", "fae_vicreg", "mlp", "cnn", "mae"]:
        m, kind = load(tag)
        if m is None: continue
        for sys_name in PDE_NAMES:
            sp = splits[sys_name]
            pairs_tr = encode_pair_pool(m, kind, sp["train_u"], full_coords, sp["ts"])
            pairs_va = encode_pair_pool(m, kind, sp["val_u"], full_coords, sp["ts"])
            # Build features
            Z_concat_tr = np.concatenate([np.concatenate([zt, ztp], axis=-1)
                                              for zt, ztp in pairs_tr], axis=0)
            Z_concat_va = np.concatenate([np.concatenate([zt, ztp], axis=-1)
                                              for zt, ztp in pairs_va], axis=0)
            Z_diff_tr = np.concatenate([ztp - zt for zt, ztp in pairs_tr], axis=0)
            Z_diff_va = np.concatenate([ztp - zt for zt, ztp in pairs_va], axis=0)
            y_tr = np.tile(sp["train_c"], n_pairs)
            y_va = np.tile(sp["val_c"], n_pairs)
            r2_lin_cat  = lin_probe(Z_concat_tr, y_tr, Z_concat_va, y_va)
            r2_lin_diff = lin_probe(Z_diff_tr, y_tr, Z_diff_va, y_va)
            r2_mlp_cat  = mlp_probe_R2(Z_concat_tr, y_tr, Z_concat_va, y_va)
            print(f"{tag:12s} | {sys_name:22s} | {r2_lin_cat:6.3f}     | "
                  f"{r2_lin_diff:6.3f}   | {r2_mlp_cat:6.3f}")
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
