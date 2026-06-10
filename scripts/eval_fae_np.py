"""Evaluate FAE-NP (functional Neural Process) against FAE+VICReg on G1.

Per system, per held-out snapshot:
  - encode μ_C, logvar_C at N=256 sensors
  - encode μ_CT, logvar_CT at full N=1024
  - linear + MLP probe of coefficient using μ_C
  - reconstruction rel-L2 at full N=1024
  - report σ_C statistics (mean, std, fraction-active, posterior contraction)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models import FAE as V3
from src.models.fae_np import FAENP as V4
from src.data.g1 import load_g1_system, PDE_NAMES, make_coords_1d

device = os.environ.get("EVAL_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = f"{ROOT}/results/checkpoints/g1"
OUT  = f"{ROOT}/results/probes/g1"
X = 1024


def lin_probe(Ztr, ytr, Zva, yva):
    sc = StandardScaler(); Ztr = sc.fit_transform(Ztr); Zva = sc.transform(Zva)
    r = Ridge(alpha=1.0).fit(Ztr, ytr); pred = r.predict(Zva)
    return float(1 - ((pred - yva)**2).sum() / max(((yva - yva.mean())**2).sum(), 1e-8))


class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def mlp_probe(Ztr, ytr, Zva, yva, epochs=200, lr=1e-3, batch=512, patience=20):
    sc = StandardScaler(); Ztr = sc.fit_transform(Ztr); Zva = sc.transform(Zva)
    ym = ytr.mean(); ys = ytr.std() + 1e-8
    Ztr_t = torch.from_numpy(Ztr).float().to(device)
    Zva_t = torch.from_numpy(Zva).float().to(device)
    ytr_t = torch.from_numpy((ytr - ym) / ys).float().to(device)
    model = MLPProbe(Ztr_t.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = -1e9; pat = patience
    for ep in range(epochs):
        perm = torch.randperm(Ztr_t.shape[0], device=device)
        model.train()
        for i0 in range(0, Ztr_t.shape[0], batch):
            ix = perm[i0:i0+batch]
            loss = F.mse_loss(model(Ztr_t[ix]), ytr_t[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Zva_t).cpu().numpy() * ys + ym
        r2 = 1 - ((pred - yva)**2).sum() / max(((yva - yva.mean())**2).sum(), 1e-8)
        if r2 > best: best = r2; pat = patience
        else:
            pat -= 1
            if pat <= 0: break
    return float(best)


@torch.no_grad()
def v4_encode(model, u, full_coords, n_sensors):
    """u: (B, X) → (mu_C, logvar_C) each (B, d_latent)."""
    B = u.size(0)
    idx = torch.arange(0, X, X // n_sensors, device=device)[:n_sensors]
    coords_in = full_coords[idx]
    return model.encode_distribution(u[:, idx].unsqueeze(-1), coords_in)


@torch.no_grad()
def v4_recon_full(model, u, full_coords, n_sensors=1024):
    """Encode at n_sensors, decode at full grid.  Returns (B, X) μ_y."""
    mu, logvar = v4_encode(model, u, full_coords, n_sensors)
    z = mu                                          # deterministic: use mean
    mu_y, _ = model.decode(z, full_coords)
    return mu_y


@torch.no_grad()
def v3_encode_pool(model, u, full_coords, n_sensors):
    B = u.size(0)
    idx = torch.arange(0, X, X // n_sensors, device=device)[:n_sensors]
    coords_in = full_coords[idx]
    tok = model.encoder(u[:, idx].unsqueeze(-1), coords_in)
    return tok.mean(dim=1)


@torch.no_grad()
def v3_recon_full(model, u, full_coords, n_sensors=1024):
    B = u.size(0)
    idx = torch.arange(0, X, X // n_sensors, device=device)[:n_sensors]
    coords_in = full_coords[idx]
    tokens = model.encoder(u[:, idx].unsqueeze(-1), coords_in)
    return model.decoder(tokens, full_coords).squeeze(-1)


def rel_l2(pred, gt):
    num = ((pred - gt) ** 2).sum(-1).sqrt()
    den = (gt ** 2).sum(-1).sqrt().clamp_min(1e-8)
    return (num / den).cpu().numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--np_ckpt", default="fae_np.pt",
                     help="filename of V4 checkpoint under results/checkpoints/g1/")
    args, _ = ap.parse_known_args()
    full_coords = make_coords_1d(device, N=X)
    rng = np.random.default_rng(0)
    per_sys = {}
    for n in PDE_NAMES:
        d = load_g1_system(n)
        u = d["u"]; c = d["coeff"]
        perm = rng.permutation(u.shape[0])
        u = u[perm]; c = c[perm]
        nv = u.shape[0] // 5
        T = u.shape[1]
        t_mid = T // 2
        per_sys[n] = {
            "train_mid": u[:-nv, t_mid].astype(np.float32),
            "val_mid":   u[-nv:, t_mid].astype(np.float32),
            "train_c": c[:-nv].astype(np.float32),
            "val_c":   c[-nv:].astype(np.float32),
        }

    # ---- load V4 ----
    p4 = f"{CKPT}/{args.np_ckpt}"
    if not os.path.exists(p4):
        print(f"No FAE-NP checkpoint at {p4}"); return
    ck4 = torch.load(p4, map_location=device, weights_only=False)
    v4 = V4(**ck4["config"]).to(device).eval()
    v4.load_state_dict(ck4["model"])
    for p in v4.parameters(): p.requires_grad_(False)
    print(f"=== FAE-NP (d_latent={v4.d_latent}, recon_kind={ck4['np_config']['recon_kind']}) ===",
          flush=True)

    # ---- load V3+VICReg for comparison ----
    p3 = f"{CKPT}/fae_vicreg.pt"
    v3 = None
    if os.path.exists(p3):
        ck3 = torch.load(p3, map_location=device, weights_only=False)
        v3 = V3(**ck3["config"]).to(device).eval()
        v3.load_state_dict(ck3["model"])
        for p in v3.parameters(): p.requires_grad_(False)
        print(f"  + FAE+VICReg loaded for reference", flush=True)

    results = {"FAE-NP": {}, "FAE+VICReg": {}}
    N_PROBE = 256

    for sys_name in PDE_NAMES:
        tr_u = per_sys[sys_name]["train_mid"]
        va_u = per_sys[sys_name]["val_mid"]
        y_tr = per_sys[sys_name]["train_c"]
        y_va = per_sys[sys_name]["val_c"]

        # --- V4 encoding ---
        def v4_enc(u_np):
            Z, S = [], []
            for i0 in range(0, u_np.shape[0], 32):
                x = torch.from_numpy(u_np[i0:i0+32]).to(device).float()
                mu, lv = v4_encode(v4, x, full_coords, N_PROBE)
                Z.append(mu.cpu().numpy())
                S.append(lv.cpu().numpy())
            return np.concatenate(Z, 0), np.concatenate(S, 0)
        Z_tr4, S_tr4 = v4_enc(tr_u); Z_va4, S_va4 = v4_enc(va_u)
        r2_lin = lin_probe(Z_tr4, y_tr, Z_va4, y_va)
        r2_mlp = mlp_probe(Z_tr4, y_tr, Z_va4, y_va)

        # --- V4 recon at full N ---
        rels = []
        for i0 in range(0, va_u.shape[0], 32):
            x = torch.from_numpy(va_u[i0:i0+32]).to(device).float()
            pred = v4_recon_full(v4, x, full_coords, n_sensors=1024)
            rels.append(rel_l2(pred, x))
        rels = np.concatenate(rels)
        # filter out tiny-norm GTs (heat decay regime)
        norms = np.linalg.norm(va_u, axis=-1)
        keep = norms > 0.1 * np.median(norms)
        rel_med = float(np.median(rels[keep]))

        # σ stats
        sigma_va = np.exp(0.5 * S_va4)
        sigma_active_frac = float((sigma_va.std(0) > 1e-3).mean())  # dims with variation across samples

        results["FAE-NP"][sys_name] = {
            "lin_single": r2_lin, "mlp_single": r2_mlp,
            "recon_relL2_median": rel_med,
            "sigma_mean": float(sigma_va.mean()),
            "sigma_active_frac": sigma_active_frac,
            "logvar_mean_C": float(S_va4.mean()),
        }
        print(f"  FAE-NP {sys_name:22s}  lin={r2_lin:.3f}  MLP={r2_mlp:.3f}  "
              f"recon_med={rel_med:.3f}  σ_mean={sigma_va.mean():.3f}  "
              f"σ_active={sigma_active_frac:.2f}", flush=True)

        # --- V3+VICReg for comparison ---
        if v3 is not None:
            def v3_enc(u_np):
                Z = []
                for i0 in range(0, u_np.shape[0], 32):
                    x = torch.from_numpy(u_np[i0:i0+32]).to(device).float()
                    Z.append(v3_encode_pool(v3, x, full_coords, N_PROBE).cpu().numpy())
                return np.concatenate(Z, 0)
            Z_tr3 = v3_enc(tr_u); Z_va3 = v3_enc(va_u)
            r2l3 = lin_probe(Z_tr3, y_tr, Z_va3, y_va)
            r2m3 = mlp_probe(Z_tr3, y_tr, Z_va3, y_va)
            # recon
            rels3 = []
            for i0 in range(0, va_u.shape[0], 32):
                x = torch.from_numpy(va_u[i0:i0+32]).to(device).float()
                pred = v3_recon_full(v3, x, full_coords, n_sensors=1024)
                rels3.append(rel_l2(pred, x))
            rels3 = np.concatenate(rels3)
            rel_med3 = float(np.median(rels3[keep]))
            results["FAE+VICReg"][sys_name] = {
                "lin_single": r2l3, "mlp_single": r2m3,
                "recon_relL2_median": rel_med3,
            }
            print(f"  FAE {sys_name:22s}  lin={r2l3:.3f}  MLP={r2m3:.3f}  "
                  f"recon_med={rel_med3:.3f}", flush=True)

    tag = args.np_ckpt.replace(".pt", "")
    out_p = f"{OUT}/{tag}_eval.json"
    json.dump(results, open(out_p, "w"), indent=2)
    print(f"\nsaved {out_p}", flush=True)


if __name__ == "__main__":
    main()
