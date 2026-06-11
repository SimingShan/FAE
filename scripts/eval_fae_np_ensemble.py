"""V4 ensemble probe — does the Gaussian posterior buy anything for linear probes?

Four probe variants per system:
  (a) probe on μ_C only                  baseline (deterministic mean)
  (b) probe on one z_C sample            single-sample stochastic
  (c) probe on mean of K z_C samples     test-time MC averaging
  (d) train on stacked K samples,        noise-as-augmentation regularization
      test on μ_C
  (e) train+test on K samples,           full ensemble (Bayesian-ish)
      ensemble = mean of K predictions   at test time

For comparison, V3+VICReg linear probe (no ensemble possible) and V4 MLP probe
on μ are also reported.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys, json, time, argparse
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
N_PROBE = 256
K = 10                              # samples per snapshot


def lin_probe(Ztr, ytr, Zva, yva, alpha=1.0):
    sc = StandardScaler(); Ztr = sc.fit_transform(Ztr); Zva = sc.transform(Zva)
    r = Ridge(alpha=alpha).fit(Ztr, ytr); pred = r.predict(Zva)
    return float(1 - ((pred - yva)**2).sum() / max(((yva - yva.mean())**2).sum(), 1e-8)), r, sc


@torch.no_grad()
def v4_encode(model, u, full_coords):
    B = u.size(0)
    idx = torch.arange(0, X, X // N_PROBE, device=device)[:N_PROBE]
    coords_in = full_coords[idx]
    return model.encode_distribution(u[:, idx].unsqueeze(-1), coords_in)


@torch.no_grad()
def encode_all(model, u_np, full_coords):
    """Return (mu, std) arrays for the full dataset, both (N, d_latent)."""
    Mu, S = [], []
    for i0 in range(0, u_np.shape[0], 32):
        x = torch.from_numpy(u_np[i0:i0+32]).to(device).float()
        mu, lv = v4_encode(model, x, full_coords)
        Mu.append(mu.cpu().numpy()); S.append(np.exp(0.5 * lv.cpu().numpy()))
    return np.concatenate(Mu, 0), np.concatenate(S, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np_ckpt", default="fae_np_b1e-4.pt")
    args = ap.parse_args()

    full_coords = make_coords_1d(device, N=X)
    rng = np.random.default_rng(0)

    per_sys = {}
    for n in PDE_NAMES:
        d = load_g1_system(n)
        u = d["u"]; c = d["coeff"]
        perm = rng.permutation(u.shape[0])
        u = u[perm]; c = c[perm]
        nv = u.shape[0] // 5
        T = u.shape[1]; t_mid = T // 2
        per_sys[n] = {
            "train_mid": u[:-nv, t_mid].astype(np.float32),
            "val_mid":   u[-nv:, t_mid].astype(np.float32),
            "train_c": c[:-nv].astype(np.float32),
            "val_c":   c[-nv:].astype(np.float32),
        }

    p4 = f"{CKPT}/{args.np_ckpt}"
    ck = torch.load(p4, map_location=device, weights_only=False)
    v4 = V4(**ck["config"]).to(device).eval()
    v4.load_state_dict(ck["model"])
    for p in v4.parameters(): p.requires_grad_(False)
    print(f"=== V4 ensemble probe ({args.np_ckpt}, K={K} samples) ===", flush=True)

    p3 = f"{CKPT}/fae_vicreg.pt"
    v3 = None
    if os.path.exists(p3):
        ck3 = torch.load(p3, map_location=device, weights_only=False)
        v3 = V3(**ck3["config"]).to(device).eval()
        v3.load_state_dict(ck3["model"])
        for p in v3.parameters(): p.requires_grad_(False)

    results = {}
    for sys_name in PDE_NAMES:
        tr_u = per_sys[sys_name]["train_mid"]
        va_u = per_sys[sys_name]["val_mid"]
        y_tr = per_sys[sys_name]["train_c"]
        y_va = per_sys[sys_name]["val_c"]

        mu_tr, std_tr = encode_all(v4, tr_u, full_coords)
        mu_va, std_va = encode_all(v4, va_u, full_coords)

        # (a) mu-only
        r2_a, _, _ = lin_probe(mu_tr, y_tr, mu_va, y_va)

        # (b) one sample per snapshot
        np.random.seed(0)
        z1_tr = mu_tr + std_tr * np.random.randn(*mu_tr.shape)
        z1_va = mu_va + std_va * np.random.randn(*mu_va.shape)
        r2_b, _, _ = lin_probe(z1_tr, y_tr, z1_va, y_va)

        # (c) mean of K samples at test time, train on μ
        zK_va = np.mean([mu_va + std_va * np.random.randn(*mu_va.shape)
                           for _ in range(K)], axis=0)       # (N_va, d_latent)
        r2_c, _, _ = lin_probe(mu_tr, y_tr, zK_va, y_va)

        # (d) train on K-stacked samples (noise-augmented), test on μ
        np.random.seed(1)
        Ztr_aug = np.concatenate([mu_tr + std_tr * np.random.randn(*mu_tr.shape)
                                       for _ in range(K)], axis=0)
        ytr_aug = np.tile(y_tr, K)
        r2_d, _, _ = lin_probe(Ztr_aug, ytr_aug, mu_va, y_va)

        # (e) train on K-stacked samples, ensemble at test (K samples averaged
        # in PREDICTION space, not feature space)
        sc = StandardScaler().fit(Ztr_aug)
        r_e = Ridge(alpha=1.0).fit(sc.transform(Ztr_aug), ytr_aug)
        preds_va = np.zeros_like(y_va, dtype=np.float64)
        for _ in range(K):
            zk = mu_va + std_va * np.random.randn(*mu_va.shape)
            preds_va += r_e.predict(sc.transform(zk))
        preds_va /= K
        r2_e = float(1 - ((preds_va - y_va)**2).sum() /
                          max(((y_va - y_va.mean())**2).sum(), 1e-8))

        # (f) V3+VICReg linear probe baseline (no ensemble)
        r2_f = None
        if v3 is not None:
            @torch.no_grad()
            def v3_enc(u_np):
                Z = []
                for i0 in range(0, u_np.shape[0], 32):
                    x = torch.from_numpy(u_np[i0:i0+32]).to(device).float()
                    idx = torch.arange(0, X, X // N_PROBE, device=device)[:N_PROBE]
                    coords_in = full_coords[idx]
                    tok = v3.encoder(x[:, idx].unsqueeze(-1), coords_in)
                    Z.append(tok.mean(dim=1).cpu().numpy())
                return np.concatenate(Z, 0)
            Z3tr = v3_enc(tr_u); Z3va = v3_enc(va_u)
            r2_f, _, _ = lin_probe(Z3tr, y_tr, Z3va, y_va)

        results[sys_name] = {
            "a_mu_only": r2_a,
            "b_one_sample": r2_b,
            "c_mean_K_at_test": r2_c,
            "d_train_aug_test_mu": r2_d,
            "e_train_aug_ensemble_test": r2_e,
            "f_fae_vicreg": r2_f,
        }
        print(f"  {sys_name:22s}  a={r2_a:.3f}  b={r2_b:.3f}  "
              f"c={r2_c:.3f}  d={r2_d:.3f}  e={r2_e:.3f}  | V3={r2_f if r2_f is None else f'{r2_f:.3f}'}",
              flush=True)

    out_p = f"{OUT}/{args.np_ckpt.replace('.pt','')}_ensemble.json"
    json.dump(results, open(out_p, "w"), indent=2)
    print(f"\nsaved {out_p}", flush=True)


if __name__ == "__main__":
    main()
