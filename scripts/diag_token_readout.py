"""Token-level readout vs mean-pool — re-evaluating the SAME checkpoints.

The dimension diagnostics located FAE's capacity cap in the readout: the 128
latent tokens carry nonlinear ID ~22 (near the field's), the mean-pooled
vector ~10. This script asks whether redefining the evaluated representation
as the TOKEN SET (flattened 128 x 320, PCA-512 for tractable probes) wins
back capacity without losing what the mean-pool delivered.

Per FAE-family method (fae kind only — the others have no token set):

  (1) PROBES     per system: coefficient ridge R^2 — pooled vs token-PCA512
                 (PCA fit on train latents only).
  (2) DIMENSION  advection k_max calibration sweep: TwoNN of pooled vs
                 token-PCA512 vs true dim 2k.
  (3) INVARIANCE heat + burgers: latent drift at R64 / offgrid_64 (vs R1024
                 reference, normalized by between-field spread) and a frozen
                 ridge head trained at on-grid R256, evaluated at offgrid_64
                 — pooled vs tokens. The risk being tested: mean-pooling may
                 CONTRIBUTE invariance by averaging per-token noise.

Outputs: results/probes/g1/diag_token_readout.{json,png}
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models import zoo
from src.data.g1 import load_g1_system, PDE_NAMES
from src.metrics.intrinsic_dim import twonn
from src.metrics import r2_score
from diag_dimension import gen_advection_snapshots

device = os.environ.get("DIAG_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = f"{ROOT}/results/checkpoints/g1"
OUT  = f"{ROOT}/results/probes/g1"

X = 1024
N_TRAIN, N_VAL = 1500, 400
KMAX_SWEEP = (2, 4, 8, 16, 32)
PCA_DIM = 512
FAE_METHODS = ["fae_recon", "fae_vicreg", "fae_spatiotemporal", "jepa_perceiver"]
INV_SYSTEMS = ("heat", "burgers")


def make_coords():
    return torch.linspace(0, 1 - 1.0/X, X, device=device).unsqueeze(-1)


@torch.no_grad()
def encode_both(model, U, coords_full, idx=None, xs=None):
    """Returns (pooled (B, 320), tokens_flat (B, 128*320)).
    idx: on-grid sensor index set; xs: continuous positions (numpy)."""
    P, T = [], []
    for i0 in range(0, len(U), 64):
        u = torch.from_numpy(U[i0:i0+64]).to(device).float()
        if xs is not None:
            xg = np.concatenate([np.arange(X) / X, [1.0]])
            v = np.stack([np.interp(xs, xg, np.concatenate([row, [row[0]]]))
                            for row in U[i0:i0+64]]).astype(np.float32)
            u_in = torch.from_numpy(v).to(device).unsqueeze(-1)
            c_in = torch.from_numpy(xs).to(device).float().view(1, -1, 1)\
                       .expand(u_in.size(0), -1, -1)
        else:
            use = idx if idx is not None else torch.arange(0, X, 4, device=device)
            u_in = u[:, use].unsqueeze(-1)
            c_in = coords_full[use]
        tok = model.encoder(u_in, c_in)                  # (B, 128, 320)
        P.append(tok.mean(1).cpu().numpy())
        T.append(tok.reshape(tok.size(0), -1).cpu().numpy())
    return np.concatenate(P), np.concatenate(T)


def ridge_r2(Ztr, ytr, Zva, yva):
    sc = StandardScaler().fit(Ztr)
    r = Ridge(alpha=1.0).fit(sc.transform(Ztr), ytr)
    return r2_score(r.predict(sc.transform(Zva)), yva)


def main():
    coords = make_coords()
    rng = np.random.default_rng(0)

    per_sys = {}
    for n in PDE_NAMES:
        d = load_g1_system(n)
        u_all, c_all = d["u"], d["coeff"]
        ti = rng.permutation(u_all.shape[0])
        tt = rng.integers(0, u_all.shape[1], size=len(ti))
        u, c = u_all[ti, tt], c_all[ti]
        per_sys[n] = {"u_tr": u[:N_TRAIN].astype(np.float32),
                       "c_tr": c[:N_TRAIN].astype(np.float32),
                       "u_va": u[N_TRAIN:N_TRAIN+N_VAL].astype(np.float32),
                       "c_va": c[N_TRAIN:N_TRAIN+N_VAL].astype(np.float32)}
        del d, u_all

    results = {}
    for name in FAE_METHODS:
        m, spec = zoo.load_method(name, CKPT, device)
        if m is None:
            print(f"[{spec.label}] SKIP"); continue
        t0 = time.time()
        print(f"\n[{spec.label}]", flush=True)
        rec = {"probes": {}, "dimension": [], "invariance": {}}

        # ---------- (1) probes: pooled vs tokens ----------
        for sys_name in PDE_NAMES:
            ps = per_sys[sys_name]
            P_tr, T_tr = encode_both(m, ps["u_tr"], coords)
            P_va, T_va = encode_both(m, ps["u_va"], coords)
            pca = PCA(n_components=PCA_DIM).fit(T_tr)
            r2_pool = ridge_r2(P_tr, ps["c_tr"], P_va, ps["c_va"])
            r2_tok = ridge_r2(pca.transform(T_tr), ps["c_tr"],
                                pca.transform(T_va), ps["c_va"])
            rec["probes"][sys_name] = {"pooled": r2_pool, "tokens": r2_tok}
            print(f"  probe {sys_name:20s} pooled={r2_pool:+.3f}  tokens={r2_tok:+.3f}",
                  flush=True)

        # ---------- (2) dimension calibration ----------
        for k_max in KMAX_SWEEP:
            U = gen_advection_snapshots(N_TRAIN, k_max, t=50 * 0.005, seed=100 + k_max)
            P, T = encode_both(m, U, coords)
            T512 = PCA(n_components=PCA_DIM).fit_transform(T)
            d_pool, d_tok = twonn(P), twonn(T512)
            rec["dimension"].append({"true_dim": 2 * k_max,
                                       "pooled": d_pool, "tokens": d_tok})
            print(f"  dim k={k_max:2d} (true {2*k_max:3d})  pooled={d_pool:.1f}  "
                  f"tokens={d_tok:.1f}", flush=True)

        # ---------- (3) invariance: drift + frozen head ----------
        for sys_name in INV_SYSTEMS:
            ps = per_sys[sys_name]
            idx256 = torch.arange(0, X, 4, device=device)
            idx64 = torch.arange(0, X, 16, device=device)
            xs64 = np.sort(np.random.default_rng(7).uniform(0, 1, 64))
            # reference (full grid) on val
            P_ref, T_ref = encode_both(m, ps["u_va"], coords,
                                          idx=torch.arange(X, device=device))
            inv = {}
            # train head + PCA at R256
            P_tr, T_tr = encode_both(m, ps["u_tr"], coords, idx=idx256)
            pca = PCA(n_components=PCA_DIM).fit(T_tr)
            scP = StandardScaler().fit(P_tr)
            headP = Ridge(alpha=1.0).fit(scP.transform(P_tr), ps["c_tr"])
            scT = StandardScaler().fit(pca.transform(T_tr))
            headT = Ridge(alpha=1.0).fit(scT.transform(pca.transform(T_tr)), ps["c_tr"])
            for cfg, kw in [("R64", {"idx": idx64}), ("offgrid_64", {"xs": xs64})]:
                P_c, T_c = encode_both(m, ps["u_va"], coords, **kw)
                drift_p = (np.linalg.norm(P_c - P_ref, axis=1).mean()
                            / np.linalg.norm(P_ref[None] - P_ref[:, None], axis=-1).mean())
                drift_t = (np.linalg.norm(T_c - T_ref, axis=1).mean()
                            / np.linalg.norm(T_ref[None] - T_ref[:, None], axis=-1).mean())
                r2_p = r2_score(headP.predict(scP.transform(P_c)), ps["c_va"])
                r2_t = r2_score(headT.predict(scT.transform(pca.transform(T_c))), ps["c_va"])
                inv[cfg] = {"drift_pooled": float(drift_p), "drift_tokens": float(drift_t),
                              "r2_pooled": float(r2_p), "r2_tokens": float(r2_t)}
                print(f"  inv {sys_name:8s} {cfg:11s} drift p/t={drift_p:.3f}/{drift_t:.3f}  "
                      f"frozen-head R² p/t={r2_p:+.2f}/{r2_t:+.2f}", flush=True)
            rec["invariance"][sys_name] = inv

        results[spec.label] = rec
        del m; torch.cuda.empty_cache()
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    json.dump(results, open(f"{OUT}/diag_token_readout.json", "w"), indent=2)
    print(f"\nsaved {OUT}/diag_token_readout.json", flush=True)

    # ---- figure: dimension calibration + probe deltas ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, recd in results.items():
        td = [c["true_dim"] for c in recd["dimension"]]
        axes[0].plot(td, [c["tokens"] for c in recd["dimension"]], marker="o",
                       label=f"{label} tokens", linewidth=1.6)
        axes[0].plot(td, [c["pooled"] for c in recd["dimension"]], marker="o",
                       ls="--", alpha=0.6, label=f"{label} pooled", linewidth=1.2)
    lim = max(td) * 1.2
    axes[0].plot([0, lim], [0, lim], color="gray", ls=":", label="y = x")
    axes[0].set_xlabel("true dim"); axes[0].set_ylabel("TwoNN ID")
    axes[0].set_title("dimension: tokens vs mean-pool")
    axes[0].legend(fontsize=6); axes[0].grid(alpha=0.25)

    labels = list(results.keys())
    w = 0.8 / max(len(labels), 1)
    xpos = np.arange(len(PDE_NAMES))
    for i, label in enumerate(labels):
        deltas = [results[label]["probes"][s]["tokens"]
                   - results[label]["probes"][s]["pooled"] for s in PDE_NAMES]
        axes[1].bar(xpos + i * w - 0.4, deltas, w, label=label)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_xticks(xpos); axes[1].set_xticklabels([s[:6] for s in PDE_NAMES])
    axes[1].set_ylabel("probe R² (tokens − pooled)")
    axes[1].set_title("probe gain from token readout")
    axes[1].legend(fontsize=7); axes[1].grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(f"{OUT}/diag_token_readout.png", dpi=120, bbox_inches="tight")
    print(f"saved {OUT}/diag_token_readout.png", flush=True)


if __name__ == "__main__":
    main()
