"""Cross-config transfer: train a probe head once, evaluate it under shifted
sensor configurations.

Train config A: N=256 sensors, uniform random positions (fresh per snapshot).
Test configs (probe head and encoder frozen):
  - A_train:  same as training                          (in-distribution)
  - N32:      count shift down to 32 sensors
  - N1024:    count shift up to the full grid
  - clust:    256 sensors clustered in [0.3, 0.7]       (layout shift)
  - jitter:   256 sensors with epsilon-perturbed coords (position noise)

Per (method, system, config): linear-probe R^2 on the coefficient;
DeltaR^2(shift) small = invariant.

Sparse-native encoders handle this directly; dense-input encoders
(CNN, MAE, JEPA-ViT) get zero-fill at non-sensor positions (caveated).

Outputs (results/probes/g1/):
  diag_cross_config.json, diag_cross_config_{delta,absolute}.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models import zoo
from src.data.g1 import load_g1_system, PDE_NAMES

device = os.environ.get("DIAG_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = f"{ROOT}/results/checkpoints/g1"
OUT  = f"{ROOT}/results/probes/g1"

X = 1024
N_TRAIN = 1500
N_VAL   = 400

# Methods that natively support sparse-position input (no zero-fill caveat).
SPARSE_NATIVE = {"FAE-recon", "FAE+VICReg", "FAE-T2", "MLPSparseAE", "JEPA-Perceiver"}

CONFIGS = ["A_train", "N32", "N1024", "clust", "jitter"]


def sample_config(config, n_snapshots, rng_seed):
    """Returns per-snapshot sensor index array (n_snapshots, n_in)."""
    rng = np.random.default_rng(rng_seed)
    if config == "A_train":
        return np.stack([rng.choice(X, size=256, replace=False) for _ in range(n_snapshots)])
    if config == "N32":
        return np.stack([rng.choice(X, size=32, replace=False) for _ in range(n_snapshots)])
    if config == "N1024":
        return np.tile(np.arange(X), (n_snapshots, 1))
    if config == "clust":
        lo, hi = int(0.3 * X), int(0.7 * X)
        idx = rng.choice(np.arange(lo, hi), size=256, replace=False)
        return np.tile(np.sort(idx), (n_snapshots, 1))
    if config == "jitter":
        return np.stack([rng.choice(X, size=256, replace=False) for _ in range(n_snapshots)])
    raise ValueError(config)


@torch.no_grad()
def encode_batch_with_idx(model, kind, u_field, idx_b, jitter_eps=0.005, rng=None):
    """u_field: (B, X), idx_b: (B, n_in) numpy int; per-snapshot sensor sets.

    Unlike zoo.encode (shared index set), this supports a different sensor
    set per snapshot plus optional coordinate jitter, so it stays local.
    """
    idx_t = torch.from_numpy(idx_b).long().to(device)              # (B, n_in)
    if kind in ("fae", "mlp"):
        u_in = u_field.gather(1, idx_t).unsqueeze(-1)              # (B, n_in, 1)
        coords_in = (idx_t.float() / X).unsqueeze(-1)              # (B, n_in, 1)
        if rng is not None and jitter_eps > 0:
            jit = torch.from_numpy(
                rng.normal(0, jitter_eps, size=coords_in.shape)).float().to(device)
            coords_in = (coords_in + jit).clamp(0.0, 1.0 - 1e-6)
        if kind == "fae":
            tok = model.encoder(u_in, coords_in)
            return tok.mean(dim=1)
        return model.encoder(u_in, coords_in)
    # dense kinds: zero-fill at non-sensor positions
    u_pad = torch.zeros_like(u_field)
    u_pad.scatter_(1, idx_t, u_field.gather(1, idx_t))
    if kind == "cnn":
        feats = model.encoder_conv(u_pad.unsqueeze(1))
        return model.latent_proj(feats.mean(dim=-1))
    if kind == "mae":
        return model.encode_full(u_pad.unsqueeze(1))
    if kind == "vit":
        return model(u_pad, masks=None).mean(dim=1)
    raise ValueError(kind)


def main():
    rng = np.random.default_rng(0)
    print(f"=== Cross-config transfer  device={device} ===", flush=True)

    per_sys = {}
    for n in PDE_NAMES:
        d = load_g1_system(n)
        u_all = d["u"]; c_all = d["coeff"]
        N_traj, T, _ = u_all.shape
        traj_idx = rng.permutation(N_traj)
        t_idx = rng.integers(0, T, size=len(traj_idx))
        u = u_all[traj_idx, t_idx]
        c = c_all[traj_idx]
        per_sys[n] = {
            "u_tr":  u[:N_TRAIN].astype(np.float32),
            "c_tr":  c[:N_TRAIN].astype(np.float32),
            "u_va":  u[N_TRAIN:N_TRAIN + N_VAL].astype(np.float32),
            "c_va":  c[N_TRAIN:N_TRAIN + N_VAL].astype(np.float32),
        }

    results = {}
    for spec in zoo.METHODS:
        m, _ = zoo.load_method(spec.name, CKPT, device)
        if m is None:
            print(f"[{spec.label}] SKIP"); continue
        t0 = time.time()
        print(f"\n[{spec.label}]  (sparse-native={spec.label in SPARSE_NATIVE})", flush=True)
        rec = {"sparse_native": spec.label in SPARSE_NATIVE}

        for sys_name in PDE_NAMES:
            u_tr = per_sys[sys_name]["u_tr"]
            c_tr = per_sys[sys_name]["c_tr"]
            u_va = per_sys[sys_name]["u_va"]
            c_va = per_sys[sys_name]["c_va"]

            # === TRAIN encoding under config A ===
            idx_tr = sample_config("A_train", N_TRAIN, rng_seed=1)
            Z_tr = []
            for i0 in range(0, N_TRAIN, 64):
                u_b = torch.from_numpy(u_tr[i0:i0+64]).to(device).float()
                z = encode_batch_with_idx(m, spec.kind, u_b, idx_tr[i0:i0+64], rng=None)
                Z_tr.append(z.cpu().numpy())
            Z_tr = np.concatenate(Z_tr, 0)
            sc = StandardScaler().fit(Z_tr)
            head = Ridge(alpha=1.0).fit(sc.transform(Z_tr), c_tr)

            # === TEST encoding under each config ===
            rec[sys_name] = {}
            for cfg in CONFIGS:
                idx_va = sample_config(cfg, N_VAL, rng_seed=cfg.__hash__() & 0xFFFF)
                jit_rng = np.random.default_rng(13) if cfg == "jitter" else None
                Z_va = []
                for i0 in range(0, N_VAL, 64):
                    u_b = torch.from_numpy(u_va[i0:i0+64]).to(device).float()
                    z = encode_batch_with_idx(m, spec.kind, u_b, idx_va[i0:i0+64],
                                                    rng=jit_rng)
                    Z_va.append(z.cpu().numpy())
                Z_va = np.concatenate(Z_va, 0)
                pred = head.predict(sc.transform(Z_va))
                rec[sys_name][cfg] = float(1 - ((pred - c_va) ** 2).sum() /
                              max(((c_va - c_va.mean()) ** 2).sum(), 1e-8))
            print(f"  {sys_name:22s}  " + " ".join(
                f"{c}:{rec[sys_name][c]:+.3f}" for c in CONFIGS), flush=True)
        results[spec.label] = rec
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
        del m; torch.cuda.empty_cache()

    out_p = f"{OUT}/diag_cross_config.json"
    json.dump(results, open(out_p, "w"), indent=2)
    print(f"\nsaved {out_p}", flush=True)

    # ---- plot: ΔR² grouped bar ----
    fig, axes = plt.subplots(1, len(PDE_NAMES), figsize=(4.5 * len(PDE_NAMES), 5),
                                  sharey=True)
    labels = [s.label for s in zoo.METHODS if s.label in results]
    shifts = ["N32", "N1024", "clust", "jitter"]
    shift_colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    width = 0.18
    x = np.arange(len(labels))
    for s_idx, sys_name in enumerate(PDE_NAMES):
        ax = axes[s_idx]
        for j, (sh, col) in enumerate(zip(shifts, shift_colors)):
            vals = [results[lab][sys_name]["A_train"] - results[lab][sys_name][sh]
                     for lab in labels]
            ax.bar(x + (j - len(shifts)/2 + 0.5) * width, vals, width,
                       label=sh, color=col, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(sys_name, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if s_idx == 0: ax.set_ylabel("ΔR²  =  R²(train cfg) − R²(test cfg)")
    axes[-1].legend(fontsize=8, title="test shift", loc="best")
    fig.suptitle("Cross-config transfer — head trained on N=256 uniform; tested under shifts\n"
                  "Lower bar (closer to 0) = more invariant.  Negative = test config is easier than train.",
                  fontsize=11, y=1.02)
    plt.tight_layout()
    p = f"{OUT}/diag_cross_config_delta.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)

    # ---- plot: absolute R² per config ----
    fig, axes = plt.subplots(1, len(PDE_NAMES), figsize=(4.5 * len(PDE_NAMES), 5),
                                  sharey=True)
    for s_idx, sys_name in enumerate(PDE_NAMES):
        ax = axes[s_idx]
        for cfg in CONFIGS:
            vals = [results[lab][sys_name][cfg] for lab in labels]
            ax.plot(range(len(labels)), vals, marker="o", linewidth=1.4, alpha=0.85,
                       label=cfg)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(sys_name, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
        ax.set_ylim(-0.15, 1.0)
        ax.grid(alpha=0.25)
        if s_idx == 0: ax.set_ylabel("coefficient probe R² (test)")
    axes[-1].legend(fontsize=8, loc="lower left", framealpha=0.85)
    fig.suptitle("Cross-config transfer — absolute R² of frozen probe head under each test config",
                  fontsize=11, y=1.02)
    plt.tight_layout()
    p = f"{OUT}/diag_cross_config_absolute.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
