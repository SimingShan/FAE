"""Off-grid sensing and cross-resolution transfer — the 'functional' test.

A function-space representation should depend on the underlying function,
not on how it was sampled. Protocol, per (method, system):

  TRAIN a ridge probe head once, on latents from the reference observation
  config (uniform on-grid, R=256). Freeze head and encoder. Then evaluate
  the SAME head under:

    res_64 / res_128 / res_256 / res_512 / res_1024
        the field observed on a uniform grid of that resolution.
        Sparse methods consume the samples directly; dense/grid methods get
        the observations linearly interpolated to their native 1024 grid
        (standard practice — caveat reported).
    offgrid_256 / offgrid_64 / offgrid_32
        sensors at CONTINUOUS positions x ~ U[0,1), values interpolated from
        the native field. Only coordinate-based encoders consume this
        natively; grid methods get interp-to-native, whose error grows as
        scattered density drops — the regime real sensor networks live in.

  Report:
    (i)  frozen-head coefficient R² per config (transfer quality)
    (ii) latent drift vs the res_1024 reference, normalized by between-field
         spread (representation stability)

Outputs (results/probes/g1/):
  diag_offgrid.json, diag_offgrid_r2.png, diag_offgrid_drift.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
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
N_TRAIN, N_VAL = 1500, 400
RES_LIST = (64, 128, 256, 512, 1024)
TRAIN_CFG = "res_256"
CONFIGS = [f"res_{r}" for r in RES_LIST] + ["offgrid_256", "offgrid_64", "offgrid_32"]

# Methods that natively consume coordinate-value sets (no interp-to-grid).
COORD_NATIVE_KINDS = ("fae", "mlp")


def grid_coords(n):
    return (np.arange(n) / n).astype(np.float64)


def interp_to_grid(vals, xs):
    """Scattered periodic observations (B, n) at positions xs (n,) -> (B, X)."""
    xg = grid_coords(X)
    out = np.empty((vals.shape[0], X), dtype=np.float32)
    xs_ext = np.concatenate([xs, [xs[0] + 1.0]])
    for i in range(vals.shape[0]):
        v_ext = np.concatenate([vals[i], [vals[i][0]]])
        out[i] = np.interp(xg, xs_ext, v_ext)
    return out


def sample_field(U, xs):
    """Sample fields U (B, X) at continuous positions xs (n,) by linear interp."""
    xg_ext = np.concatenate([grid_coords(X), [1.0]])
    out = np.empty((U.shape[0], len(xs)), dtype=np.float32)
    for i in range(U.shape[0]):
        u_ext = np.concatenate([U[i], [U[i][0]]])
        out[i] = np.interp(xs, xg_ext, u_ext)
    return out


def observation(U, config, rng):
    """Returns (vals (B, n), xs (n,)) for a config."""
    if config.startswith("res_"):
        r = int(config.split("_")[1])
        idx = np.arange(0, X, X // r)[:r]
        return U[:, idx], grid_coords(X)[idx]
    if config.startswith("offgrid_"):
        n = int(config.split("_")[1])
        xs = np.sort(rng.uniform(0, 1, n))
        return sample_field(U, xs), xs
    raise ValueError(config)


@torch.no_grad()
def encode_obs(model, kind, vals, xs):
    """Encode observations (vals at positions xs) -> (B, D) latents."""
    Z = []
    for i0 in range(0, len(vals), 64):
        v = vals[i0:i0+64]
        if kind in COORD_NATIVE_KINDS:
            u_t = torch.from_numpy(v).to(device).float().unsqueeze(-1)
            c_t = torch.from_numpy(xs).to(device).float().view(1, -1, 1)\
                       .expand(u_t.size(0), -1, -1)
            if kind == "fae":
                z = model.encoder(u_t, c_t).mean(dim=1)
            else:
                z = model.encoder(u_t, c_t)
        else:
            # grid methods: interpolate observations to the native grid
            u_grid = torch.from_numpy(interp_to_grid(v, xs)).to(device).float()
            if kind == "cnn":
                feats = model.encoder_conv(u_grid.unsqueeze(1))
                z = model.latent_proj(feats.mean(dim=-1))
            elif kind == "mae":
                z = model.encode_full(u_grid.unsqueeze(1))
            elif kind == "vit":
                z = model(u_grid, masks=None).mean(dim=1)
            else:
                raise ValueError(kind)
        Z.append(z.cpu().numpy())
    return np.concatenate(Z, 0)


def main():
    rng = np.random.default_rng(0)
    print(f"=== off-grid / cross-resolution transfer  device={device} ===", flush=True)

    per_sys = {}
    for n in PDE_NAMES:
        d = load_g1_system(n)
        u_all = d["u"]; c_all = d["coeff"]
        N_traj, T, _ = u_all.shape
        ti = rng.permutation(N_traj)
        tt = rng.integers(0, T, size=len(ti))
        u = u_all[ti, tt]; c = c_all[ti]
        per_sys[n] = {"u_tr": u[:N_TRAIN].astype(np.float32),
                       "c_tr": c[:N_TRAIN].astype(np.float32),
                       "u_va": u[N_TRAIN:N_TRAIN+N_VAL].astype(np.float32),
                       "c_va": c[N_TRAIN:N_TRAIN+N_VAL].astype(np.float32)}
        del d, u_all

    results = {}
    for spec in zoo.METHODS:
        m, _ = zoo.load_method(spec.name, CKPT, device)
        if m is None:
            print(f"[{spec.label}] SKIP"); continue
        t0 = time.time()
        native = spec.kind in COORD_NATIVE_KINDS
        print(f"\n[{spec.label}]  (coordinate-native={native})", flush=True)
        rec = {"coordinate_native": native}

        for sys_name in PDE_NAMES:
            ps = per_sys[sys_name]
            # train head on reference config
            v_tr, x_tr = observation(ps["u_tr"], TRAIN_CFG, rng)
            Z_tr = encode_obs(m, spec.kind, v_tr, x_tr)
            sc = StandardScaler().fit(Z_tr)
            head = Ridge(alpha=1.0).fit(sc.transform(Z_tr), ps["c_tr"])

            # reference latents for drift normalization
            v_ref, x_ref = observation(ps["u_va"], "res_1024", rng)
            Z_ref = encode_obs(m, spec.kind, v_ref, x_ref)
            between = float(np.linalg.norm(
                Z_ref[None] - Z_ref[:, None], axis=-1).mean())

            rec[sys_name] = {}
            for cfg in CONFIGS:
                v_va, x_va = observation(ps["u_va"], cfg,
                                            np.random.default_rng(7))
                Z_va = encode_obs(m, spec.kind, v_va, x_va)
                pred = head.predict(sc.transform(Z_va))
                r2 = float(1 - ((pred - ps["c_va"]) ** 2).sum() /
                              max(((ps["c_va"] - ps["c_va"].mean()) ** 2).sum(), 1e-8))
                drift = float(np.linalg.norm(Z_va - Z_ref, axis=1).mean()
                                / max(between, 1e-8))
                rec[sys_name][cfg] = {"r2": r2, "latent_drift": drift}
            print(f"  {sys_name:20s} " + " ".join(
                f"{c.replace('res_','R')}:{rec[sys_name][c]['r2']:+.2f}"
                for c in CONFIGS), flush=True)
            print(f"  {'':20s} drift " + " ".join(
                f"{c.replace('res_','R')}:{rec[sys_name][c]['latent_drift']:.3f}"
                for c in CONFIGS), flush=True)
        results[spec.label] = rec
        del m; torch.cuda.empty_cache()
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    out_p = f"{OUT}/diag_offgrid.json"
    json.dump(results, open(out_p, "w"), indent=2)
    print(f"\nsaved {out_p}", flush=True)

    # ---- figures ----
    labels = list(results.keys())
    for key, fname, ylab, title in [
        ("r2", "diag_offgrid_r2.png", "frozen-head coefficient R²",
         "Cross-resolution + off-grid transfer of a probe head trained at R=256"),
        ("latent_drift", "diag_offgrid_drift.png",
         "latent drift / between-field spread (vs R=1024)",
         "Representation stability under observation shift (lower = functional)")]:
        fig, axes = plt.subplots(1, len(PDE_NAMES),
                                    figsize=(4.5 * len(PDE_NAMES), 5), sharey=True)
        xticks = [c.replace("res_", "R") for c in CONFIGS]
        for s_idx, sys_name in enumerate(PDE_NAMES):
            ax = axes[s_idx]
            for l in labels:
                ys = [results[l][sys_name][c][key] for c in CONFIGS]
                ls = "-" if results[l]["coordinate_native"] else "--"
                ax.plot(range(len(CONFIGS)), ys, marker="o", label=l,
                          linewidth=1.5, alpha=0.85, linestyle=ls)
            ax.set_xticks(range(len(CONFIGS)))
            ax.set_xticklabels(xticks, rotation=30, ha="right", fontsize=8)
            ax.set_title(sys_name, fontsize=10)
            ax.grid(alpha=0.25)
            if key == "r2":
                ax.set_ylim(-0.3, 1.0)
                ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            if s_idx == 0:
                ax.set_ylabel(ylab)
        axes[-1].legend(fontsize=7, loc="best",
                          title="solid = coordinate-native", title_fontsize=7)
        fig.suptitle(title, fontsize=11, y=1.02)
        plt.tight_layout()
        p = f"{OUT}/{fname}"
        fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
