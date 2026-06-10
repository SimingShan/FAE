"""Representation-richness diagnostics for the G1 benchmark.

Four questions, answered for each trained method (loaded via the model zoo):

(1) PARTICIPATION RATIO of the latent covariance.
    PR = (sum lambda)^2 / sum lambda^2 over covariance eigenvalues.
    Computed (a) over the whole dataset, (b) per-system — (b) catches
    per-system collapse that (a) hides.
    Intrinsic-dim reference: ~25 GRF IC modes + coeff + time ~ 27.

(2) WITHIN-FIELD DISPERSION (sparse-capable encoders only).
    Fix M=15 snapshots; sample 20 random sensor sets per N; report
    within_spread / between_spread. Small = discretization-invariant.

(3) RECONSTRUCTION rel-L2 vs SENSOR COUNT N in {16, 32, 64, 256, 1024}.

(4) RICHER PROBES beyond the coefficient: energy, 4th moment, max amplitude,
    time index. Scalar statistics saturate even for collapsed encoders —
    they are the floor, not evidence of richness.

Outputs (results/probes/g1/):
  diag_richness.json, diag_richness_{pr,dispersion,reconN,probes}.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models import zoo
from src.data.g1 import load_g1_system, PDE_NAMES
from src.metrics import lin_probe_split

device = os.environ.get("DIAG_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = f"{ROOT}/results/checkpoints/g1"
OUT  = f"{ROOT}/results/probes/g1"

X = 1024
N_PER_SYS = 1500            # held-out snapshots per system for PR / probes
N_SENSORS_DEFAULT = 256

# Kinds that can encode an arbitrary sensor subset (mae via zero-fill).
DISPERSION_KINDS = ("fae", "mlp", "mae")


def make_coords(N=X):
    return torch.linspace(0, 1 - 1.0/N, N, device=device).unsqueeze(-1)


def participation_ratio(Z):
    """Z: (N, d). Returns (PR, eigenvalues descending)."""
    Z = Z - Z.mean(0)
    cov = Z.T @ Z / max(Z.shape[0] - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0, None)
    s = eig.sum()
    s2 = (eig ** 2).sum()
    pr = (s ** 2) / max(s2, 1e-30)
    return float(pr), eig[::-1]


@torch.no_grad()
def dispersion_ratio(model, kind, full_coords, U_fixed, N_list, n_samples=20):
    """U_fixed: (M, X) fixed snapshots; vary sensor positions per N.
    Returns dict {N: within_spread / between_spread}."""
    out = {}
    if kind not in DISPERSION_KINDS:
        return out
    for Nn in N_list:
        all_z = []
        for _ in range(n_samples):
            idx = torch.randperm(X, device=device)[:Nn].sort().values
            z = zoo.encode(model, kind, U_fixed, full_coords, idx=idx)
            all_z.append(z)
        Z = torch.stack(all_z, dim=1)                # (M, S, D)
        within = Z.std(dim=1).mean(dim=-1)            # (M,)
        between = Z.mean(dim=1).std(dim=0).mean()     # scalar
        out[Nn] = float((within.mean() / max(between, 1e-8)).item())
    return out


def main():
    full_coords = make_coords(X)
    rng = np.random.default_rng(0)

    print("=== Loading G1 systems ===", flush=True)
    per_sys = {}
    for n in PDE_NAMES:
        d = load_g1_system(n)
        u_all = d["u"]                                # (N_traj, T, X)
        c_all = d["coeff"]
        N_traj, T, _ = u_all.shape
        traj_idx = rng.permutation(N_traj)[:N_PER_SYS]
        t_idx = rng.integers(0, T, size=N_PER_SYS)
        u = u_all[traj_idx, t_idx]                    # (N_PER_SYS, X)
        per_sys[n] = {"u": u.astype(np.float32),
                       "c": c_all[traj_idx].astype(np.float32),
                       "t": t_idx.astype(np.float32),
                       "energy": (u**2).mean(-1).astype(np.float32),
                       "fourth": (u**4).mean(-1).astype(np.float32),
                       "max_amp": np.abs(u).max(-1).astype(np.float32)}
    U_all = np.concatenate([per_sys[n]["u"] for n in PDE_NAMES])
    sys_idx = np.concatenate([np.full(len(per_sys[n]["u"]), i, dtype=np.int64)
                                  for i, n in enumerate(PDE_NAMES)])
    print(f"  total snapshots: {len(U_all)}", flush=True)

    DISPERSION_M = 15
    disp_idxs = rng.permutation(len(U_all))[:DISPERSION_M]
    U_disp = torch.from_numpy(U_all[disp_idxs]).to(device).float()

    results = {}
    for spec in zoo.METHODS:
        m, _ = zoo.load_method(spec.name, CKPT, device)
        if m is None:
            print(f"[{spec.label}] SKIP"); continue
        t0 = time.time()
        print(f"\n[{spec.label}]  ({spec.name}.{spec.branch})", flush=True)
        rec = {"label": spec.label, "kind": spec.kind}

        # === encode all latents ===
        Z_all = []
        for i0 in range(0, len(U_all), 64):
            u_b = torch.from_numpy(U_all[i0:i0+64]).to(device).float()
            z = zoo.encode(m, spec.kind, u_b, full_coords,
                            n_sensors=N_SENSORS_DEFAULT)
            Z_all.append(z.cpu().numpy())
        Z_all = np.concatenate(Z_all, 0)               # (N_all, D)
        rec["latent_dim"] = int(Z_all.shape[1])

        # === (1) Participation ratio ===
        pr_full, eig_full = participation_ratio(Z_all)
        rec["pr_full"] = pr_full
        rec["eigvals_top50"] = eig_full[:50].tolist()
        pr_per_sys = {}
        for i, n in enumerate(PDE_NAMES):
            pr_n, _ = participation_ratio(Z_all[sys_idx == i])
            pr_per_sys[n] = pr_n
        rec["pr_per_system"] = pr_per_sys

        # === (4) Richer probes per system (linear only) ===
        probes = {n: {} for n in PDE_NAMES}
        for i, n in enumerate(PDE_NAMES):
            Z_n = Z_all[sys_idx == i]
            for target_name in ("c", "energy", "fourth", "max_amp", "t"):
                key = {"c": "coeff", "t": "time"}.get(target_name, target_name)
                probes[n][key] = lin_probe_split(Z_n, per_sys[n][target_name])
        rec["lin_probes"] = probes

        # === (2) Within-field dispersion ===
        if spec.kind in DISPERSION_KINDS:
            rec["dispersion"] = dispersion_ratio(
                m, spec.kind, full_coords, U_disp,
                N_list=[16, 32, 64, 256, 1024], n_samples=20)
        else:
            rec["dispersion"] = None

        # === (3) Recon vs N ===
        if spec.has_decoder:
            recN = {}
            n_eval = 200
            eval_idx = rng.permutation(len(U_all))[:n_eval]
            U_e = U_all[eval_idx]
            for n_sensors in [16, 32, 64, 256, 1024]:
                rels = []
                for i0 in range(0, n_eval, 32):
                    u_b = torch.from_numpy(U_e[i0:i0+32]).to(device).float()
                    pred = zoo.decode_sparse(m, spec.kind, u_b, full_coords, n_sensors)
                    if pred is None: continue
                    num = ((pred - u_b)**2).sum(-1).sqrt()
                    den = (u_b**2).sum(-1).sqrt().clamp_min(1e-8)
                    rels.append((num / den).cpu().numpy())
                if rels:
                    recN[n_sensors] = float(np.concatenate(rels).mean())
            rec["recon_relL2_vs_N"] = recN
        else:
            rec["recon_relL2_vs_N"] = None

        results[spec.label] = rec
        elapsed = time.time() - t0
        print(f"  PR_full={pr_full:.2f}  (per-sys: " +
              " ".join(f'{n[:4]}={pr_per_sys[n]:.1f}' for n in PDE_NAMES) + ")", flush=True)
        if rec["dispersion"]:
            print("  dispersion: " +
                  " ".join(f'N={k}:{v:.3f}' for k, v in rec["dispersion"].items()), flush=True)
        if rec["recon_relL2_vs_N"]:
            print("  recon rel-L2: " +
                  " ".join(f'N={k}:{v:.3f}' for k, v in rec["recon_relL2_vs_N"].items()), flush=True)
        print(f"  ({elapsed:.0f}s)", flush=True)
        del m; torch.cuda.empty_cache()

    out_p = f"{OUT}/diag_richness.json"
    json.dump(results, open(out_p, "w"), indent=2)
    print(f"\nsaved {out_p}", flush=True)

    # ---- plot 1: PR + singular spectrum ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = list(results.keys())
    pr_vals = [results[k]["pr_full"] for k in labels]
    bars = axes[0].bar(range(len(labels)), pr_vals, color="#1f77b4", alpha=0.85)
    for b, v in zip(bars, pr_vals):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}",
                          ha="center", fontsize=8)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    axes[0].axhline(2, color="red",  linestyle=":", alpha=0.6, label="trivial (~1-2)")
    axes[0].axhline(27, color="green", linestyle=":", alpha=0.6, label="intrinsic est.\n(~25 IC + 1 + 1)")
    axes[0].set_ylabel("Participation Ratio (full dataset)")
    axes[0].set_title("PR of latent covariance  (closer to intrinsic = richer)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", alpha=0.25)

    for k in labels:
        eig = np.array(results[k]["eigvals_top50"])
        eig = eig / eig.max()
        axes[1].plot(range(1, len(eig) + 1), eig, label=k, linewidth=1.4, alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-6, 2)
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("normalized eigenvalue λ / λ_max")
    axes[1].set_title("Singular spectrum (top 50)", fontsize=10)
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    p = f"{OUT}/diag_richness_pr.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)

    # ---- plot 2: dispersion ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in labels:
        d = results[k].get("dispersion")
        if not d: continue
        ns = sorted(d.keys(), key=int)
        ax.plot([int(n) for n in ns], [d[n] for n in ns], marker="o", label=k, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("N sensors")
    ax.set_ylabel("within-field std / between-field std")
    ax.set_title("Within-field dispersion (lower = more discretization-invariant)\n"
                  "20 random sensor sets per N × 15 fixed snapshots", fontsize=10)
    ax.axhline(1.0, color="red", linestyle=":", alpha=0.6, label="parity")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    p = f"{OUT}/diag_richness_dispersion.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)

    # ---- plot 3: recon vs N ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in labels:
        r = results[k].get("recon_relL2_vs_N")
        if not r: continue
        ns = sorted(r.keys(), key=int)
        ax.plot([int(n) for n in ns], [r[n] for n in ns], marker="o", label=k, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("N sensors")
    ax.set_ylabel("recon rel-L2 (held-out, mean)")
    ax.set_title("Reconstruction quality vs sensor count\n"
                  "Flat curve → manifold saturates or model ignores extra sensors", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    p = f"{OUT}/diag_richness_reconN.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)

    # ---- plot 4: richer probes per system ----
    fig, axes = plt.subplots(1, len(PDE_NAMES), figsize=(4.2 * len(PDE_NAMES), 5), sharey=True)
    target_names = ["coeff", "energy", "fourth", "max_amp", "time"]
    for s_idx, sys_name in enumerate(PDE_NAMES):
        ax = axes[s_idx]
        for k in labels:
            ys = [results[k]["lin_probes"][sys_name][t] for t in target_names]
            ax.plot(target_names, ys, marker="o", label=k, linewidth=1.3, alpha=0.85)
        ax.set_title(sys_name, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
        ax.set_ylim(-0.15, 1.05)
        ax.grid(alpha=0.25)
        if s_idx == 0: ax.set_ylabel("linear-probe R²")
    axes[-1].legend(fontsize=7, loc="lower left", framealpha=0.85)
    fig.suptitle("Richer linear probes — does the latent linearize ONLY the coefficient,\n"
                  "or also energy, higher moments, time index, max amplitude?",
                  fontsize=11, y=1.02)
    plt.tight_layout()
    p = f"{OUT}/diag_richness_probes.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
