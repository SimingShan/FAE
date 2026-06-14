"""Dimension fidelity: true PDE intrinsic dimension vs learned dimension.

G1 is generated, so the true dimensionality of the snapshot manifold is known
by construction: a snapshot is determined by the IC's 2*k_max GRF mode
amplitudes + the coefficient (+ time, fixed when we hold the frame index).

Three parts:

(1) FULL-G1 TABLE. Per system (1500 held-out snapshots, mixed times):
    nonlinear ID (TwoNN, MLE) and linear dim (PR, PCA-95) of the raw fields
    and of every method's latent.
      - A faithful encoder preserves nonlinear ID (collapse below = lost
        information).
      - Field-space linear dim >> nonlinear ID (curvature). A representation
        that FLATTENS the manifold has linear dim ~ nonlinear ID — that is
        what makes linear probes work.

(2) CALIBRATION SWEEP (the y=x plot). Advection snapshots generated on the
    fly with restricted IC bands k_max in {2, 4, 8, 16, 32}: true dim =
    2*k_max exactly (advection preserves the IC manifold; beta adds nothing).
    Learned dim vs true dim, one line per method.

(3) DIMENSION-VS-TIME (heat). True effective dimension decays as diffusion
    kills modes; does each latent's dimension track the data's?

Outputs (results/probes/g1/):
  diag_dimension.json, diag_dimension_table.png,
  diag_dimension_calibration.png, diag_dimension_time.png
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

from src.models import zoo
from src.data.g1 import load_g1_system, PDE_NAMES
from src.metrics.intrinsic_dim import all_estimates

device = os.environ.get("DIAG_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = f"{ROOT}/results/checkpoints/g1"
OUT  = f"{ROOT}/results/probes/g1"

X = 1024
N_SNAP = 1500
N_SENSORS = 256
KMAX_SWEEP = (2, 4, 8, 16, 32)
TIME_SLICES = (5, 20, 50, 80, 95)
TRUE_DOF_FULL = 2 * 48 + 1            # generator: k_max=48 GRF + coefficient


# ----------------------------------------------------------------------
# On-the-fly generation, mirroring data_gen/gen_g1_all.py exactly
# (GRF alpha=1.5; advection dt=0.005, mid-frame t = 50*dt)
# ----------------------------------------------------------------------
def grf_ic(N, n_traj, k_max=48, alpha=1.5, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    u_hat = torch.zeros(n_traj, N, dtype=torch.complex64, device=device)
    ks = torch.fft.fftfreq(N, d=1.0/N).to(device)
    ks_int = torch.round(ks).long()
    for k in range(1, k_max + 1):
        pos = int(torch.where(ks_int == k)[0][0])
        neg = int(torch.where(ks_int == -k)[0][0])
        amp = k ** (-alpha / 2.0)
        re = torch.randn(n_traj, generator=g, device=device) * amp
        im = torch.randn(n_traj, generator=g, device=device) * amp
        u_hat[:, pos] = torch.complex(re,  im)
        u_hat[:, neg] = torch.complex(re, -im)
    u = torch.fft.ifft(u_hat, dim=-1).real
    u = u / u.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return u


def gen_advection_snapshots(n_traj, k_max, t, seed=0):
    """Advection mid-evolution snapshots with a restricted IC band.

    Advection PRESERVES the IC manifold (exact spectral shift), so the true
    snapshot dimension is 2*k_max by construction — beta adds nothing because
    a global shift of a free-phase GRF is another GRF sample (this is the
    beta-unidentifiability result seen from the other side). Heat is the
    wrong testbed for this sweep: diffusion kills the high modes by mid-time
    and the realized dimension saturates near 4 regardless of k_max.
    """
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.1, 4.0, n_traj).astype(np.float32)
    u0 = grf_ic(X, n_traj, k_max=k_max, seed=seed + 1)
    k = torch.fft.fftfreq(X, d=1.0/X).to(device) * 2 * np.pi
    shift = torch.exp(-1j * k.unsqueeze(0)
                        * torch.from_numpy(beta).to(device).unsqueeze(-1) * t)
    u_hat = torch.fft.fft(u0, dim=-1) * shift
    return torch.fft.ifft(u_hat, dim=-1).real.float().cpu().numpy()


def make_coords(N=X):
    return torch.linspace(0, 1 - 1.0/N, N, device=device).unsqueeze(-1)


@torch.no_grad()
def encode_all(model, kind, U, full_coords):
    Z = []
    for i0 in range(0, len(U), 64):
        u_b = torch.from_numpy(U[i0:i0+64]).to(device).float()
        Z.append(zoo.encode(model, kind, u_b, full_coords,
                              n_sensors=N_SENSORS).cpu().numpy())
    return np.concatenate(Z, 0)


def main():
    full_coords = make_coords(X)
    rng = np.random.default_rng(0)
    results = {"true_dof_full": TRUE_DOF_FULL, "fields": {}, "latents": {},
                "calibration": {}, "time_curve": {}}

    # ---------- assemble held-out snapshots per system ----------
    per_sys, per_sys_heat_t = {}, {}
    for n in PDE_NAMES:
        d = load_g1_system(n)
        u_all = d["u"]; N_traj, T, _ = u_all.shape
        ti = rng.permutation(N_traj)[:N_SNAP]
        tt = rng.integers(0, T, size=N_SNAP)
        per_sys[n] = u_all[ti, tt].astype(np.float32)
        if n == "heat":
            for ts in TIME_SLICES:
                per_sys_heat_t[ts] = u_all[ti, ts].astype(np.float32)
        del d, u_all

    # ---------- (1) field-space dimensions ----------
    print("=== field-space dimensions ===", flush=True)
    for n in PDE_NAMES:
        est = all_estimates(per_sys[n])
        results["fields"][n] = est
        print(f"  {n:20s} " + "  ".join(f"{k}={v:.1f}" for k, v in est.items()),
              flush=True)

    # ---------- (1) latent dimensions per method ----------
    cal_data = {}   # method -> list of (true_dim, estimates)
    for spec in zoo.METHODS:
        m, _ = zoo.load_method(spec.name, CKPT, device)
        if m is None:
            print(f"[{spec.label}] SKIP"); continue
        t0 = time.time()
        print(f"\n[{spec.label}]", flush=True)
        results["latents"][spec.label] = {}
        for n in PDE_NAMES:
            Z = encode_all(m, spec.kind, per_sys[n], full_coords)
            est = all_estimates(Z)
            results["latents"][spec.label][n] = est
            print(f"  {n:20s} " + "  ".join(f"{k}={v:.1f}" for k, v in est.items()),
                  flush=True)

        # ---------- (2) calibration sweep on this method ----------
        cal = []
        for k_max in KMAX_SWEEP:
            U = gen_advection_snapshots(N_SNAP, k_max, t=50 * 0.005, seed=100 + k_max)
            Z = encode_all(m, spec.kind, U, full_coords)
            est_z = all_estimates(Z)
            est_u = all_estimates(U)
            cal.append({"k_max": k_max, "true_dim": 2 * k_max,
                          "latent": est_z, "field": est_u})
            print(f"  k_max={k_max:2d} (true {2*k_max:3d})  "
                  f"latent twonn={est_z['twonn']:.1f} mle={est_z['mle_k20']:.1f} "
                  f"pr={est_z['pr']:.1f}  | field twonn={est_u['twonn']:.1f}",
                  flush=True)
        results["calibration"][spec.label] = cal

        # ---------- (3) heat dimension vs time ----------
        tc = []
        for ts in TIME_SLICES:
            Z = encode_all(m, spec.kind, per_sys_heat_t[ts], full_coords)
            tc.append({"t_idx": ts, "latent": all_estimates(Z)})
        results["time_curve"][spec.label] = tc

        del m; torch.cuda.empty_cache()
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # field-space references for calibration + time curve
    results["calibration"]["_field"] = [
        {"k_max": k, "true_dim": 2 * k,
         "field": all_estimates(gen_advection_snapshots(N_SNAP, k, t=50 * 0.005,
                                                            seed=100 + k))}
        for k in KMAX_SWEEP]
    results["time_curve"]["_field"] = [
        {"t_idx": ts, "field": all_estimates(per_sys_heat_t[ts])}
        for ts in TIME_SLICES]

    out_p = f"{OUT}/diag_dimension.json"
    json.dump(results, open(out_p, "w"), indent=2)
    print(f"\nsaved {out_p}", flush=True)

    # =================== figures ===================
    labels = list(results["latents"].keys())

    # ---- table figure: nonlinear ID + linear dim per system ----
    fig, axes = plt.subplots(2, len(PDE_NAMES), figsize=(4.6 * len(PDE_NAMES), 8.5),
                                sharey="row")
    for s_idx, n in enumerate(PDE_NAMES):
        for r, (key, name) in enumerate([("twonn", "nonlinear ID (TwoNN)"),
                                            ("pr", "linear dim (PR)")]):
            ax = axes[r, s_idx]
            vals = [results["latents"][l][n][key] for l in labels]
            ax.bar(range(len(labels)), vals, alpha=0.85, color="#1f77b4")
            ax.axhline(results["fields"][n][key], color="green", ls="--",
                        label=f"field {name.split()[0]}")
            ax.axhline(results["fields"][n]["twonn"], color="red", ls=":",
                        label="field nonlinear ID")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            if s_idx == 0: ax.set_ylabel(name)
            if r == 0: ax.set_title(n, fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            if s_idx == 0 and r == 0: ax.legend(fontsize=7)
    fig.suptitle("Latent dimension vs data-manifold dimension\n"
                  "top: nonlinear ID (must be preserved) — bottom: linear dim "
                  "(should match nonlinear ID = flattened manifold)", fontsize=11, y=1.0)
    plt.tight_layout()
    p = f"{OUT}/diag_dimension_table.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)

    # ---- calibration figure: learned vs true (TwoNN), one line per method ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for est_key, ax in zip(["twonn", "pr"], axes):
        true_dims = [2 * k for k in KMAX_SWEEP]
        for l in labels:
            ys = [c["latent"][est_key] for c in results["calibration"][l]]
            ax.plot(true_dims, ys, marker="o", label=l, linewidth=1.5, alpha=0.85)
        ys_f = [c["field"][est_key] for c in results["calibration"]["_field"]]
        ax.plot(true_dims, ys_f, marker="s", color="black", ls="--",
                 label="raw field", linewidth=1.5)
        lim = max(true_dims) * 1.3
        ax.plot([0, lim], [0, lim], color="gray", ls=":", label="y = x (truth)")
        ax.set_xlabel("true manifold dimension (2·k_max, by construction)")
        ax.set_ylabel(f"estimated dimension ({est_key})")
        ax.set_title({"twonn": "nonlinear ID — information preserved?",
                       "pr": "linear dim — manifold flattened?"}[est_key], fontsize=10)
        ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Dimension calibration: advection with restricted IC bands "
                  f"(k_max ∈ {list(KMAX_SWEEP)}), true dim known by construction",
                  fontsize=11, y=1.02)
    plt.tight_layout()
    p = f"{OUT}/diag_dimension_calibration.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)

    # ---- time-curve figure ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in labels:
        ys = [c["latent"]["twonn"] for c in results["time_curve"][l]]
        ax.plot(TIME_SLICES, ys, marker="o", label=l, linewidth=1.5, alpha=0.85)
    ys_f = [c["field"]["twonn"] for c in results["time_curve"]["_field"]]
    ax.plot(TIME_SLICES, ys_f, marker="s", color="black", ls="--",
             label="raw field", linewidth=1.8)
    ax.set_xlabel("time index (heat: diffusion kills modes → dimension decays)")
    ax.set_ylabel("nonlinear ID (TwoNN)")
    ax.set_title("Does the latent's dimension track the physics over time?", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    plt.tight_layout()
    p = f"{OUT}/diag_dimension_time.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
