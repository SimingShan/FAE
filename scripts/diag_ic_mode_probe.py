"""IC-mode amplitude probe: can each encoder linearly recover the Fourier
mode amplitudes of the snapshot?

For each held-out snapshot, take the FFT and extract |u_hat_k| for
k = 1..20 (the IC generator's band). For each method x system x mode,
linear-probe (ridge) the pooled latent on |u_hat_k|. Report per-mode R^2
and the mean across modes.

Encoders that pass across many modes genuinely encode spectral structure,
not just the coefficient.

Outputs (results/probes/g1/):
  diag_ic_modes.json, diag_ic_modes.png, diag_ic_modes_summary.png
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
from src.metrics import lin_probe

device = os.environ.get("DIAG_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = f"{ROOT}/results/checkpoints/g1"
OUT  = f"{ROOT}/results/probes/g1"

X = 1024
K_MAX = 20
N_PER_SYS = 1500
N_SENSORS = 256


def make_coords(N=X):
    return torch.linspace(0, 1 - 1.0/N, N, device=device).unsqueeze(-1)


def fft_mode_amplitudes(U, k_max=K_MAX):
    """U: (N, X) -> (N, k_max+1) magnitudes |u_hat_k| for k = 0..k_max."""
    F = np.fft.rfft(U, axis=-1)
    return np.abs(F[:, :k_max + 1]).astype(np.float32)


def main():
    full_coords = make_coords(X)
    rng = np.random.default_rng(0)
    per_sys = {}
    for n in PDE_NAMES:
        d = load_g1_system(n)
        u_all = d["u"]; N_traj, T, _ = u_all.shape
        traj_idx = rng.permutation(N_traj)[:N_PER_SYS]
        t_idx = rng.integers(0, T, size=N_PER_SYS)
        u = u_all[traj_idx, t_idx]                                 # (N_PER_SYS, X)
        per_sys[n] = {"u": u.astype(np.float32),
                       "amps": fft_mode_amplitudes(u, k_max=K_MAX)}

    U_all = np.concatenate([per_sys[n]["u"] for n in PDE_NAMES])
    sys_idx = np.concatenate([np.full(len(per_sys[n]["u"]), i, dtype=np.int64)
                                  for i, n in enumerate(PDE_NAMES)])

    results = {}
    nv = N_PER_SYS // 5
    for spec in zoo.METHODS:
        m, _ = zoo.load_method(spec.name, CKPT, device)
        if m is None:
            print(f"[{spec.label}] SKIP"); continue
        t0 = time.time()
        print(f"[{spec.label}]", flush=True)
        Z_all = []
        for i0 in range(0, len(U_all), 64):
            u_b = torch.from_numpy(U_all[i0:i0+64]).to(device).float()
            Z_all.append(zoo.encode(m, spec.kind, u_b, full_coords,
                                       n_sensors=N_SENSORS).cpu().numpy())
        Z_all = np.concatenate(Z_all, 0)

        results[spec.label] = {"kind": spec.kind, "latent_dim": int(Z_all.shape[1]),
                                 "per_mode": {}}
        for i, sys_name in enumerate(PDE_NAMES):
            mask = sys_idx == i
            Z = Z_all[mask]; amps = per_sys[sys_name]["amps"]
            n = Z.shape[0]
            ridx = np.random.default_rng(0).permutation(n)
            tr = ridx[:-nv]; va = ridx[-nv:]
            per_mode = []
            for k in range(K_MAX + 1):
                y = amps[:, k]
                if y.std() < 1e-8:
                    per_mode.append(np.nan); continue
                per_mode.append(lin_probe(Z[tr], y[tr], Z[va], y[va]))
            results[spec.label]["per_mode"][sys_name] = per_mode
            arr = np.array(per_mode)
            print(f"  {sys_name:22s}  k=1-5 R²={np.nanmean(arr[1:6]):.3f}  "
                  f"k=6-20 R²={np.nanmean(arr[6:K_MAX+1]):.3f}  "
                  f"all R²={np.nanmean(arr[1:K_MAX+1]):.3f}", flush=True)
        del m; torch.cuda.empty_cache()
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    out_p = f"{OUT}/diag_ic_modes.json"
    json.dump(results, open(out_p, "w"), indent=2)
    print(f"\nsaved {out_p}", flush=True)

    # ---- plot: per-mode R² per system ----
    fig, axes = plt.subplots(1, len(PDE_NAMES), figsize=(4.2 * len(PDE_NAMES), 5),
                                sharey=True)
    for s_idx, sys_name in enumerate(PDE_NAMES):
        ax = axes[s_idx]
        for label, rec in results.items():
            arr = rec["per_mode"][sys_name]
            ax.plot(range(len(arr)), arr, marker="o", linewidth=1.3, alpha=0.85,
                       label=label, markersize=4)
        ax.set_title(sys_name, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
        ax.set_ylim(-0.15, 1.05)
        ax.set_xlabel("Fourier mode k")
        ax.grid(alpha=0.25)
        if s_idx == 0:
            ax.set_ylabel("linear-probe R²  for  |û_k|")
    axes[-1].legend(fontsize=7, loc="upper right", framealpha=0.85)
    fig.suptitle("Spectral richness — does the latent linearly encode INDIVIDUAL Fourier modes?\n"
                  "(probe target = magnitude of mode k of the held-out snapshot)",
                  fontsize=11, y=1.02)
    plt.tight_layout()
    p = f"{OUT}/diag_ic_modes.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)

    # ---- summary bar: mean R² per (method, system) ----
    fig, ax = plt.subplots(figsize=(13, 5))
    labels = list(results.keys())
    n_sys = len(PDE_NAMES)
    width = 0.18
    x = np.arange(len(labels))
    sys_colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    for i, (sys_name, col) in enumerate(zip(PDE_NAMES, sys_colors)):
        vals = [np.nanmean(np.array(results[l]["per_mode"][sys_name])[1:K_MAX+1])
                  for l in labels]
        bars = ax.bar(x + (i - n_sys/2 + 0.5) * width, vals, width,
                          label=sys_name, color=col, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, max(v + 0.02, 0.04),
                      f"{v:.2f}", ha="center", fontsize=7, rotation=90)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mean linear-probe R² across modes k=1..20")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("Average spectral mode recovery — higher = encoder preserves more spectral structure",
                    fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p = f"{OUT}/diag_ic_modes_summary.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
