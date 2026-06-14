"""Unified generator for the new G1: heat, advection, burgers, reaction-diffusion.

All 4 systems share:
  - 5000 trajectories, Nx = 1024, T = 100 frames
  - Domain x ∈ [0, 1) periodic (consistent units)
  - Continuous, log/uniform coefficient sampling (no discrete PDEBench bins)
  - GPU spectral solvers (FFT)
  - Output: data/1d/{system}/{system}_g1.h5  in the canonical G1 schema

Per-system specifics chosen to give rich, non-degenerate dynamics with the
coefficient visibly encoded in a mid-frame snapshot.
"""
from __future__ import annotations
import os, time, argparse
import numpy as np
import torch
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =========================================================================
# Common: random IC (smooth GRF on the periodic domain)
# =========================================================================
def grf_ic(N, n_traj, device, k_max=48, alpha=1.5, seed=0):
    """Smooth GRF on [0, 1) periodic, mean-zero, unit-amplitude after norm."""
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
    # Normalize per-trajectory to unit max-abs amplitude
    u = u / (u.abs().amax(dim=-1, keepdim=True) + 1e-8)
    return u                                                    # (n_traj, N) in [-1, 1]


# =========================================================================
# Smoothed step IC for Fisher-KPP (classic front IC)
# =========================================================================
def allen_cahn_ic(N, n_traj, device, seed=0):
    """Allen-Cahn IC: smooth random function squashed into [-1, 1] via tanh.
    Yields multiple ±1 domains separated by fronts that coarsen over time."""
    g = torch.Generator(device=device).manual_seed(seed)
    u_hat = torch.zeros(n_traj, N, dtype=torch.complex64, device=device)
    ks = torch.fft.fftfreq(N, d=1.0/N).to(device)
    ks_int = torch.round(ks).long()
    # Random energy at k ∈ [2, 10] — controls how many domains form
    for k in range(2, 11):
        pos = int(torch.where(ks_int == k)[0][0])
        neg = int(torch.where(ks_int == -k)[0][0])
        amp = 1.0 / k
        re = torch.randn(n_traj, generator=g, device=device) * amp
        im = torch.randn(n_traj, generator=g, device=device) * amp
        u_hat[:, pos] = torch.complex(re,  im)
        u_hat[:, neg] = torch.complex(re, -im)
    u = torch.fft.ifft(u_hat, dim=-1).real
    u = u / (u.abs().amax(dim=-1, keepdim=True) + 1e-8) * 3.0    # amplify
    return torch.tanh(u)                                          # → [-1, 1]


# =========================================================================
# 1) Heat: u_t = ν u_xx        (spectral exact)
# =========================================================================
def gen_heat(n_traj, N, T, dt, device, seed=0):
    rng = np.random.default_rng(seed)
    nu = (10.0 ** rng.uniform(-3, -1, n_traj)).astype(np.float32)
    u0 = grf_ic(N, n_traj, device, seed=seed + 1)
    k = torch.fft.fftfreq(N, d=1.0/N).to(device) * 2 * np.pi
    k2 = (k ** 2).unsqueeze(0)
    nu_t = torch.from_numpy(nu).to(device)
    u_hat0 = torch.fft.fft(u0, dim=-1)
    out = torch.empty(n_traj, T, N, dtype=torch.float32, device=device)
    for ti in range(T):
        t = ti * dt
        decay = torch.exp(-nu_t.unsqueeze(-1) * k2 * t)
        u_hat = u_hat0 * decay
        out[:, ti] = torch.fft.ifft(u_hat, dim=-1).real
    return out.cpu().numpy(), nu


# =========================================================================
# 2) Advection: u_t = -β u_x        (spectral exact)
# =========================================================================
def gen_advection(n_traj, N, T, dt, device, seed=0):
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.1, 4.0, n_traj).astype(np.float32)
    u0 = grf_ic(N, n_traj, device, seed=seed + 1)
    k = torch.fft.fftfreq(N, d=1.0/N).to(device) * 2 * np.pi
    beta_t = torch.from_numpy(beta).to(device)
    u_hat0 = torch.fft.fft(u0, dim=-1)
    out = torch.empty(n_traj, T, N, dtype=torch.float32, device=device)
    for ti in range(T):
        t = ti * dt
        # phase shift exp(-i β k t)
        phase = torch.exp(-1j * (beta_t.unsqueeze(-1) * k) * t)
        u_hat = u_hat0 * phase
        out[:, ti] = torch.fft.ifft(u_hat, dim=-1).real
    return out.cpu().numpy(), beta


# =========================================================================
# 3) Burgers: u_t + u u_x = ν u_xx     (pseudo-spectral, 2/3 dealias, IMEX-RK4)
# =========================================================================
def gen_burgers(n_traj, N, T, dt, device, seed=0):
    """Conservation form: u_t + 0.5 (u²)_x = ν u_xx.
    Pseudo-spectral with explicit nonlinear term (RK4) and exact diffusion via
    integrating factor exp(-ν k² dt)."""
    rng = np.random.default_rng(seed)
    nu = (10.0 ** rng.uniform(-4, -2, n_traj)).astype(np.float32)
    u0 = grf_ic(N, n_traj, device, seed=seed + 1) * 0.5         # smaller amp for stability
    k = torch.fft.fftfreq(N, d=1.0/N).to(device) * 2 * np.pi
    ik = 1j * k.unsqueeze(0)                                     # for derivative in spectral
    k2 = (k ** 2).unsqueeze(0)
    nu_t = torch.from_numpy(nu).to(device)
    # 2/3 dealiasing mask
    dealias = (torch.abs(k) <= (N // 3)).float().unsqueeze(0)
    # Inner substeps for CFL-like stability with shocks
    n_inner = 4
    dt_i = dt / n_inner

    u = u0.clone()
    out = torch.empty(n_traj, T, N, dtype=torch.float32, device=device)
    out[:, 0] = u

    def rhs(u_in):
        u_hat = torch.fft.fft(u_in, dim=-1) * dealias
        u2_hat = torch.fft.fft(u_in * u_in, dim=-1) * dealias
        return torch.fft.ifft(-0.5 * ik * u2_hat, dim=-1).real

    for ti in range(1, T):
        decay = torch.exp(-nu_t.unsqueeze(-1) * k2 * dt_i)
        for _ in range(n_inner):
            k1 = rhs(u)
            k2v = rhs(u + 0.5 * dt_i * k1)
            k3 = rhs(u + 0.5 * dt_i * k2v)
            k4 = rhs(u + dt_i * k3)
            u = u + (dt_i / 6.0) * (k1 + 2 * k2v + 2 * k3 + k4)
            u_hat = torch.fft.fft(u, dim=-1) * decay * dealias
            u = torch.fft.ifft(u_hat, dim=-1).real
            u = torch.clamp(u, -5.0, 5.0)                       # safety
        out[:, ti] = u
    return out.cpu().numpy(), nu


# =========================================================================
# 4) Reaction-diffusion (Allen-Cahn): u_t = D u_xx + u - u³
#    Bistable, fronts of width ∝ √D, coarsening dynamics. Snapshot encodes D
#    via front width directly.
# =========================================================================
def gen_rd(n_traj, N, T, dt, device, seed=0):
    rng = np.random.default_rng(seed)
    D = (10.0 ** rng.uniform(-4, -2, n_traj)).astype(np.float32)
    u0 = allen_cahn_ic(N, n_traj, device, seed=seed + 1)
    k = torch.fft.fftfreq(N, d=1.0/N).to(device) * 2 * np.pi
    k2 = (k ** 2).unsqueeze(0)
    D_t = torch.from_numpy(D).to(device)
    u = u0.clone()
    out = torch.empty(n_traj, T, N, dtype=torch.float32, device=device)
    out[:, 0] = u
    # IMEX: diffusion exact via integrating factor, reaction explicit
    n_inner = 4                                                  # extra stability
    dt_i = dt / n_inner
    for ti in range(1, T):
        decay = torch.exp(-D_t.unsqueeze(-1) * k2 * dt_i)
        for _ in range(n_inner):
            u_hat = torch.fft.fft(u, dim=-1) * decay
            u = torch.fft.ifft(u_hat, dim=-1).real
            u = u + dt_i * (u - u ** 3)
            u = torch.clamp(u, -1.5, 1.5)
        out[:, ti] = u
    return out.cpu().numpy(), D


# =========================================================================
# Driver
# =========================================================================
SYSTEMS = {
    "heat": dict(gen=gen_heat, T=100, dt=0.05,
                  coeff_name="nu", pde_class=0),
    "advection": dict(gen=gen_advection, T=100, dt=0.005,
                       coeff_name="beta", pde_class=1),
    "burgers": dict(gen=gen_burgers, T=100, dt=0.005,
                      coeff_name="nu", pde_class=2),
    "reaction_diffusion": dict(gen=gen_rd, T=100, dt=0.5,
                                 coeff_name="D", pde_class=3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+",
                     default=list(SYSTEMS.keys()))
    ap.add_argument("--n_traj", type=int, default=5000)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    for sys_name in args.systems:
        cfg = SYSTEMS[sys_name]
        out_dir = f"{ROOT}/data/1d/{sys_name}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{sys_name}_g1.h5"
        print(f"\n=== {sys_name}  (T={cfg['T']}, dt={cfg['dt']}, "
              f"coeff={cfg['coeff_name']}) ===", flush=True)
        t0 = time.time()
        u_all = np.empty((args.n_traj, cfg["T"], args.N), dtype=np.float32)
        coeff_all = np.empty(args.n_traj, dtype=np.float32)
        for i0 in range(0, args.n_traj, args.batch):
            i1 = min(i0 + args.batch, args.n_traj)
            B = i1 - i0
            u, c = cfg["gen"](B, args.N, cfg["T"], cfg["dt"],
                                device, seed=args.seed + i0)
            u_all[i0:i1] = u; coeff_all[i0:i1] = c
            print(f"  {i1}/{args.n_traj}  ({time.time()-t0:.0f}s)", flush=True)
        # Per-trajectory mid-frame std (sanity)
        mid_std = u_all[:, cfg["T"] // 2].std(axis=-1).mean()
        c_lo, c_hi = float(coeff_all.min()), float(coeff_all.max())
        print(f"  → {sys_name}: u range=[{u_all.min():.3f}, {u_all.max():.3f}]  "
              f"mid-std={mid_std:.4f}  coeff∈[{c_lo:.4g}, {c_hi:.4g}]", flush=True)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("u", data=u_all, compression="gzip", compression_opts=4)
            f.create_dataset("coeff", data=coeff_all)
            f.attrs["pde_class"] = cfg["pde_class"]
            f.attrs["coeff_name"] = cfg["coeff_name"]
            f.attrs["pde"] = sys_name
            f.attrs["dt"] = cfg["dt"]
        print(f"  saved {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
