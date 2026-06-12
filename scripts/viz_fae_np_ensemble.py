"""V4 ensemble visualization: given sparse context C, show the spread of
predictions from sampling z ~ q(z|C).

For each system, pick a held-out trajectory and visualize:
  - N=32 sensors  (sparse context) → K=30 sampled field predictions
  - N=256 sensors (denser context) → K=30 sampled field predictions

In each panel: GT, sensor crosses, K sample predictions (faint), mean curve,
±1σ band from sampled z, and the decoder's per-location heteroscedastic σ_y
as a separate band.

The contrast worth seeing: how much does q(z|C) spread shrink when N grows
from 32 → 256?  Our diagnostic said σ_active=0 (input-independent), so we
expect approximately no change.  This figure makes that visible.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.fae_np import FAENP as V4
from src.data.g1 import load_g1_system, PDE_NAMES, make_coords_1d

device = os.environ.get("VIZ_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = f"{ROOT}/results/checkpoints/g1"
OUT  = f"{ROOT}/results/probes/g1"
X = 1024
K = 30                                  # ensemble samples per snapshot


def load_v4(name="fae_np_b1e-4.pt"):
    ck = torch.load(f"{CKPT}/{name}", map_location=device, weights_only=False)
    m = V4(**ck["config"]).to(device).eval()
    m.load_state_dict(ck["model"])
    for p in m.parameters(): p.requires_grad_(False)
    return m


@torch.no_grad()
def encode_sparse(model, u_field, full_coords, n_sensors, seed=0):
    """u_field: (1, X). Returns (mu, logvar) of (1, d_latent) and sensor idx."""
    rng = torch.Generator(device=device).manual_seed(seed)
    idx = torch.randperm(X, generator=rng, device=device)[:n_sensors].sort().values
    coords_in = full_coords[idx]
    mu, lv = model.encode_distribution(u_field[:, idx].unsqueeze(-1), coords_in)
    return mu, lv, idx


@torch.no_grad()
def sample_predictions(model, mu, logvar, full_coords, K=K):
    """For K different z samples, decode at full grid.
    Returns (K, X) μ_y and (K, X) σ_y."""
    Mu_y, Sigma_y = [], []
    for k in range(K):
        eps = torch.randn_like(mu)
        z = mu + (0.5 * logvar).exp() * eps
        mu_y, logvar_y = model.decode(z, full_coords)         # (1, X), (1, X)
        Mu_y.append(mu_y.squeeze(0).cpu().numpy())
        Sigma_y.append((0.5 * logvar_y).exp().squeeze(0).cpu().numpy())
    return np.stack(Mu_y), np.stack(Sigma_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np_ckpt", default="fae_np_b1e-4.pt")
    args = ap.parse_args()

    full_coords = make_coords_1d(device, N=X)
    v4 = load_v4(args.np_ckpt)
    print(f"V4 loaded ({args.np_ckpt})", flush=True)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(PDE_NAMES), 2, figsize=(14, 12),
                                  sharex=True)

    for r_i, sys_name in enumerate(PDE_NAMES):
        d = load_g1_system(sys_name)
        u_all = d["u"]; coeff = d["coeff"]
        # mid-coefficient held-out trajectory
        held = rng.permutation(u_all.shape[0])[u_all.shape[0]//5:]    # held-out side
        order = np.argsort(coeff[held])
        traj_idx = int(held[order[len(order)//2]])
        t_idx = u_all.shape[1] // 2
        gt = u_all[traj_idx, t_idx].astype(np.float32)             # (X,)
        u_t = torch.from_numpy(gt[None]).to(device).float()

        for c_i, n_sensors in enumerate([32, 256]):
            ax = axes[r_i, c_i]
            mu, lv, sensor_idx = encode_sparse(v4, u_t, full_coords, n_sensors,
                                                      seed=42)
            Mu_y, Sigma_y = sample_predictions(v4, mu, lv, full_coords, K=K)
            mean_pred = Mu_y.mean(axis=0)
            std_pred  = Mu_y.std(axis=0)             # spread across z samples
            sigma_y_mean = Sigma_y.mean(axis=0)      # heteroscedastic σ_y (averaged over K)
            x = full_coords.squeeze(-1).cpu().numpy()
            si = sensor_idx.cpu().numpy()

            # background: K sample lines (faint)
            for k in range(K):
                ax.plot(x, Mu_y[k], color="#1f77b4", linewidth=0.5, alpha=0.18)

            # mean prediction
            ax.plot(x, mean_pred, color="#1f77b4", linewidth=1.6, label="ensemble mean")

            # ±1σ from sample spread (z-uncertainty)
            ax.fill_between(x, mean_pred - std_pred, mean_pred + std_pred,
                                color="#1f77b4", alpha=0.18, label="±1σ from z-samples")

            # heteroscedastic σ_y band (decoder uncertainty)
            ax.fill_between(x, mean_pred - sigma_y_mean, mean_pred + sigma_y_mean,
                                color="#d62728", alpha=0.10, label="±1σ_y (decoder)")

            # GT and sensors
            ax.plot(x, gt, color="black", linewidth=1.0, alpha=0.85, label="GT")
            ax.scatter(x[si], gt[si], c="lime", s=15, marker="x", zorder=4,
                          label=f"N={n_sensors} sensors")

            # diagnostics
            std_global  = float(std_pred.mean())
            sigma_global = float(sigma_y_mean.mean())
            rel_l2 = np.linalg.norm(mean_pred - gt) / max(np.linalg.norm(gt), 1e-8)
            ax.set_title(f"{sys_name}  |  N={n_sensors}  |  rel-L2={rel_l2:.3f}  "
                            f"⟨std_z⟩={std_global:.3f}  ⟨σ_y⟩={sigma_global:.3f}",
                            fontsize=9)
            ax.grid(alpha=0.25)
            if r_i == 0 and c_i == 0:
                ax.legend(fontsize=7, loc="upper right")
            if r_i == len(PDE_NAMES) - 1:
                ax.set_xlabel("x")
            if c_i == 0:
                ax.set_ylabel(f"{sys_name}\nu(x)")

    fig.suptitle("V4 NP ensemble — sample K=30 z's from q(z|C), decode each.\n"
                  "Blue band = spread from z-sampling.  Red band = heteroscedastic σ_y from decoder.\n"
                  "If V4's posterior were input-aware, the blue band should NARROW going 32→256 sensors.",
                  fontsize=11, y=1.005)
    plt.tight_layout()
    tag = args.np_ckpt.replace(".pt", "")
    p = f"{OUT}/{tag}_recon_ensemble.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}", flush=True)

    # ----- separate plot: how does ensemble spread vary with N? -----
    # Aggregate: pick 20 held-out trajectories per system, encode at N ∈
    # {16, 32, 64, 128, 256, 512, 1024}, report mean(std across samples).
    N_LIST = [16, 32, 64, 128, 256, 512, 1024]
    summary = {n: {} for n in PDE_NAMES}
    print(f"\n=== ensemble-spread vs N (avg over 20 held-out trajs) ===", flush=True)
    for sys_name in PDE_NAMES:
        d = load_g1_system(sys_name)
        u_all = d["u"]
        held = rng.permutation(u_all.shape[0])[u_all.shape[0]//5:]
        sel = held[:20]
        t_idx = u_all.shape[1] // 2
        line = f"  {sys_name:22s}  "
        for n_sensors in N_LIST:
            spreads = []
            for tidx in sel:
                u_t = torch.from_numpy(u_all[tidx, t_idx][None]).to(device).float()
                mu, lv, _ = encode_sparse(v4, u_t, full_coords, n_sensors, seed=int(tidx))
                Mu_y, _ = sample_predictions(v4, mu, lv, full_coords, K=15)
                spreads.append(float(Mu_y.std(axis=0).mean()))
            avg_spread = float(np.mean(spreads))
            summary[sys_name][n_sensors] = avg_spread
            line += f"N={n_sensors}:{avg_spread:.3f}  "
        print(line, flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for sys_name in PDE_NAMES:
        ys = [summary[sys_name][n] for n in N_LIST]
        ax.plot(N_LIST, ys, marker="o", linewidth=1.5, label=sys_name)
    ax.set_xscale("log")
    ax.set_xlabel("N sensors (context size)")
    ax.set_ylabel("ensemble spread  ⟨std(μ_y) across K=15 samples⟩")
    ax.set_title("V4 NP — does the predictive ensemble narrow as context grows?\n"
                  "(if posterior were input-aware, expect a clear monotone decrease)",
                  fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    plt.tight_layout()
    p2 = f"{OUT}/{tag}_spread_vs_N.png"
    fig.savefig(p2, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved {p2}", flush=True)

    import json
    json.dump(summary, open(f"{OUT}/{tag}_spread_vs_N.json", "w"), indent=2)


if __name__ == "__main__":
    main()
