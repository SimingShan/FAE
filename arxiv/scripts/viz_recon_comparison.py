"""Reconstruction comparison: GT vs V3-recon vs MLP vs CNN, one row per PDE."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt

from src.models import FAE as V3
from src.models.baselines import MLPSparseAE, CNN1DAE
from src.data.g1 import load_g1_system, PDE_NAMES

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = f"{ROOT}/results/checkpoints/g1"
OUT = f"{ROOT}/results/probes/g1"

def make_coords(N=1024):
    return torch.linspace(0, 1 - 1.0/N, N, device=device).unsqueeze(-1)

def load(name, kind):
    ck = torch.load(f"{CKPT}/{name}.pt", map_location=device, weights_only=False)
    if kind == "v3":
        m = V3(**ck["config"]).to(device).eval()
    elif kind == "mlp":
        m = MLPSparseAE(coord_dim=1, latent_dim=320, enc_emb=640, dec_emb=640).to(device).eval()
    elif kind == "cnn":
        m = CNN1DAE().to(device).eval()
    m.load_state_dict(ck["model"])
    for p in m.parameters(): p.requires_grad_(False)
    return m

@torch.no_grad()
def recon_v3(model, u, idx_in, coords_in, full_coords):
    u_in = u[idx_in].unsqueeze(0).unsqueeze(-1)
    pred, _ = model(u_in, coords_in, full_coords.unsqueeze(0))
    return pred.squeeze().cpu().numpy()

@torch.no_grad()
def recon_mlp(model, u, idx_in, coords_in, full_coords):
    u_in = u[idx_in].unsqueeze(0).unsqueeze(-1)
    pred, _ = model(u_in, coords_in.unsqueeze(0), full_coords.unsqueeze(0))
    return pred.squeeze().cpu().numpy()

@torch.no_grad()
def recon_cnn(model, u_full):
    x = u_full.unsqueeze(0).unsqueeze(0)
    pred, _ = model(x)
    return pred.squeeze().cpu().numpy()


def main():
    full_coords = make_coords(1024)
    X = 1024
    Ns_sparse = [64, 256, 1024]                                  # sensor counts for sparse methods
    v3 = load("fae_recon", "v3")
    mlp = load("mlp", "mlp")
    cnn = load("cnn", "cnn")

    # Pick a representative trajectory per PDE system
    fig, axes = plt.subplots(len(PDE_NAMES), 5, figsize=(20, 12))
    rng = np.random.default_rng(7)
    for r, name in enumerate(PDE_NAMES):
        d = load_g1_system(name)
        u_traj = d["u"]
        coeff_val = d["coeff"]
        # pick a "median coefficient" trajectory
        sorted_ix = np.argsort(coeff_val)
        idx = int(sorted_ix[len(sorted_ix) // 2])
        coeff_str = f"{coeff_val[idx]:.4g}"
        u_full = u_traj[idx, u_traj.shape[1] // 2]               # mid-frame
        u_t = torch.from_numpy(u_full).to(device).float()
        vmin, vmax = float(u_full.min()), float(u_full.max())
        pad = 0.1 * (vmax - vmin + 1e-6)

        # Column 0: GT
        ax = axes[r, 0]
        ax.plot(u_full, color="black", linewidth=1.4)
        ax.set_title(f"{name}\n(coeff={coeff_str})", fontsize=10)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.grid(alpha=0.3)
        if r == 0: ax.set_ylabel("u(x, t_mid)", fontsize=10)

        # Columns 1, 2, 3: V3, MLP at N=64, 256, 1024
        for c_off, N in enumerate(Ns_sparse):
            ax = axes[r, c_off + 1]
            idx_in = torch.arange(0, X, X // N, device=device)[:N]
            coords_in = full_coords[idx_in]
            # V3 recon
            pred_v3 = recon_v3(v3, u_t, idx_in, coords_in, full_coords)
            pred_mlp = recon_mlp(mlp, u_t, idx_in, coords_in, full_coords)
            ax.plot(u_full, color="black", linewidth=1.0, alpha=0.5, label="GT")
            ax.plot(pred_v3, color="#1f77b4", linewidth=1.0, label="V3")
            ax.plot(pred_mlp, color="#ff7f0e", linewidth=1.0, alpha=0.85, label="MLP")
            sensor_x = idx_in.cpu().numpy()
            ax.scatter(sensor_x, u_full[sensor_x], color="lime",
                          s=8, marker="x", alpha=0.6, zorder=5)
            rel_v3 = np.linalg.norm(pred_v3 - u_full) / max(np.linalg.norm(u_full), 1e-8)
            rel_mlp = np.linalg.norm(pred_mlp - u_full) / max(np.linalg.norm(u_full), 1e-8)
            ax.set_title(f"N={N}\nV3:{rel_v3:.3f}  MLP:{rel_mlp:.3f}", fontsize=9)
            ax.set_ylim(vmin - pad, vmax + pad)
            ax.grid(alpha=0.3)
            if r == 0 and c_off == 0:
                ax.legend(fontsize=8, loc="best")

        # Column 4: CNN (dense, always full grid input)
        ax = axes[r, 4]
        pred_cnn = recon_cnn(cnn, u_t)
        ax.plot(u_full, color="black", linewidth=1.0, alpha=0.5, label="GT")
        ax.plot(pred_cnn, color="#9467bd", linewidth=1.0, label="CNN")
        rel_cnn = np.linalg.norm(pred_cnn - u_full) / max(np.linalg.norm(u_full), 1e-8)
        ax.set_title(f"CNN (dense)\nrel-L2:{rel_cnn:.3f}", fontsize=9)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.grid(alpha=0.3)
        if r == 0:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("Reconstruction comparison on G1 v2 (mid-frame, median-coeff trajectory)",
                  fontsize=13, y=1.005)
    plt.tight_layout()
    out = f"{OUT}/recon_comparison.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
