"""More intuitive consistency visualization + sparse-recon-vs-N curves.

Outputs:
  - results/probes/g1/consistency_demo.png : per-trajectory, show 4 different
    sensor subsets and their reconstructions, with pairwise cosine similarity
    of the latents shown.
  - results/probes/g1/recon_vs_N.png : rel-L2 vs N sensors for V3 / MLP
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models import FAE as V3
from src.models.baselines import MLPSparseAE
from src.data.g1 import load_g1_system, PDE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "results", "checkpoints", "g1")
OUT = os.path.join(ROOT, "results", "probes", "g1")
os.makedirs(OUT, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def make_coords(N=1024):
    return torch.linspace(0, 1 - 1.0/N, N, device=device).unsqueeze(-1)


def load_v3():
    ck = torch.load(os.path.join(CKPT, "fae_recon.pt"), map_location=device, weights_only=False)
    m = V3(**ck["config"]).to(device).eval()
    m.load_state_dict(ck["model"])
    for p in m.parameters(): p.requires_grad_(False)
    return m


def load_mlp():
    ck = torch.load(os.path.join(CKPT, "mlp.pt"), map_location=device, weights_only=False)
    m = MLPSparseAE(coord_dim=1, latent_dim=320, enc_emb=640, dec_emb=640).to(device).eval()
    m.load_state_dict(ck["model"])
    for p in m.parameters(): p.requires_grad_(False)
    return m


@torch.no_grad()
def encode_and_decode(model, kind, gt, sensor_idx, full_coords):
    """gt: (X,) numpy. Returns (latent (D,), recon (X,) numpy)."""
    u = torch.from_numpy(gt).to(device).float()
    coords_in = full_coords[sensor_idx]
    u_in = u[sensor_idx].unsqueeze(0).unsqueeze(-1)
    if kind == "v3":
        tok = model.encoder(u_in, coords_in)
        z = tok.mean(dim=1)
        pred = model.decoder(tok, full_coords.unsqueeze(0))
    else:  # mlp
        z = model.encoder(u_in, coords_in.unsqueeze(0))
        pred = model.decoder(z, full_coords.unsqueeze(0))
    return z.squeeze(0).cpu().numpy(), pred.squeeze().cpu().numpy()


def plot_consistency_demo():
    """For one trajectory per PDE class, encode with 4 different sensor subsets
    of size M=32, show:
      - GT  (top)
      - 4 reconstructions, one per subset
      - cos sim between all latent pairs (matrix in title)
    """
    v3 = load_v3()
    mlp = load_mlp()
    full_coords = make_coords()
    X = 1024
    M = 32                                           # very sparse to make differences visible

    # Sample 4 different sensor subsets
    subsets = []
    for s in range(4):
        g = torch.Generator(); g.manual_seed(s * 7)
        idx = torch.randperm(X, generator=g)[:M].sort().values.to(device)
        subsets.append(idx)

    fig, axes = plt.subplots(len(PDE_NAMES), 6,
                              figsize=(3.2 * 6, 3.0 * len(PDE_NAMES)),
                              gridspec_kw={"hspace": 0.5})

    for r, name in enumerate(PDE_NAMES):
        d = load_g1_system(name)
        u = d["u"][0]
        t_mid = u.shape[0] // 2
        gt = u[t_mid].astype(np.float32)
        vmin, vmax = gt.min(), gt.max()

        # Encode with each subset
        for method_name, model, kind in [("V3", v3, "v3"), ("MLP", mlp, "mlp")]:
            latents = []
            recons = []
            for idx in subsets:
                z, recon = encode_and_decode(model, kind, gt, idx, full_coords)
                latents.append(z)
                recons.append(recon)

            # cosine between all latent pairs
            L = np.stack(latents)
            L_n = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
            cos_mat = L_n @ L_n.T
            mask = np.triu(np.ones_like(cos_mat, dtype=bool), k=1)
            mean_cos = cos_mat[mask].mean()

            # Plot row: this is half the row (only V3 or MLP)
            # Col 0: GT
            # Col 1-4: 4 subsets' reconstructions
            # Col 5: text summary
            if method_name == "V3" and r == 0:
                # Annotate columns once
                pass

            # Use top half of row for V3, bottom for MLP? No too complex.
            # Instead make two figures
            pass

        # Simpler design: per PDE row, only 1 method (V3), then add MLP as second fig
        # Let's do V3 first
        latents_v3 = []
        recons_v3 = []
        for idx in subsets:
            z, recon = encode_and_decode(v3, "v3", gt, idx, full_coords)
            latents_v3.append(z); recons_v3.append(recon)
        L = np.stack(latents_v3)
        L_n = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
        cos_mat = L_n @ L_n.T
        mask = np.triu(np.ones_like(cos_mat, dtype=bool), k=1)
        mean_cos_v3 = cos_mat[mask].mean()

        axes[r, 0].plot(gt, color="black", linewidth=1.5)
        axes[r, 0].set_title(f"{name}\nGT (t={t_mid})", fontsize=10)
        axes[r, 0].set_xlim(0, X-1); axes[r, 0].grid(alpha=0.3)
        axes[r, 0].set_ylim(vmin - 0.1*(vmax-vmin), vmax + 0.1*(vmax-vmin))

        for c, (idx, recon) in enumerate(zip(subsets, recons_v3)):
            ax = axes[r, c + 1]
            ax.plot(gt, color="black", linewidth=0.8, alpha=0.4, label="GT")
            ax.plot(recon, color="tab:blue", linewidth=1.0, label="V3")
            ax.scatter(idx.cpu().numpy(), gt[idx.cpu().numpy()],
                        c="lime", s=18, marker="x", zorder=5, label=f"M={M}")
            rel = np.linalg.norm(recon - gt) / max(np.linalg.norm(gt), 1e-8)
            ax.set_title(f"V3 subset {c}\nrel-L2={rel:.3f}", fontsize=9)
            ax.set_xlim(0, X-1); ax.grid(alpha=0.3)
            ax.set_ylim(vmin - 0.1*(vmax-vmin), vmax + 0.1*(vmax-vmin))
            if r == 0 and c == 0:
                ax.legend(fontsize=8)

        axes[r, 5].axis("off")
        txt = (f"V3 cos sim matrix\n(latent agreement\nbetween 4 subsets)\n\n"
                f"{cos_mat[0,1]:.4f}  {cos_mat[0,2]:.4f}  {cos_mat[0,3]:.4f}\n"
                f"      {cos_mat[1,2]:.4f}  {cos_mat[1,3]:.4f}\n"
                f"            {cos_mat[2,3]:.4f}\n\n"
                f"mean = {mean_cos_v3:.4f}")
        axes[r, 5].text(0.05, 0.5, txt, transform=axes[r, 5].transAxes,
                          fontsize=9, va="center", family="monospace")

    fig.suptitle("V3-recon: same field, 4 different M=32 sensor subsets, 4 reconstructions",
                  fontsize=13, y=1.005)
    plt.tight_layout()
    out = os.path.join(OUT, "consistency_demo_v3.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}", flush=True)

    # ---- Same plot for MLP ----
    fig, axes = plt.subplots(len(PDE_NAMES), 6,
                              figsize=(3.2 * 6, 3.0 * len(PDE_NAMES)),
                              gridspec_kw={"hspace": 0.5})
    for r, name in enumerate(PDE_NAMES):
        d = load_g1_system(name)
        u = d["u"][0]
        t_mid = u.shape[0] // 2
        gt = u[t_mid].astype(np.float32)
        vmin, vmax = gt.min(), gt.max()
        latents_mlp = []; recons_mlp = []
        for idx in subsets:
            z, recon = encode_and_decode(mlp, "mlp", gt, idx, full_coords)
            latents_mlp.append(z); recons_mlp.append(recon)
        L = np.stack(latents_mlp)
        L_n = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
        cos_mat = L_n @ L_n.T
        mask = np.triu(np.ones_like(cos_mat, dtype=bool), k=1)
        mean_cos_mlp = cos_mat[mask].mean()
        axes[r, 0].plot(gt, color="black", linewidth=1.5)
        axes[r, 0].set_title(f"{name}\nGT (t={t_mid})", fontsize=10)
        axes[r, 0].set_xlim(0, X-1); axes[r, 0].grid(alpha=0.3)
        axes[r, 0].set_ylim(vmin - 0.1*(vmax-vmin), vmax + 0.1*(vmax-vmin))
        for c, (idx, recon) in enumerate(zip(subsets, recons_mlp)):
            ax = axes[r, c + 1]
            ax.plot(gt, color="black", linewidth=0.8, alpha=0.4, label="GT")
            ax.plot(recon, color="tab:orange", linewidth=1.0, label="MLP")
            ax.scatter(idx.cpu().numpy(), gt[idx.cpu().numpy()],
                        c="lime", s=18, marker="x", zorder=5)
            rel = np.linalg.norm(recon - gt) / max(np.linalg.norm(gt), 1e-8)
            ax.set_title(f"MLP subset {c}\nrel-L2={rel:.3f}", fontsize=9)
            ax.set_xlim(0, X-1); ax.grid(alpha=0.3)
            ax.set_ylim(vmin - 0.1*(vmax-vmin), vmax + 0.1*(vmax-vmin))
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
        axes[r, 5].axis("off")
        txt = (f"MLP cos sim matrix\n(latent agreement\nbetween 4 subsets)\n\n"
                f"{cos_mat[0,1]:.4f}  {cos_mat[0,2]:.4f}  {cos_mat[0,3]:.4f}\n"
                f"      {cos_mat[1,2]:.4f}  {cos_mat[1,3]:.4f}\n"
                f"            {cos_mat[2,3]:.4f}\n\n"
                f"mean = {mean_cos_mlp:.4f}")
        axes[r, 5].text(0.05, 0.5, txt, transform=axes[r, 5].transAxes,
                          fontsize=9, va="center", family="monospace")
    fig.suptitle("MLP-sparse: same field, 4 different M=32 sensor subsets, 4 reconstructions",
                  fontsize=13, y=1.005)
    plt.tight_layout()
    out = os.path.join(OUT, "consistency_demo_mlp.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}", flush=True)


def plot_recon_vs_N():
    """For each PDE class, sweep N and plot mean rel-L2 for V3 / MLP.

    Also include N=1024 (full grid) at the right end for fair comparison
    with dense methods (which require N=1024).
    """
    v3 = load_v3()
    mlp = load_mlp()
    full_coords = make_coords()
    X = 1024
    Ns = [8, 16, 32, 64, 128, 256, 512, 1024]
    n_samples = 10
    results = {name: {"v3": [], "mlp": [], "v3_std": [], "mlp_std": []} for name in PDE_NAMES}

    for name in PDE_NAMES:
        d = load_g1_system(name)
        u = d["u"][:n_samples]
        t_mid = u.shape[1] // 2
        frames = u[:, t_mid].astype(np.float32)
        for n in Ns:
            v3_errs = []; mlp_errs = []
            for i in range(n_samples):
                gt = frames[i]
                g = torch.Generator(); g.manual_seed(i)
                idx = torch.randperm(X, generator=g)[:n].sort().values.to(device)
                _, recon_v3 = encode_and_decode(v3, "v3", gt, idx, full_coords)
                _, recon_mlp = encode_and_decode(mlp, "mlp", gt, idx, full_coords)
                v3_errs.append(np.linalg.norm(recon_v3 - gt) / max(np.linalg.norm(gt), 1e-8))
                mlp_errs.append(np.linalg.norm(recon_mlp - gt) / max(np.linalg.norm(gt), 1e-8))
            results[name]["v3"].append(np.mean(v3_errs))
            results[name]["v3_std"].append(np.std(v3_errs))
            results[name]["mlp"].append(np.mean(mlp_errs))
            results[name]["mlp_std"].append(np.std(mlp_errs))
        print(f"  {name} done", flush=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for ax, name in zip(axes, PDE_NAMES):
        ax.errorbar(Ns, results[name]["v3"], yerr=results[name]["v3_std"],
                      fmt="o-", color="tab:blue", linewidth=2, capsize=3, label="V3 recon")
        ax.errorbar(Ns, results[name]["mlp"], yerr=results[name]["mlp_std"],
                      fmt="s-", color="tab:orange", linewidth=2, capsize=3, label="MLP")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(Ns); ax.set_xticklabels(Ns)
        ax.set_xlabel("N sensors"); ax.set_title(name, fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
        if name == PDE_NAMES[0]: ax.set_ylabel("rel-L2 reconstruction error")
    fig.suptitle("Reconstruction error vs N sensors (V3 vs MLP, mean ± std over 10 val trajectories)",
                  fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT, "recon_vs_N.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    plot_consistency_demo()
    plot_recon_vs_N()
    print("ALL DONE", flush=True)
