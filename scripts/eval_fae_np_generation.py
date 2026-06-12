"""Generation check for an FAE-NP checkpoint.

Three modes, per system:
  (a) UNCONDITIONAL  z ~ N(0, I) -> decode.
      Meaningful only for anchor-trained models (KL(q(z|C) || N(0,I)) > 0);
      the pre-fix checkpoints fail this by construction.
  (b) MOMENT-MATCHED  z ~ N(mean(mu), diag(std(mu)^2)) over encoded training
      latents -> decode. A weak latent prior; the floor for "is the decoder
      usable off-manifold at all".
  (c) CONDITIONAL  z ~ q(z|C) from N_ctx context points -> decode; report
      sample spread / field std vs N_ctx (functional-uncertainty calibration:
      spread should shrink as context grows).

Metrics per system:
  - log-spectrum L1 between real snapshots and (a)/(b) samples (modes 1..59)
  - decoded std ratio vs real field std for (a)/(b)
  - conditional spread/field-std at N_ctx in {32, 64, 256}
Saves results/probes/g1/<tag>_generation.json and a figure per run.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.fae_np import FAENP
from src.data.g1 import load_g1_system, make_coords_1d, PDE_NAMES

device = os.environ.get("EVAL_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = f"{ROOT}/results/checkpoints/g1"
OUT  = f"{ROOT}/results/probes/g1"
X = 1024
N_ENC = 1000          # training latents per system for the moment-matched prior
N_GEN = 64            # generated samples per mode
N_CTX_LIST = (32, 64, 256)
K_COND = 8            # conditional samples per context


def log_spectrum(u, k_lo=1, k_hi=60):
    f = np.abs(np.fft.rfft(u, axis=-1))[:, k_lo:k_hi]
    return np.log10(f + 1e-8).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np_ckpt", required=True,
                     help="filename under results/checkpoints/g1/")
    args = ap.parse_args()
    tag = args.np_ckpt.replace(".pt", "")

    ck = torch.load(f"{CKPT}/{args.np_ckpt}", map_location=device, weights_only=False)
    m = FAENP(**ck["config"]).to(device).eval()
    m.load_state_dict(ck["model"])
    for p in m.parameters(): p.requires_grad_(False)
    coords = make_coords_1d(device, N=X)
    print(f"=== generation check: {tag} "
          f"(logvar_param={ck['config'].get('logvar_param', 'clamp')}, "
          f"anchor={ck.get('np_config', {}).get('anchor', 'n/a')}) ===", flush=True)

    rng = np.random.default_rng(0)
    results = {}
    fig, axes = plt.subplots(len(PDE_NAMES), 4,
                                figsize=(19, 3.4 * len(PDE_NAMES)))

    for row, sys_name in enumerate(PDE_NAMES):
        d = load_g1_system(sys_name)
        perm = rng.permutation(d["u"].shape[0])
        U_tr = d["u"][perm[:N_ENC], 50].astype(np.float32)
        u_gt = torch.from_numpy(d["u"][perm[N_ENC], 50].astype(np.float32)).to(device)

        # encode training latents (N=256 uniform) for the moment-matched prior
        idx = torch.arange(0, X, 4, device=device)
        mus = []
        with torch.no_grad():
            for i0 in range(0, N_ENC, 64):
                x = torch.from_numpy(U_tr[i0:i0+64]).to(device)
                mu, _ = m.encode_distribution(x[:, idx].unsqueeze(-1), coords[idx])
                mus.append(mu.cpu())
        MU = torch.cat(mus).numpy()

        with torch.no_grad():
            # (a) unconditional from N(0, I)
            z_a = torch.randn(N_GEN, m.d_latent, device=device)
            u_a = m.decode(z_a, coords)[0].cpu().numpy()
            # (b) moment-matched diagonal Gaussian on aggregate posterior
            mu0 = torch.from_numpy(MU.mean(0)).to(device)
            sd0 = torch.from_numpy(MU.std(0)).to(device)
            z_b = mu0 + sd0 * torch.randn(N_GEN, m.d_latent, device=device)
            u_b = m.decode(z_b, coords)[0].cpu().numpy()
            # (c) conditional spread vs N_ctx
            cond = {}
            for n_ctx in N_CTX_LIST:
                cidx = torch.from_numpy(np.sort(
                    rng.choice(X, n_ctx, replace=False))).to(device)
                tokens_c = m.encoder(u_gt[cidx].view(1, -1, 1), coords[cidx])
                mu_c, lv_c = m.latent_head(tokens_c)
                zs = mu_c + (0.5 * lv_c).exp() * torch.randn(
                    K_COND, m.d_latent, device=device)
                det = (tokens_c.expand(K_COND, -1, -1)
                        if getattr(m, "det_path", False) else None)
                uc = m.decode(zs, coords, det_tokens=det)[0]
                cond[n_ctx] = float(uc.std(0).mean() / u_gt.std())

        spec_real = log_spectrum(U_tr)
        rec = {
            "uncond_spec_l1":  float(np.abs(log_spectrum(u_a) - spec_real).mean()),
            "momatch_spec_l1": float(np.abs(log_spectrum(u_b) - spec_real).mean()),
            "uncond_std_ratio":  float(u_a.std() / U_tr.std()),
            "momatch_std_ratio": float(u_b.std() / U_tr.std()),
            "cond_spread_over_fieldstd": cond,
            "mu_abs_mean": float(np.abs(MU).mean()),
            "mu_norm_mean": float(np.linalg.norm(MU, axis=1).mean()),
        }
        results[sys_name] = rec
        print(f"  {sys_name:20s} uncond_specL1={rec['uncond_spec_l1']:.2f}  "
              f"momatch_specL1={rec['momatch_spec_l1']:.2f}  "
              f"cond_spread={ {k: round(v, 3) for k, v in cond.items()} }  "
              f"|mu|={rec['mu_abs_mean']:.2f}", flush=True)

        xs = np.linspace(0, 1, X)
        for u in u_a[:4]: axes[row, 0].plot(xs, u, lw=0.8)
        axes[row, 0].set_title(f"{sys_name}: (a) z~N(0,I)", fontsize=9)
        for u in u_b[:4]: axes[row, 1].plot(xs, u, lw=0.8)
        axes[row, 1].set_title("(b) moment-matched", fontsize=9)
        for u in U_tr[:4]: axes[row, 2].plot(xs, u, lw=0.8)
        axes[row, 2].set_title("real", fontsize=9)
        axes[row, 3].plot(spec_real, label="real", lw=1.8)
        axes[row, 3].plot(log_spectrum(u_b), label="(b)", lw=1.4)
        axes[row, 3].plot(log_spectrum(u_a), label="(a)", lw=1.4, ls=":")
        axes[row, 3].set_title("mean log-spectrum", fontsize=9)
        if row == 0: axes[row, 3].legend(fontsize=8)

    plt.tight_layout()
    p = f"{OUT}/{tag}_generation.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    json.dump(results, open(f"{OUT}/{tag}_generation.json", "w"), indent=2)
    print(f"saved {OUT}/{tag}_generation.json and {p}", flush=True)


if __name__ == "__main__":
    main()
