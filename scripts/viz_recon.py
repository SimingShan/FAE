"""FAE reconstruction from K sparse sensors (the Senseiver capability), for shear / typhoon.
true field vs FAE recon at decreasing sensor counts -> shows graceful degradation. Channel 0.
Writes results/figs/<dataset>/recon_fae.png."""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.plotstyle import apply; apply()                                # canonical style
from src.models.fae import FAE
from src.data.well2d import make_coords_2d, fields_to_tokens
from scripts.probe_all import get_data, _frame0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
ap.add_argument("--sensors", type=int, nargs="+", default=[None, 256, 64, 16])
ap.add_argument("--n", type=int, default=3, help="num sample fields")
args = ap.parse_args()
side = {"shear": 128, "typhoon": 128, "ns": 128}[args.dataset]
ck = sorted(glob.glob(f"results/checkpoints/{args.dataset}/fae/*_s*.pt"))[0]
a = torch.load(ck, map_location=DEVICE)["train_args"]
inc = a.get("in_chans") or (1 if args.dataset == "typhoon" else 4 if args.dataset == "shear" else 3)
m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), depth_per_iter=a.get("depth_per_iter", 4),
        num_latents=a["num_latents"], n_freq=32, max_freq=32, coord_dim=2, in_chans=inc).to(DEVICE)
m.load_state_dict(torch.load(ck, map_location=DEVICE)["model"]); m.eval()
coords = make_coords_2d(n_side=side, device=DEVICE); NPIX = side * side
g = torch.Generator(device=DEVICE).manual_seed(0)

ds = get_data(args.dataset, "valid", side)
sens = [s if s else NPIX for s in args.sensors]
cols = 1 + len(sens)
fig, axs = plt.subplots(args.n, cols, figsize=(cols * 2, args.n * 2))


@torch.no_grad()
def recon(f0, k):
    idx = torch.arange(NPIX, device=DEVICE) if k >= NPIX else torch.randperm(NPIX, generator=g, device=DEVICE)[:k]
    tok = m.encode_tokens(fields_to_tokens(f0, idx), coords[idx])
    return m.decoder(tok, coords).permute(0, 2, 1).reshape(1, inc, side, side)


def relL2(p, t): return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8))

step = max(1, len(ds) // (args.n + 1))
for r in range(args.n):
    f0 = _frame0(ds[r * step][0].unsqueeze(0)).to(DEVICE)
    true = f0[0, 0].cpu().numpy(); vmax = float(np.abs(true).max()) or 1.0
    axs[r, 0].imshow(true, cmap="RdBu_r", vmin=-vmax, vmax=vmax); axs[r, 0].set_xticks([]); axs[r, 0].set_yticks([])
    if r == 0: axs[0, 0].set_title("true", fontsize=12)
    for c, k in enumerate(sens):
        rc = recon(f0, k)[0, 0].cpu().numpy()
        axs[r, c + 1].imshow(rc, cmap="RdBu_r", vmin=-vmax, vmax=vmax); axs[r, c + 1].set_xticks([]); axs[r, c + 1].set_yticks([])
        if r == 0: axs[0, c + 1].set_title(f"{'full' if k>=NPIX else k} sensors", fontsize=11)
        axs[r, c + 1].set_xlabel(f"relL2={relL2(rc, true):.2f}", fontsize=12)
out = f"results/figs/{args.dataset}/recon_fae.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}", flush=True)
