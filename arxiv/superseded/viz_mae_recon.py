"""MAE reconstruction: true field vs 75%-masked input vs MAE recon, for shear / typhoon. Channel 0.
Writes results/figs/<dataset>/recon_mae.png."""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.plotstyle import apply; apply()                                # canonical style
from scripts.train_baseline import build_model
from scripts.probe_all import get_data, _frame0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
ap.add_argument("--mask", type=float, default=0.75)
ap.add_argument("--n", type=int, default=3)
args = ap.parse_args()
side = {"shear": 128, "typhoon": 128, "ns": 128}[args.dataset]
ck = sorted(glob.glob(f"results/checkpoints/{args.dataset}/mae/*_s*.pt"))[0]
a = torch.load(ck, map_location=DEVICE)["train_args"]
m = build_model("mae", resolution=a["resolution"], in_chans=a["in_chans"], embed_dim=a.get("embed_dim"),
                depth=a.get("depth"), patch_size=a.get("patch_size")).to(DEVICE)
m.load_state_dict(torch.load(ck, map_location=DEVICE)["model"]); m.eval()
torch.manual_seed(0)
ds = get_data(args.dataset, "valid", side)


@torch.no_grad()
def recon(f0):
    loss, pred, mask = m(f0, args.mask)                            # pred (B,L,pd), mask (B,L) 1=masked
    p = m.patchify(f0); mk = mask[:, :, None]
    paste = p * (1 - mk) + pred * mk                               # standard MAE viz: visible=orig, masked=pred
    return m.unpatchify(p * (1 - mk)), m.unpatchify(paste)         # (masked input, reconstruction)


def relL2(p, t): return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8))

step = max(1, len(ds) // (args.n + 1))
fig, axs = plt.subplots(args.n, 3, figsize=(3 * 2, args.n * 2))
for r in range(args.n):
    f0 = _frame0(ds[r * step][0].unsqueeze(0)).to(DEVICE)
    vis, rc = recon(f0)
    t = f0[0, 0].cpu().numpy(); v = vis[0, 0].cpu().numpy(); rr = rc[0, 0].cpu().numpy()
    vmax = float(np.abs(t).max()) or 1.0
    for c, (lab, img) in enumerate([("true", t), (f"{int(args.mask*100)}% masked", v), ("MAE recon", rr)]):
        ax = axs[r, c]; ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(lab, fontsize=12)
    axs[r, 2].set_xlabel(f"relL2={relL2(rr, t):.2f}", fontsize=12)
out = f"results/figs/{args.dataset}/recon_mae.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}", flush=True)
