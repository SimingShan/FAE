"""Reconstruction viz for FAE+VICReg on trl_2D: sparse sensors -> dense field."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.models import FAE
from src.data.well2d import TRL2DSnapshotDataset, make_coords_2d, fields_to_tokens

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = W = 128
CH_NAMES = ["density", "pressure", "v_x", "v_y"]
N_SENSORS = 512

ck = torch.load("results/checkpoints/g1/fae_vicreg_trl2d.pt", map_location=DEV, weights_only=False)
m = FAE(**ck["config"]).to(DEV).eval()
m.load_state_dict(ck["model"])
coords = make_coords_2d(device=DEV)                       # (16384, 2) full grid

va = TRL2DSnapshotDataset("valid", frame_stride=4, stats=ck["stats"])
# pick 4 snapshots spanning the tcool range
order = np.argsort(va.log_tcool)
picks = [order[int(f*(len(order)-1))] for f in (0.1, 0.4, 0.6, 0.9)]

g = torch.Generator(device=DEV).manual_seed(3)
sidx = torch.randperm(H*W, generator=g, device=DEV)[:N_SENSORS]
sx = (sidx % W).cpu().numpy(); sy = (sidx // W).cpu().numpy()

@torch.no_grad()
def recon(field):                                         # field (4,H,W)
    fb = field.unsqueeze(0).to(DEV)
    vals = fields_to_tokens(fb, sidx)                     # (1, N, 4)
    pred, _ = m(vals, coords[sidx], coords)               # (1, 16384, 4)
    return pred.permute(0,2,1).reshape(4, H, W).cpu().numpy()

# === Figure 1: density channel, sparse->dense across snapshots ===
fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.1*len(picks)))
for r, p in enumerate(picks):
    field, _ = va[p]; gt = field.numpy(); rc = recon(field)
    tc = 10**va.log_tcool[p]
    vmin, vmax = gt[0].min(), gt[0].max()
    axes[r,0].imshow(gt[0], vmin=vmin, vmax=vmax, cmap="viridis")
    axes[r,0].set_ylabel(f"t_cool={tc:.2f}", fontsize=10)
    axes[r,0].set_title("ground truth (density)" if r==0 else "")
    axes[r,1].imshow(np.zeros((H,W)), cmap="gray", vmin=0, vmax=1)
    axes[r,1].scatter(sx, sy, c=gt[0][sy,sx], s=6, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[r,1].set_title(f"sparse input (N={N_SENSORS})" if r==0 else ""); axes[r,1].set_xlim(0,W); axes[r,1].set_ylim(H,0)
    axes[r,2].imshow(rc[0], vmin=vmin, vmax=vmax, cmap="viridis")
    axes[r,2].set_title("FAE reconstruction" if r==0 else "")
    err = np.abs(rc[0]-gt[0])
    axes[r,3].imshow(err, cmap="magma"); axes[r,3].set_title("|error|" if r==0 else "")
    rel = np.linalg.norm(rc-gt)/np.linalg.norm(gt)
    axes[r,3].set_xlabel(f"rel-L2(all ch)={rel:.3f}", fontsize=9)
    for a in axes[r]: a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"FAE+VICReg on trl_2D: reconstruct dense density field from {N_SENSORS} scattered sensors", y=1.0)
plt.tight_layout(); plt.savefig("results/probes/g1/fae_trl2d_recon_density.png", dpi=120, bbox_inches="tight")
print("saved results/probes/g1/fae_trl2d_recon_density.png")

# === Figure 2: all 4 channels, GT vs recon, one snapshot ===
field, _ = va[picks[2]]; gt = field.numpy(); rc = recon(field)
fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
for c in range(4):
    vmin, vmax = gt[c].min(), gt[c].max()
    axes[0,c].imshow(gt[c], vmin=vmin, vmax=vmax, cmap="RdBu_r"); axes[0,c].set_title(CH_NAMES[c])
    axes[1,c].imshow(rc[c], vmin=vmin, vmax=vmax, cmap="RdBu_r")
    rel = np.linalg.norm(rc[c]-gt[c])/np.linalg.norm(gt[c])
    axes[1,c].set_xlabel(f"rel-L2={rel:.3f}", fontsize=9)
    for a in (axes[0,c],axes[1,c]): a.set_xticks([]); a.set_yticks([])
axes[0,0].set_ylabel("ground truth", fontsize=11); axes[1,0].set_ylabel("FAE recon", fontsize=11)
fig.suptitle(f"FAE+VICReg trl_2D: all 4 channels from N={N_SENSORS} sensors  (t_cool={10**va.log_tcool[picks[2]]:.2f})", y=1.0)
plt.tight_layout(); plt.savefig("results/probes/g1/fae_trl2d_recon_4channel.png", dpi=120, bbox_inches="tight")
print("saved results/probes/g1/fae_trl2d_recon_4channel.png")
