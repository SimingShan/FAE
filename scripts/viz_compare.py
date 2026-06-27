"""Two figures for the fair NS-buoyancy comparison:
  1) convergence.png  — train loss + in-loop probe R^2 vs epoch (FAE / MAE / JEPA), shows all converged.
  2) reconstruction.png — one NS test sample: input | what-the-model-sees | output | |error|.
     FAE (native coord decoder, sparse->dense) and MAE (native decoder, 75% masked) reconstruct natively.
     JEPA has NO pixel decoder (latent prediction) -> we fit a minimal shared per-patch linear readout
     to visualize recoverable content (this is exactly the RAE motivation).
"""
import os, re, glob
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEV = "cpu"
ROOT = "/gpfs/radev/scratch/lu_lu/ss5235/WFAE"
from src.config import ckpt_file
OUT = os.path.join(ROOT, "results/figures"); os.makedirs(OUT, exist_ok=True)
COL = {"FAE": "#d62728", "MAE": "#1f77b4", "JEPA": "#2ca02c"}

# ----------------------------------------------------------------------
# 1) convergence — parse logs
# ----------------------------------------------------------------------
LOGS = {"FAE": "logs/run_fae_2027851.out", "MAE": "logs/run_mae_2027852.out", "JEPA": "logs/run_jepa_2027853.out"}

def parse_log(path):
    ep, loss, r2 = [], [], []
    for ln in open(os.path.join(ROOT, path)):
        m = re.match(r"ep\s+(\d+)/\d+\s+(?:rec|loss)=([0-9.eE+-]+).*?buoyancy=([+\-0-9.]+)", ln)
        if m:
            ep.append(int(m.group(1))); loss.append(float(m.group(2))); r2.append(float(m.group(3)))
    return np.array(ep), np.array(loss), np.array(r2)

fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
for name, lp in LOGS.items():
    ep, loss, r2 = parse_log(lp)
    a0.plot(ep, r2, "-o", ms=3, color=COL[name], label=name)
    a1.semilogy(ep, loss, "-o", ms=3, color=COL[name], label=name)
a0.set(xlabel="epoch", ylabel="in-loop probe $R^2$ (buoyancy)", title="Probe $R^2$ vs epoch — all converged")
a0.axhline(0.0, color="gray", lw=.7, ls=":"); a0.grid(alpha=.3); a0.legend(); a0.set_ylim(-0.05, 1.0)
a1.set(xlabel="epoch", ylabel="training loss (own objective, log)", title="Training loss vs epoch (own scale)")
a1.grid(alpha=.3, which="both"); a1.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "convergence.png"), dpi=150)
print("wrote convergence.png")

# ----------------------------------------------------------------------
# 2) reconstruction
# ----------------------------------------------------------------------
import sys; sys.path.insert(0, ROOT)
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens

SIDE = 128; CH = 0  # smoke channel
ds = NSDataset("test", side=SIDE, mode="clip", clip_len=2, frame_stride=4, n_traj=8)
clips = torch.stack([ds[i][0] for i in range(min(64, len(ds)))])      # (N,C,T,H,W)
x = clips[:, :, 0]                                                     # frame-0 fields (N,C,H,W)
sample = 7
img = x[sample:sample+1].to(DEV)                                      # (1,C,H,W)

# --- FAE: native sparse->dense ---
fa = torch.load(ckpt_file("fae","fae_ns128",0), map_location=DEV); afa = fa["train_args"]
from src.models.fae import FAE
IN_CH = 3
fae = FAE(emb_dim=afa["emb_dim"], num_iter=afa.get("num_iter", 4), num_latents=afa["num_latents"],
          in_chans=IN_CH, coord_dim=2).to(DEV); fae.load_state_dict(fa["model"]); fae.eval()
coords = make_coords_2d(n_side=SIDE, device=DEV)
g = torch.Generator(device=DEV).manual_seed(0)
sidx = torch.randperm(SIDE * SIDE, generator=g, device=DEV)[:1024]
with torch.no_grad():
    u = fields_to_tokens(img, sidx)
    pred, _ = fae(u, coords[sidx], coords)                            # (1, NPIX, C)
fae_rec = pred[0].T.reshape(IN_CH, SIDE, SIDE).cpu()
fae_see = torch.full((SIDE * SIDE,), float("nan")); fae_see[sidx.cpu()] = img[0, CH].reshape(-1)[sidx.cpu()]
fae_see = fae_see.reshape(SIDE, SIDE)

# --- MAE: native masked recon ---
ma = torch.load(ckpt_file("mae","mae_ns128",0), map_location=DEV); ama = ma["train_args"]
from scripts.train_baseline import build_model
mae = build_model("mae", resolution=SIDE, in_chans=IN_CH, embed_dim=ama["embed_dim"],
                  depth=ama["depth"], patch_size=ama["patch_size"]).to(DEV)
mae.load_state_dict(ma["model"]); mae.eval()
torch.manual_seed(0)
with torch.no_grad():
    loss, pdt, mask = mae(img, 0.75)                                 # pred patches, mask 1=removed
    orig_p = mae.patchify(img)
    disp = orig_p * (1 - mask.unsqueeze(-1)) + pdt * mask.unsqueeze(-1)
    mae_rec = mae.unpatchify(disp)[0].cpu()
    masked_p = orig_p * (1 - mask.unsqueeze(-1))                     # zero the removed patches for "sees"
    mae_see = mae.unpatchify(masked_p)[0, CH].cpu()
    mae_see[mae.unpatchify(mask.unsqueeze(-1).expand(-1, -1, orig_p.shape[-1]).float())[0, CH].cpu() > 0] = float("nan")

# --- JEPA: latent only -> fit shared per-patch linear readout (RAE-lite) ---
ja = torch.load(ckpt_file("jepa","jepa_ns128",0), map_location=DEV); aja = ja["train_args"]
jepa = build_model("ijepa", resolution=SIDE, in_chans=IN_CH, embed_dim=aja["embed_dim"],
                   depth=aja["depth"], patch_size=aja["patch_size"]).to(DEV)
jepa.load_state_dict(ja["model"]); jepa.eval()
P = aja["patch_size"]; G = SIDE // P; C = IN_CH
def patchify(im):                                                    # (B,C,H,W)->(B,G*G,P*P*C)
    B = im.shape[0]; t = im.reshape(B, C, G, P, G, P)
    return t.permute(0, 2, 4, 3, 5, 1).reshape(B, G * G, P * P * C)
def unpatchify(t):
    B = t.shape[0]; t = t.reshape(B, G, G, P, P, C)
    return t.permute(0, 5, 1, 3, 2, 4).reshape(B, C, SIDE, SIDE)
with torch.no_grad():
    F = jepa.target(x.to(DEV))                                       # (N,P,D) features for fit
    T = patchify(x.to(DEV))                                          # (N,P,p*p*C) targets
Ff = F.reshape(-1, F.shape[-1]); Tf = T.reshape(-1, T.shape[-1])     # per-patch least squares
W = torch.linalg.lstsq(Ff, Tf).solution                             # (D, p*p*C)
with torch.no_grad():
    Fi = jepa.target(img); jrec = unpatchify((Fi @ W).reshape(1, G * G, -1))[0].cpu()

# ---- render: rows = methods, cols = input | sees | output | |error| ----
inp = img[0, CH].cpu()
vmin, vmax = inp.min().item(), inp.max().item()
rows = [("FAE", fae_see, fae_rec[CH], "1024 sensors"),
        ("MAE", mae_see, mae_rec[CH], "75% masked"),
        ("JEPA", None, jrec[CH], "full img (latent)")]
fig, ax = plt.subplots(3, 4, figsize=(13, 9.6))
for r, (name, see, rec, seetxt) in enumerate(rows):
    ax[r, 0].imshow(inp, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax[r, 0].set_ylabel(name, fontsize=15, fontweight="bold", color=COL[name])
    if see is not None:
        ax[r, 1].imshow(see, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    else:
        ax[r, 1].imshow(inp, vmin=vmin, vmax=vmax, cmap="RdBu_r"); ax[r, 1].text(
            .5, .5, "no decoder\n(latent only)", ha="center", va="center", transform=ax[r, 1].transAxes,
            fontsize=11, color="white", fontweight="bold")
    ax[r, 1].set_title(f"sees: {seetxt}" if r == 0 else seetxt, fontsize=10)
    ax[r, 2].imshow(rec, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax[r, 2].set_title("reconstruction" if r == 0 else "", fontsize=11)
    err = (rec - inp).abs(); relL2 = (err.norm() / (inp.norm() + 1e-8)).item()
    print(f"{name:5s} recon relL2 = {relL2:.3f}")
    im = ax[r, 3].imshow(err, vmin=0, vmax=(vmax - vmin) * 0.5, cmap="magma")
    ax[r, 3].set_title((("abs error\n" if r == 0 else "")) + f"relL2={relL2:.3f}",
                       fontsize=11 if r == 0 else 10)
ax[0, 0].set_title("input (smoke)", fontsize=11)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
ax[2, 2].set_xlabel("linear readout (RAE-lite)", fontsize=9)
fig.suptitle("NS-buoyancy held-out sample — native recon (FAE, MAE) vs latent-only (JEPA needs a decoder)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(OUT, "reconstruction.png"), dpi=150)
print("wrote reconstruction.png")
print(f"FAE PR / MAE / JEPA recon shown; sample={sample}")
