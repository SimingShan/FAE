"""DINOv2-style per-patch feature-PCA, comparing FAE / MAE / JEPA on the SAME PDE field.
PCA the per-patch features -> RGB -> overlay. Coherent regions tracking flow structure = good
representation; spatial mush = the geometric/interpolation shortcut. Diagnostic, not decoration."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.generate import fae_setup, enc_setup, get_frames, DEVICE
from src.data.well2d import make_coords_2d, fields_to_tokens


def pca_rgb(feat):                                   # (N, D) -> (g, g, 3) in [0,1]
    f = feat.float() - feat.float().mean(0)
    _, _, V = torch.pca_lowrank(f, q=3, niter=4)
    p = f @ V[:, :3]
    p = (p - p.min(0).values) / (p.max(0).values - p.min(0).values + 1e-6)
    g = int(round(feat.shape[0] ** 0.5))
    return p.reshape(g, g, 3).cpu().numpy()


@torch.no_grad()
def fae_feat(field, R):
    m, coords, _, pc = fae_setup("results/checkpoints/g1/faep_twoview_fae_ns_tw.pt", R, 4)
    g = torch.Generator(device=DEVICE).manual_seed(0); sidx = torch.randperm(R * R, generator=g, device=DEVICE)[:2048]
    lat = m.encode_tokens(fields_to_tokens(field, sidx), coords[sidx])
    return m.decoder(lat, pc, return_feats=True)[0]          # (256, 320) at 16x16 patch centers


@torch.no_grad()
def enc_feat(method, ckpt, field, R):
    m = enc_setup(method, ckpt, R, 3)
    tok = m.forward_encoder(field, 0.0)[0][:, 1:] if method == "mae" else m.target(field)
    return tok[0]                                            # (Npatch, D), native ViT grid


def main():
    R = 64; va = get_frames("ns", "valid", R)
    field = torch.from_numpy(va[len(va) // 2][0].numpy()).unsqueeze(0).to(DEVICE)   # (1,3,R,R)
    panels = [("field (smoke)", field[0, 0].cpu().numpy(), "RdBu_r")]
    panels.append(("FAE (recon) PCA", pca_rgb(fae_feat(field, R)), None))
    panels.append(("MAE PCA", pca_rgb(enc_feat("mae", "results/checkpoints/g1/mae_shear_mae_ns_s0.pt", field, R)), None))
    panels.append(("JEPA PCA", pca_rgb(enc_feat("jepa", "results/checkpoints/g1/ijepa_shear_ijepa_ns_s0.pt", field, R)), None))
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.7))
    for a, (nm, im, cm) in zip(ax, panels):
        a.imshow(im, cmap=cm, interpolation="nearest"); a.set_title(nm, fontsize=11); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Per-patch feature PCA — does the encoder track coherent physical structure?", fontsize=12)
    fig.tight_layout(); fig.savefig("results/figs/encoder_pca.png", dpi=120)
    print("saved results/figs/encoder_pca.png  (FAE grid 16x16, MAE/JEPA 4x4)", flush=True)


if __name__ == "__main__":
    main()
