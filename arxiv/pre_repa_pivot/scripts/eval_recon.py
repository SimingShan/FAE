"""Recon demo on a DEVELOPED (late) frame, + present-vs-future sanity for ours:
when we ask to reconstruct x_t we must output x_t, NOT x_{t+Δ}. Also observation-
invariance (two sparse views of the same frame). Saves figures to results/plots/."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scripts.train_baseline as tb
from src.models import FAE
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; tb.DEVICE = DEVICE
os.makedirs("results/plots", exist_ok=True)
CLIP_LEN, STRIDE, T_IDX, DGAP = 16, 4, 2, 13     # big gap (13 strided=52 raw frames) for real divergence


def load_ours():
    ck = torch.load(f"results/checkpoints/g1/faep_twoview_tvb_s0.pt", map_location="cpu")
    ta = ck["train_args"]; R = ta["resolution"]
    m = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128, num_cross_heads=4,
            num_self_heads=8, n_freq=32, max_freq=32, coord_dim=2, in_chans=4).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=STRIDE, clip_len=CLIP_LEN, side=R, stats=ck["stats"])
    return m, va, R


@torch.no_grad()
def decode_full(m, tok, coords, R):
    return m.decoder(tok, coords)[0].permute(1, 0).reshape(4, R, R).cpu()


def main():
    m, va, R = load_ours(); NPIX = R * R
    coords = make_coords_2d(R, DEVICE)
    # pick the MOST diverged clip: max field change between frame T_IDX and T_IDX+DGAP
    best, bidx = -1.0, 0
    for i in range(0, len(va), max(1, len(va) // 40)):
        c = va[i][0]
        d = float(((c[:, T_IDX] - c[:, T_IDX + DGAP]) ** 2).mean())
        if d > best: best, bidx = d, i
    clip = va[bidx][0].to(DEVICE)
    print(f"  picked clip {bidx} (frame-gap MSE={best:.4f})")
    xt = clip[:, T_IDX].unsqueeze(0); xtd = clip[:, T_IDX + DGAP].unsqueeze(0)   # (1,4,R,R)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    with torch.no_grad():
        iA = torch.randperm(NPIX, generator=g, device=DEVICE)[:256]
        iB = torch.randperm(NPIX, generator=g, device=DEVICE)[:512]
        tA = m.encode_tokens(fields_to_tokens(xt, iA), coords[iA])      # encode frame t,  view A
        tB = m.encode_tokens(fields_to_tokens(xt, iB), coords[iB])      # encode frame t,  view B (invariance)
        tD = m.encode_tokens(fields_to_tokens(xtd, iA), coords[iA])     # encode frame t+Δ
        rec_t = decode_full(m, tA, coords, R)                          # reconstruct asked frame t
        rec_td = decode_full(m, tD, coords, R)                         # reconstruct asked frame t+Δ
        cos = float(F.cosine_similarity(m.represent(tA), m.represent(tB), dim=-1))
    xt_c, xtd_c = xt[0].cpu(), xtd[0].cpu()
    mse = lambda a, b: float(((a - b) ** 2).mean())
    print(f"\nframe gap MSE(x_t, x_t+Δ) = {mse(xt_c, xtd_c):.4f}   (>0 => the two frames really differ)")
    print("\n===== SANITY: reconstruct-t must match x_t, NOT x_t+Δ (rows=asked frame, cols=truth) =====")
    print(f"                      vs x_t      vs x_t+Δ")
    print(f"  reconstruct x_t :   {mse(rec_t, xt_c):.4f}     {mse(rec_t, xtd_c):.4f}   (LOW on x_t, HIGH on x_t+Δ)")
    print(f"  reconstruct x_t+Δ:  {mse(rec_td, xt_c):.4f}     {mse(rec_td, xtd_c):.4f}   (HIGH on x_t, LOW on x_t+Δ)")
    print(f"\n  observation-invariance: cosine(rep viewA, viewB) = {cos:.4f}")

    ch = 0
    f, ax = plt.subplots(2, 3, figsize=(10.5, 7))
    for a in ax.ravel(): a.axis("off")
    P = lambda a, im, t: (a.imshow(im, cmap="RdBu_r"), a.set_title(t, fontsize=9))
    P(ax[0, 0], xt_c[ch], f"truth x_t (frame {T_IDX})")
    P(ax[0, 1], rec_t[ch], f"ours recon x_t\nMSE={mse(rec_t,xt_c):.3f}")
    P(ax[0, 2], (rec_t[ch] - xt_c[ch]), "x_t recon error")
    P(ax[1, 0], xtd_c[ch], f"truth x_t+Δ (frame {T_IDX+DGAP})")
    P(ax[1, 1], rec_td[ch], f"ours recon x_t+Δ\nMSE={mse(rec_td,xtd_c):.3f}")
    P(ax[1, 2], (xtd_c[ch] - xt_c[ch]), "x_t+Δ - x_t (the change)")
    f.suptitle("ours — reconstruct t and t+Δ (developed frame); each matches its own frame", fontsize=11)
    f.tight_layout()
    f.savefig("results/plots/recon_present_future.png", dpi=110); print("\n  saved results/plots/recon_present_future.png")


if __name__ == "__main__":
    main()
