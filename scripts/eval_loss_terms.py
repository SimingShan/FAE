"""Per-view / per-term reconstruction breakdown for twoview, computed from a saved ckpt
(no retraining). Four terms: view A {present x_t, future x_{t+Δ}} and view B {present, future},
where A=512-sensor view, B=256-sensor view, sharing the SAME query targets. Mirrors the
training loss in train_fae_predict.py exactly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models import FAE
from src.models.fjepa import TokenPredictor
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens
DEVICE = "cpu"
CKPT = "results/checkpoints/g1/faep_twoview_tvp_s0.pt"


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["train_args"]; R = a["resolution"]; NPIX = R * R
    fae = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128, num_cross_heads=4,
              num_self_heads=8, n_freq=32, max_freq=32, coord_dim=2, in_chans=4)
    fae.load_state_dict(ck["model"]); fae.eval()
    pred = TokenPredictor(320, depth=a["pred_depth"], heads=8)
    pred.load_state_dict(ck["predictor"]); pred.eval()
    clip_len = max(a["dt_max"], a["dt_fixed"]) + 1
    va = ShearFlowClipDataset("valid", n_seed=4, frame_stride=a["frame_stride"], clip_len=clip_len, side=R, stats=ck["stats"])
    coords = make_coords_2d(R, DEVICE)
    g = torch.Generator().manual_seed(0)
    agg = {k: 0.0 for k in ["A_present", "A_future", "B_present", "B_future"]}; n = 0
    for clip, _ in DataLoader(va, batch_size=32, shuffle=False):
        B = clip.size(0); K = clip.size(2); bidx = torch.arange(B)
        delta = torch.full((B,), a["dt_fixed"]) if a["dt_fixed"] > 0 else torch.randint(1, a["dt_max"] + 1, (B,), generator=g)
        ts = (torch.rand(B, generator=g) * (K - delta).float()).long()
        fa = clip[bidx, :, ts]; fb = clip[bidx, :, ts + delta]; dt = delta.float() / a["dt_max"]
        iA = torch.randperm(NPIX, generator=g)[:512]; iB = torch.randperm(NPIX, generator=g)[:256]
        iq = torch.randperm(NPIX, generator=g)[:a["n_query"]]
        tgt_t = fields_to_tokens(fa, iq); tgt_f = fields_to_tokens(fb, iq)
        with torch.no_grad():
            tA = fae.encode_tokens(fields_to_tokens(fa, iA), coords[iA])
            tB = fae.encode_tokens(fields_to_tokens(fa, iB), coords[iB])
            agg["A_present"] += F.mse_loss(fae.decoder(tA, coords[iq]), tgt_t).item() * B
            agg["A_future"] += F.mse_loss(fae.decoder(pred(tA, dt), coords[iq]), tgt_f).item() * B
            agg["B_present"] += F.mse_loss(fae.decoder(tB, coords[iq]), tgt_t).item() * B
            agg["B_future"] += F.mse_loss(fae.decoder(pred(tB, dt), coords[iq]), tgt_f).item() * B
        n += B
        if n >= 300: break
    print(f"twoview per-term recon MSE  (n={n} val frames; viewA=512 sensors, viewB=256; Δ={a['dt_fixed'] or '1..'+str(a['dt_max'])})")
    for k in ["A_present", "A_future", "B_present", "B_future"]:
        print(f"  {k:12s} {agg[k]/n:.4f}")
    print(f"  --> view-A total {(agg['A_present']+agg['A_future'])/n:.4f} | view-B total {(agg['B_present']+agg['B_future'])/n:.4f}")
    print(f"  --> present total {(agg['A_present']+agg['B_present'])/n:.4f} | future total {(agg['A_future']+agg['B_future'])/n:.4f}")


if __name__ == "__main__":
    main()
