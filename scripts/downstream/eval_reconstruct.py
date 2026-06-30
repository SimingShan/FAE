"""Sparse RECONSTRUCT (the sparse-regime headline): frozen encoder + trained recon decoder reconstruct the
FULL field from k sparse sensors. Reports field relL2 vs SENSOR COUNT. The ONLY difference from dense recon is
input handling: FAE encodes the k (coord,value) points DIRECTLY; MAE can't, so k -> interpolate-to-grid -> encode
(architecture-axis baseline). No operator — just encode(sparse) -> decode. Decoder is the dense-trained recon head.
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, numpy as np
from scipy.interpolate import griddata
from src.data.preprocessed import PDEDataset
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.encoders import load_fae, load_vit, mae_ordered_tokens
from src.models.latent_decoder import LatentDecoder

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", required=True); ap.add_argument("--method", choices=["fae", "mae", "jepa"], required=True)
    ap.add_argument("--decoder_ckpt", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--sensors", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 0], help="0 = dense full grid")
    ap.add_argument("--max_traj", type=int, default=64); ap.add_argument("--frames_per_traj", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = json.load(open(f"data/{args.dataset}/meta.json")); H, W, C = meta["H"], meta["W"], meta["C"]
    NPIX = H * W
    coords = make_coords_2d_hw(H, W, device=DEV); IDX = torch.arange(NPIX, device=DEV)
    grid_xy = coords.cpu().numpy()                                          # (NPIX, 2) for griddata targets

    enc, _ = (load_fae if args.method == "fae" else load_vit)(args.encoder_ckpt, DEV); enc.eval()
    dc = torch.load(args.decoder_ckpt, map_location=DEV)
    dec = LatentDecoder(token_dim=dc["token_dim"], num_heads=dc["train_args"].get("dec_heads", 6), coord_dim=2, out_chans=C,
                        **{k: v for k, v in dc["train_args"].items() if k in ("dec_dim", "dec_depth", "n_freq", "max_freq")}).to(DEV)
    dec.load_state_dict(dc["model"]); dec.eval()
    print(f"=== eval_reconstruct {args.method} on {args.dataset}  (sparse-regime: relL2 vs #sensors) ===", flush=True)

    @torch.no_grad()
    def encode(x, k):                                                       # x (B,C,H,W)
        sidx = IDX if k == 0 else torch.randperm(NPIX, device=DEV)[:k]
        if args.method == "fae":
            return enc.encode_tokens(fields_to_tokens(x, sidx), coords[sidx])
        if k == 0:
            return mae_ordered_tokens(enc, x)
        xg = []                                                             # MAE: interpolate k sparse pts -> grid (nearest), then encode
        pts = grid_xy[sidx.cpu().numpy()]; flat = x.reshape(x.size(0), C, NPIX).cpu().numpy()
        for b in range(x.size(0)):
            ch = [griddata(pts, flat[b, c, sidx.cpu().numpy()], grid_xy, method="nearest") for c in range(C)]
            xg.append(np.stack(ch).reshape(C, H, W))
        return mae_ordered_tokens(enc, torch.from_numpy(np.stack(xg)).float().to(DEV))

    @torch.no_grad()
    def relL2(pred, tgt):
        return (((pred - tgt) ** 2).sum((1, 2, 3)).sqrt() / (tgt ** 2).sum((1, 2, 3)).sqrt().clamp_min(1e-6)).mean().item()

    ds = PDEDataset(args.dataset, "test", mode="single"); fields, mean, std = ds.fields, ds.mean, ds.std
    Ntr, T = fields.shape[:2]; ntj = min(args.max_traj, Ntr)
    ts = list(range(0, T, max(1, T // args.frames_per_traj)))[:args.frames_per_traj]
    X = torch.stack([torch.from_numpy((np.asarray(fields[tr, t], np.float32) - mean) / std)
                     for tr in range(ntj) for t in ts]).to(DEV)            # (ntj*frames, C, H, W)
    print(f"  {'#sensors':>9}  relL2", flush=True)
    for k in args.sensors:
        errs = []
        with torch.no_grad():
            for b in range(0, X.size(0), 16):
                xb = X[b:b + 16]; z = encode(xb, k)
                pred = dec(z, coords[IDX]).permute(0, 2, 1).reshape(-1, C, H, W)
                errs.append(relL2(pred, xb))
        print(f"  {('dense' if k==0 else k):>9}  {np.mean(errs):.4f}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
