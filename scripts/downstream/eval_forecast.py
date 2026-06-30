"""Downstream eval (capstone): frozen encoder -> latent operator -> recon decoder, error in FIELD space.

Rolls the dt-conditioned operator DIRECTLY to each horizon (z_{t+dt} = op(z_t, dt)), decodes the full grid,
and reports relative-L2 vs the true field — as a function of HORIZON and of input SENSOR COUNT (the sparse-regime
x-axis). FAE encodes sparse points DIRECTLY; MAE/JEPA can't, so sparse -> interpolate-to-grid -> encode
(the architecture-axis baseline). Dense (full-grid input) is the clean operator comparison; sweep down for sparse.
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, numpy as np
from src.data.preprocessed import PDEDataset
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.encoders import load_fae, load_vit, mae_ordered_tokens
from src.models.latent_decoder import LatentDecoder
from src.models.fae import TokenPredictor

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", required=True); ap.add_argument("--method", choices=["fae", "mae", "jepa"], required=True)
    ap.add_argument("--decoder_ckpt", required=True); ap.add_argument("--operator_ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--sensors", type=int, nargs="+", default=[0], help="input sensor counts; 0 = dense full grid")
    ap.add_argument("--max_traj", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = json.load(open(f"data/{args.dataset}/meta.json")); H, W, C = meta["H"], meta["W"], meta["C"]
    NPIX = H * W; dt_max_data = meta["dt_max"]
    coords = make_coords_2d_hw(H, W, device=DEV); IDX = torch.arange(NPIX, device=DEV)

    enc, _ = (load_fae if args.method == "fae" else load_vit)(args.encoder_ckpt, DEV); enc.eval()
    dc = torch.load(args.decoder_ckpt, map_location=DEV); dec = LatentDecoder(token_dim=dc["token_dim"], **{
        k: v for k, v in dc["train_args"].items() if k in ("dec_dim", "dec_depth", "n_freq", "max_freq")},
        num_heads=dc["train_args"].get("dec_heads", 6), coord_dim=2, out_chans=C).to(DEV)
    dec.load_state_dict(dc["model"]); dec.eval()
    oc = torch.load(args.operator_ckpt, map_location=DEV); op_dtmax = oc["train_args"]["dt_max"]
    op = TokenPredictor(oc["token_dim"], depth=oc["train_args"]["pred_depth"], heads=oc["train_args"].get("pred_heads", 8)).to(DEV)
    op.load_state_dict(oc["model"]); op.eval()
    print(f"=== eval_forecast {args.method} on {args.dataset}  (operator dt_max={op_dtmax}) ===", flush=True)

    @torch.no_grad()
    def encode(x, k):                                       # x (B,C,H,W); k=0 dense full grid, else k sparse sensors
        if k == 0:
            sidx = IDX
        else:
            sidx = torch.randperm(NPIX, device=DEV)[:k]
        if args.method == "fae":
            if k == 0 or True:                              # FAE ingests the chosen sensor set directly
                return enc.encode_tokens(fields_to_tokens(x, sidx), coords[sidx])
        # MAE/JEPA: must see a grid -> scatter sparse onto grid + nearest-fill (architecture-axis baseline)
        if k == 0:
            xg = x
        else:
            flat = x.reshape(x.size(0), C, NPIX)
            xg = torch.zeros_like(flat); cnt = torch.zeros(1, 1, NPIX, device=DEV)
            xg[:, :, sidx] = flat[:, :, sidx]; cnt[:, :, sidx] = 1.0
            xg = xg.reshape(x.size(0), C, H, W)             # zero-filled (simple baseline; griddata-interp is a TODO upgrade)
        return mae_ordered_tokens(enc, xg)

    ds = PDEDataset(args.dataset, "test", mode="single")
    fields, mean, std = ds.fields, ds.mean, ds.std
    Ntr, T = fields.shape[:2]; ntj = min(args.max_traj, Ntr)

    @torch.no_grad()
    def relL2(pred, tgt):                                   # (B,C,H,W) -> scalar
        return (((pred - tgt) ** 2).sum((1, 2, 3)).sqrt() / (tgt ** 2).sum((1, 2, 3)).sqrt().clamp_min(1e-6)).mean().item()

    print(f"  {'sensors':>8} " + "".join(f"  h={h:<5}" for h in args.horizons), flush=True)
    for k in args.sensors:
        row = []
        for dt in args.horizons:
            errs = []
            with torch.no_grad():
                for tr in range(ntj):
                    ts = range(0, T - dt, max(1, (T - dt) // 4))            # a few start frames / traj
                    x0 = torch.stack([torch.from_numpy((np.asarray(fields[tr, t], np.float32) - mean) / std) for t in ts]).to(DEV)
                    xt = torch.stack([torch.from_numpy((np.asarray(fields[tr, t + dt], np.float32) - mean) / std) for t in ts]).to(DEV)
                    z0 = encode(x0, k)
                    zt = op(z0, torch.full((z0.size(0),), dt / op_dtmax, device=DEV))   # direct dt-conditioned step
                    pred = dec(zt, coords[IDX]).permute(0, 2, 1).reshape(-1, C, H, W)
                    errs.append(relL2(pred, xt))
            row.append(np.mean(errs))
        tag = "dense" if k == 0 else f"{k}"
        print(f"  {tag:>8} " + "".join(f"  {e:6.4f}" for e in row), flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
