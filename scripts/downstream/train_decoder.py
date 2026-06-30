"""Downstream stage 1: freeze the SSL encoder, train the unified LatentDecoder on DENSE-input reconstruction.

Both FAE and MAE ingest the DENSE field (full grid) -> tokens -> decode full field. This ISOLATES the latent
(same information in; only the representation differs) and is the L-DeepONet-faithful setup (encode once frozen,
train the head). The recon decoder IS the 'reconstruct' downstream AND provides the decoder for 'forecast'.
Sparse is the EVAL axis (error vs sensor count), applied later with this head FROZEN. Encoder weights never train.
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
from src.data.preprocessed import PDEDataset
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.encoders import load_fae, load_vit, mae_ordered_tokens
from src.models.latent_decoder import LatentDecoder

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", required=True)
    ap.add_argument("--method", choices=["fae", "mae", "jepa"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dec_dim", type=int, default=384); ap.add_argument("--dec_depth", type=int, default=4)
    ap.add_argument("--dec_heads", type=int, default=6); ap.add_argument("--n_freq", type=int, default=32)
    ap.add_argument("--max_freq", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--n_query", type=int, default=2048); ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="dec")
    ap.add_argument("--ckpt_out", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = json.load(open(f"data/{args.dataset}/meta.json")); H, W, C = meta["H"], meta["W"], meta["C"]
    NPIX = H * W
    coords = make_coords_2d_hw(H, W, device=DEV); IDX = torch.arange(NPIX, device=DEV)
    a = torch.load(args.encoder_ckpt, map_location=DEV)["train_args"]
    token_dim = a.get("emb_dim") or a.get("embed_dim")
    print(f"=== train_decoder [{args.tag}] {args.method} on {args.dataset}  res={H}x{W} C={C} token_dim={token_dim} ===", flush=True)

    if args.method == "fae":
        enc, _ = load_fae(args.encoder_ckpt, DEV)
    else:
        enc, _ = load_vit(args.encoder_ckpt, DEV)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def encode_dense(x):                                     # x (B,C,H,W) -> tokens (B,N_tok,token_dim)
        if args.method == "fae":
            return enc.encode_tokens(fields_to_tokens(x, IDX), coords[IDX])
        return mae_ordered_tokens(enc, x)         # strip cls -> patch tokens

    dec = LatentDecoder(token_dim=token_dim, dec_dim=args.dec_dim, dec_depth=args.dec_depth, num_heads=args.dec_heads,
                        n_freq=args.n_freq, max_freq=args.max_freq, coord_dim=2, out_chans=C).to(DEV)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"  LatentDecoder params={sum(p.numel() for p in dec.parameters())/1e6:.2f}M (frozen encoder)", flush=True)

    tr = PDEDataset(args.dataset, "train", mode="single")
    te = PDEDataset(args.dataset, "test", mode="single")
    print(f"  train {len(tr)} test {len(te)}", flush=True)

    @torch.no_grad()                                                     # encoder is FROZEN -> encode every field ONCE, cache latents (L-DeepONet-style)
    def precompute(ds):
        Z = [encode_dense(x.to(DEV)).cpu() for x, _ in DataLoader(ds, batch_size=args.batch)]
        return torch.cat(Z)
    t0 = time.time(); Ztr = precompute(tr); Zte = precompute(te)
    print(f"  precomputed dense latents: train {tuple(Ztr.shape)} test {tuple(Zte.shape)}  ({time.time()-t0:.1f}s)", flush=True)

    def run_epoch(Z, ds, train):
        dec.train(train); tot, n = 0.0, 0
        order = torch.randperm(len(ds)) if train else torch.arange(len(ds))
        step = args.batch
        for b in range(0, len(order) - (step if train else 0) + 1, step):
            bi = order[b:b + step]
            if len(bi) == 0: break
            tok = Z[bi].to(DEV)
            x = torch.stack([ds[int(i)][0] for i in bi]).to(DEV); B = x.size(0)
            qi = torch.randperm(NPIX, device=DEV)[:args.n_query]
            tgt = x.reshape(B, C, NPIX)[:, :, qi].permute(0, 2, 1)        # (B, n_query, C)
            with torch.set_grad_enabled(train):
                pred = dec(tok, coords[qi]); loss = ((pred - tgt) ** 2).mean()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * B; n += B
        return tot / max(n, 1)

    for ep in range(args.epochs):
        t = time.time()
        tl = run_epoch(Ztr, tr, True)
        if (ep + 1) % args.eval_every == 0 or ep == 0:
            with torch.no_grad():
                vl = run_epoch(Zte, te, False)
            print(f"ep {ep+1:3d}/{args.epochs}  train_mse={tl:.4e}  test_mse={vl:.4e}  ({time.time()-t:.1f}s)", flush=True)

    out = args.ckpt_out or f"results/checkpoints/{args.dataset}/decoder/{args.tag}.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": dec.state_dict(), "train_args": vars(args), "token_dim": token_dim}, out)
    print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
