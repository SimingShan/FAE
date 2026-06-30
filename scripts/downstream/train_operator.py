"""Downstream stage 2: freeze the SSL encoder, train the LATENT OPERATOR (TokenPredictor) on the frozen latents.

z_t -> operator(dt) -> ẑ_{t+dt}, trained on LATENT-MSE (L-DeepONet-style: learn the operator IN the latent space;
encoder frozen, decoder not needed here). dt ~ Uniform{1..dt_max} per sample, normalized to [0,1]. Dense input
(full grid) so the latent matches the decoder's training distribution; sparse is the EVAL axis (eval_forecast).
Precomputes per-(traj,frame) dense latents ONCE (frozen). Field-space error is measured later by decoding ẑ.
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, numpy as np
from src.data.preprocessed import PDEDataset
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.encoders import load_fae, load_vit, mae_ordered_tokens
from src.models.fae import TokenPredictor

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", required=True)
    ap.add_argument("--method", choices=["fae", "mae", "jepa"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dt_max", type=int, default=4, help="train horizons Uniform{1..dt_max} (in saved-frame units)")
    ap.add_argument("--pred_depth", type=int, default=4); ap.add_argument("--pred_heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--steps_per_epoch", type=int, default=0, help="0 = (n_traj*T)//batch")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="op")
    ap.add_argument("--ckpt_out", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = json.load(open(f"data/{args.dataset}/meta.json")); H, W, C = meta["H"], meta["W"], meta["C"]
    coords = make_coords_2d_hw(H, W, device=DEV); IDX = torch.arange(H * W, device=DEV)
    a = torch.load(args.encoder_ckpt, map_location=DEV)["train_args"]
    token_dim = a.get("emb_dim") or a.get("embed_dim")
    print(f"=== train_operator [{args.tag}] {args.method} on {args.dataset}  dt_max={args.dt_max} token_dim={token_dim} ===", flush=True)

    enc, _ = (load_fae if args.method == "fae" else load_vit)(args.encoder_ckpt, DEV)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def encode_dense(x):                                     # x (B,C,H,W) -> (B,N_tok,token_dim)
        if args.method == "fae":
            return enc.encode_tokens(fields_to_tokens(x, IDX), coords[IDX])
        return mae_ordered_tokens(enc, x)

    from torch.utils.data import DataLoader

    @torch.no_grad()                                         # cache per-(traj,frame) dense latents ONCE (batched; index is traj-major n*T+t -> reshape)
    def precompute(split):
        ds = PDEDataset(args.dataset, split, mode="single", start_stride=1)
        Ntr, T = ds.fields.shape[:2]
        Z = torch.cat([encode_dense(x.to(DEV)).cpu() for x, _ in DataLoader(ds, batch_size=32)])
        return Z.reshape(Ntr, T, Z.shape[1], Z.shape[2]), T

    t0 = time.time(); Ztr, T = precompute("train"); Zte, _ = precompute("test")
    assert T > args.dt_max, f"T={T} must exceed dt_max={args.dt_max}"
    print(f"  cached latents: train {tuple(Ztr.shape)} test {tuple(Zte.shape)}  ({time.time()-t0:.1f}s)", flush=True)

    op = TokenPredictor(token_dim, depth=args.pred_depth, heads=args.pred_heads).to(DEV)
    opt = torch.optim.AdamW(op.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    spe = args.steps_per_epoch or max(1, (Ztr.shape[0] * T) // args.batch)
    print(f"  TokenPredictor params={sum(p.numel() for p in op.parameters())/1e6:.2f}M  steps/epoch={spe}", flush=True)

    def sample(Z, T, B):
        Ntr = Z.shape[0]
        dt = torch.randint(1, args.dt_max + 1, (B,))
        t = torch.randint(0, T - args.dt_max, (B,))                                            # t+dt < T guaranteed
        tr = torch.randint(0, Ntr, (B,))
        return Z[tr, t].to(DEV), Z[tr, t + dt].to(DEV), (dt.float() / args.dt_max).to(DEV)

    def rel_err(Z, T):                                       # mean relative latent error over a fixed eval draw
        op.eval()
        with torch.no_grad():
            zt, ztd, dtn = sample(Z, T, min(512, Z.shape[0] * 2))
            pred = op(zt, dtn)
            return (((pred - ztd) ** 2).sum((-1, -2)).sqrt() / (ztd ** 2).sum((-1, -2)).sqrt().clamp_min(1e-6)).mean().item()

    for ep in range(args.epochs):
        op.train(); te = time.time(); tot = 0.0
        for _ in range(spe):
            zt, ztd, dtn = sample(Ztr, T, args.batch)
            pred = op(zt, dtn)
            loss = ((pred - ztd) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % args.eval_every == 0 or ep == 0:
            print(f"ep {ep+1:3d}/{args.epochs}  train_mse={tot/spe:.4e}  test_relL2(latent)={rel_err(Zte, T):.4f}  ({time.time()-te:.1f}s)", flush=True)

    out = args.ckpt_out or f"results/checkpoints/{args.dataset}/operator/{args.tag}.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": op.state_dict(), "train_args": vars(args), "token_dim": token_dim}, out)
    print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
