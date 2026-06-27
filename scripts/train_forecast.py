"""Latent forecasting on NS (L-DeepONet-style, mentor pivot Job 1).

ONE harness, two arches (single-variable swap — only the encoder differs):
  --arch fae   : OURS — frozen FAE coordinate set-latent + SetOperator
  --arch grid  : L-DeepONet baseline — frozen grid-CAE flat latent + FlatOperator
Frozen encoder/decoder, train ONLY the operator z_t -> z_{t+Δ} (all gaps), score direct + stepwise vs
persistence, against each AE's own recon-floor ceiling. Dense input now; sparse (idx=K sensors) is next.

  python scripts/train_forecast.py --arch fae  --ae_ckpt results/checkpoints/ns/fae/fae_ns128_s0.pt --smoke
  python scripts/train_forecast.py --arch grid --ae_ckpt results/checkpoints/ns/grid_ae/grid_cae_s0.pt --smoke
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.models.fae import FAE
from src.grid_ae import GridCAE
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.latent_op import SetOperator, FlatOperator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIDE = 64                                                       # encode/decode grid
DT_DIV = 8.0                                                    # normalize integer gaps off the integer-π grid
                                                               # (sin/cos(dt·f·π) aliases integer dt: 1≡3, 2≡4)


def load_fae(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE); a = ck["train_args"]
    inc = a.get("in_chans") or (3 if a.get("dataset") in ("ns", "flowbench") else 4)
    m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), depth_per_iter=4,
            num_latents=a["num_latents"], num_cross_heads=4, num_self_heads=8,
            n_freq=32, max_freq=32, coord_dim=2, in_chans=inc,
            fourier_geometric=a.get("fourier_geometric", False)).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, inc


def load_grid(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE); a = ck["train_args"]
    m = GridCAE(in_ch=a["in_chans"], side=a["side"], latent=a["latent"], ch=a["ch"]).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, a["in_chans"]


def relL2(pred, true):
    return (torch.linalg.vector_norm((pred - true).flatten(1), dim=1) /
            torch.linalg.vector_norm(true.flatten(1), dim=1).clamp_min(1e-8)).mean().item()


@torch.no_grad()
def precompute_latents(enc, ds):
    """The encoder is FROZEN, so its latents never change between epochs — encode every clip's frames
    ONCE and train the operator on the cache (kills the per-epoch re-encode; FAE then ~ matches the grid's
    per-epoch budget). Returns a CPU tensor (N_clips, T, ...latent...)."""
    Z = []
    for clip, _ in DataLoader(ds, batch_size=64):
        clip = clip.to(DEVICE); T = clip.size(2)
        Z.append(torch.stack([enc(clip[:, :, t]) for t in range(T)], 1).cpu())
    return torch.cat(Z)


def build_arch(args):
    """Return enc(field)->z, dec(z)->(B,C,SIDE,SIDE), the operator, and C — bound to the chosen arch."""
    if args.arch == "fae":
        fae, C = load_fae(args.ae_ckpt)
        coords = make_coords_2d(n_side=SIDE, device=DEVICE)
        idx = torch.arange(SIDE * SIDE, device=DEVICE)         # DENSE input (all points)
        def enc(f): return fae.encode_tokens(fields_to_tokens(f, idx), coords[idx])
        def dec(z): return fae.decoder(z, coords).permute(0, 2, 1).reshape(z.size(0), C, SIDE, SIDE)
        op = SetOperator(fae.emb_dim, depth=args.depth, heads=args.heads).to(DEVICE)
    else:
        cae, C = load_grid(args.ae_ckpt)
        def enc(f): return cae.encode(f)
        def dec(z): return cae.decode(z)
        if args.op == "deeponet":                              # faithful L-DeepONet operator (branch CNN + trunk)
            from src.deeponet import DeepONetOperator
            op = DeepONetOperator(cae.latent, p=args.p).to(DEVICE)
            with torch.no_grad():                              # init the branch LazyLinear before the optimizer sees params
                op(torch.zeros(1, cae.latent, device=DEVICE), torch.ones(1, device=DEVICE))
        else:
            op = FlatOperator(cae.latent, hidden=args.hidden, depth=args.depth).to(DEVICE)
    return enc, dec, op, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["fae", "grid"], default="fae")
    ap.add_argument("--ae_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--rollout", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)               # fae operator
    ap.add_argument("--hidden", type=int, default=512)            # grid mlp operator
    ap.add_argument("--op", choices=["deeponet", "mlp"], default="deeponet")  # grid operator (faithful DeepONet default)
    ap.add_argument("--p", type=int, default=4)                   # DeepONet basis functions
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_traj", type=int, default=8)
    ap.add_argument("--ckpt_out", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_traj, args.rollout = 2, 2, 2

    enc, dec, op, C = build_arch(args)
    op = op.to(DEVICE)
    cl = args.rollout + 1
    tr = NSDataset("train", side=SIDE, mode="clip", clip_len=cl, frame_stride=args.frame_stride, n_traj=args.n_traj)
    va = NSDataset("valid", side=SIDE, mode="clip", clip_len=cl, frame_stride=args.frame_stride,
                   n_traj=args.n_traj, stats=tr.stats)
    tl = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    print(f"=== forecast [{args.arch.upper()}] NS {C}ch res={SIDE} fstride={args.frame_stride} "
          f"rollout={args.rollout} train_clips={len(tr)}  op={sum(p.numel() for p in op.parameters())/1e6:.2f}M ===", flush=True)
    opt = torch.optim.AdamW(op.parameters(), lr=args.lr)

    @torch.no_grad()
    def evaluate():
        op.eval(); R = args.rollout
        direct = np.zeros(R); step = np.zeros(R); persist = np.zeros(R); recon = np.zeros(R); nb = 0
        for clip, _ in DataLoader(va, batch_size=64):
            clip = clip.to(DEVICE); f0 = clip[:, :, 0]; n = f0.size(0)
            z0 = enc(f0); zr = z0
            for k in range(R):
                zd = op(z0, torch.full((n,), (k + 1) / DT_DIV, device=DEVICE))    # DIRECT jump 0 -> k+1
                zr = op(zr, torch.full((n,), 1.0 / DT_DIV, device=DEVICE))        # STEPWISE rollout
                true = clip[:, :, k + 1]
                direct[k] += relL2(dec(zd), true) * n
                step[k] += relL2(dec(zr), true) * n
                persist[k] += relL2(f0, true) * n
                recon[k] += relL2(dec(enc(true)), true) * n                   # this AE's round-trip CEILING
            nb += n
        op.train(); return direct / nb, step / nb, persist / nb, recon / nb

    tcache = time.time()
    Zcache = precompute_latents(enc, tr)                              # encode ONCE (frozen) -> (N,T,...)
    ztl = DataLoader(TensorDataset(Zcache), batch_size=32, shuffle=True, drop_last=True)
    print(f"  cached {tuple(Zcache.shape)} latents in {time.time()-tcache:.0f}s (frozen encode, once)", flush=True)

    for ep in range(args.epochs):
        t0 = time.time()
        for (Z,) in ztl:
            Z = Z.to(DEVICE); n, T = Z.size(0), Z.size(1)
            loss = 0.0; npair = 0
            for i in range(T - 1):
                for j in range(i + 1, T):
                    zih = op(Z[:, i], torch.full((n,), (j - i) / DT_DIV, device=DEVICE))
                    loss = loss + F.mse_loss(zih, Z[:, j]); npair += 1
            loss = loss / npair
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == args.epochs - 1 or args.smoke:
            d, s, p, r = evaluate()
            msg = "  ".join(f"t+{k+1}:dir{d[k]:.3f}/step{s[k]:.3f}/per{p[k]:.3f}/rec{r[k]:.3f}" for k in range(args.rollout))
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  {msg}  ({time.time()-t0:.0f}s)", flush=True)

    if args.ckpt_out:
        os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
        d, s, p, r = evaluate()
        torch.save({"op": op.state_dict(), "args": vars(args), "direct": d.tolist(), "step": s.tolist(),
                    "persist": p.tolist(), "recon": r.tolist()}, args.ckpt_out)
        print(f"=== saved {args.ckpt_out}  direct={np.round(d,4).tolist()}  step={np.round(s,4).tolist()}  "
              f"recon_floor={np.round(r,4).tolist()} ===", flush=True)
    if args.smoke:
        print("=== SMOKE OK ===", flush=True)


if __name__ == "__main__":
    main()
