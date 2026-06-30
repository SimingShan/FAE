"""Reconstruction-based SSL pretraining. The frozen encoder is the product (decoder/predictor
discarded at eval). Four objective cells:
  recon            encode sparse x_t -> decode x_t.                       (Senseiver)
  recon_both       + predictor(dt) -> decode x_{t+dt}.                    (+temporal)
  twoview_present  recon + a 2nd sparsity view sharing the x_t target.    (+dual-view)
  twoview          dual-view + temporal.                                  (full FAE, default)
Eval: frozen encoder -> mean[/+std] readout -> ridge probe + participation ratio.
"""
import sys, os, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models import FAE
from src.models.fae import TokenPredictor
from src.data.well2d import ShearFlowClipDataset, make_coords_2d_hw, make_coords_3d, fields_to_tokens
from src.metrics import lin_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARAMS = ["logRe", "Sc"]


def participation_ratio(Z):
    Z = Z - Z.mean(0); e = np.clip(np.linalg.eigvalsh(Z.T @ Z / max(len(Z) - 1, 1)), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-30))


@torch.no_grad()
def embed(model, ds, coords, idx, batch=128):
    """Frozen encoder on frame 0 -> (mean+std, mean) readouts."""
    model.eval(); Zms, Zm, Y = [], [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        tok = model.encode_tokens(fields_to_tokens(clip[:, :, 0].to(DEVICE), idx), coords[idx])
        Zms.append(torch.cat([tok.mean(1), tok.std(1)], -1).cpu().numpy())
        Zm.append(tok.mean(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Zms), np.concatenate(Zm), np.concatenate(Y)


def probe2(Ztr, Ytr, Zva, Yva):
    out = {}
    for j, nm in enumerate(PARAMS):
        yt, yv = Ytr[:, j], Yva[:, j]; m, s = yt.mean(), yt.std() + 1e-8
        out[nm] = lin_probe(Ztr, (yt - m) / s, Zva, (yv - m) / s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="twoview", choices=["recon", "recon_both", "twoview", "twoview_present"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    ap.add_argument("--dt_max", type=int, default=0, help="0 = use meta dt_max (decorrelation-set); >0 overrides")
    ap.add_argument("--dt_fixed", type=int, default=0, help="0 = random dt in [1,dt_max]; >0 = fixed horizon")
    ap.add_argument("--mcnt", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--mcnt_range", type=int, nargs=2, default=None, help="if set, sample sensor count in [lo,hi] per view")
    ap.add_argument("--sensor_pattern", default="discrete", choices=["discrete", "continuous"])
    ap.add_argument("--n_query", type=int, default=1024)
    ap.add_argument("--query_mode", choices=["global", "neighborhood"], default="global",
                    help="decoder target geometry: 'global' = uniform full-field points (default); "
                         "'neighborhood' = points drawn from disks around the input sensors (point-to-neighborhood)")
    ap.add_argument("--query_radius", type=int, default=8, help="neighborhood half-width in pixels (query_mode=neighborhood)")
    ap.add_argument("--pred_depth", type=int, default=2)
    ap.add_argument("--predictor", choices=["attn", "linear"], default="attn",
                    help="latent predictor: 'attn' = self-attention TokenPredictor (default); 'linear' = Koopman-style dt-conditioned linear operator (bottleneck)")
    ap.add_argument("--pred_n_basis", type=int, default=4, help="linear predictor: # dt-modulated generator matrices")
    ap.add_argument("--num_latents", type=int, default=128)
    ap.add_argument("--num_iter", type=int, default=4)
    ap.add_argument("--depth_per_iter", type=int, default=5)
    ap.add_argument("--num_cross_heads", type=int, default=4)
    ap.add_argument("--num_self_heads", type=int, default=8)
    ap.add_argument("--n_freq", type=int, default=32)
    ap.add_argument("--max_freq", type=int, default=32)
    ap.add_argument("--emb_dim", type=int, default=320)
    ap.add_argument("--val_dim", type=int, default=32, help="token dims for the sensor VALUE (rest = coordinate); default 32 of emb_dim")
    ap.add_argument("--use_local", action=argparse.BooleanOptionalAction, default=False,
                    help="mini-PointNet local-neighbourhood token embedding before the Perceiver cross-attn")
    ap.add_argument("--local_k", type=int, default=8, help="k nearest neighbours per token (local embed)")
    ap.add_argument("--local_dim", type=int, default=48, help="local feature dim (stolen from coord-proj budget)")
    ap.add_argument("--n_seed", type=int, default=24)
    ap.add_argument("--n_traj", type=int, default=12, help="NS trajectories/file")
    ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--res_h", type=int, default=None, help="non-square FAE height (shear native 128x256)")
    ap.add_argument("--res_w", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="faep")
    ap.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ckpt_out", default=None)
    ap.add_argument("--dataset", choices=["shear", "flowbench", "ns", "typhoon", "sw", "mhd", "rbc"], default="shear")
    ap.add_argument("--in_chans", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=20, help="probe cadence (epochs)")
    args = ap.parse_args()
    from src.utils import set_seed
    set_seed(args.seed)
    import json
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    meta = json.load(open(os.path.join(REPO, "data", args.dataset, "meta.json")))   # preprocessed spec
    RH, RW, in_chans = meta["H"], meta["W"], meta["C"]; NPIX = RH * RW; R = RH; vol = False
    PARAMS[:] = meta["label_names"]
    if args.dt_max <= 0: args.dt_max = meta["dt_max"]           # principled dt_max from decorrelation
    temporal = args.mode in ("recon_both", "twoview")          # predictor + future frame
    dual = args.mode in ("twoview", "twoview_present")          # 2nd sparsity view (observation-invariance)
    clip_len = (max(args.dt_max, args.dt_fixed) + 1) if temporal else 2
    qdesc = f"{args.query_mode}" + (f"(r={args.query_radius})" if args.query_mode == "neighborhood" else "")
    print(f"=== FAE-{args.mode} [{args.tag}] {args.dataset} dt_max={args.dt_max} mcnt={args.mcnt_range or args.mcnt} "
          f"n_query={args.n_query} query={qdesc} res={RH}x{RW} ===", flush=True)

    from src.data.preprocessed import PDEDataset
    tr = PDEDataset(args.dataset, "train", mode="clip", clip_len=clip_len)
    va = PDEDataset(args.dataset, "test", mode="clip", clip_len=clip_len)
    print(f"  dataset={args.dataset} in_chans={in_chans}  train {len(tr)} test {len(va)}", flush=True)
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    coords = make_coords_3d(R, R, device=DEVICE) if vol else make_coords_2d_hw(RH, RW, device=DEVICE)

    model = FAE(emb_dim=args.emb_dim, num_iter=args.num_iter, depth_per_iter=args.depth_per_iter,
                num_latents=args.num_latents, num_cross_heads=args.num_cross_heads, num_self_heads=args.num_self_heads,
                n_freq=args.n_freq, max_freq=args.max_freq, val_dim=args.val_dim, coord_dim=3 if vol else 2, in_chans=in_chans,
                use_local=args.use_local, local_k=args.local_k, local_dim=args.local_dim).to(DEVICE)
    print(f"  num_latents={args.num_latents} num_iter={args.num_iter} use_local={args.use_local} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    if not temporal:
        predictor = None
    elif args.predictor == "linear":
        from src.models.fae import LinearTokenPredictor
        predictor = LinearTokenPredictor(args.emb_dim, n_basis=args.pred_n_basis).to(DEVICE)
    else:
        predictor = TokenPredictor(args.emb_dim, depth=args.pred_depth, heads=8).to(DEVICE)
    if predictor is not None:
        print(f"  predictor={args.predictor}" + (f"(n_basis={args.pred_n_basis})" if args.predictor == "linear" else f"(depth={args.pred_depth})")
              + f"  pred_params={sum(p.numel() for p in predictor.parameters())/1e3:.0f}K", flush=True)
    params = list(model.parameters()) + (list(predictor.parameters()) if predictor else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas))
    warm = max(1, int(args.warmup_frac * args.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warm if ep < warm else 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, args.epochs - warm))))
    probe_idx = torch.randperm(NPIX, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)[:1024]

    def sensor_idx(n):
        """n input-sensor flat indices: 'discrete' random scatter, or 'continuous' contiguous block."""
        if args.sensor_pattern == "continuous":
            s = min(R, int(math.ceil(n ** 0.5)))
            top = int(torch.randint(0, R - s + 1, (1,)).item()); left = int(torch.randint(0, R - s + 1, (1,)).item())
            rows = (top + torch.arange(s, device=DEVICE)).view(s, 1); cols = (left + torch.arange(s, device=DEVICE)).view(1, s)
            return (rows * R + cols).flatten()[:n]
        return torch.randperm(NPIX, device=DEVICE)[:n]

    def n_sensors():
        return np.random.randint(args.mcnt_range[0], args.mcnt_range[1] + 1) if args.mcnt_range else int(np.random.choice(args.mcnt))

    # neighborhood = DENSE local patch: filled-disk offset template, reused around each centre sensor
    _o = torch.arange(-args.query_radius, args.query_radius + 1, device=DEVICE)
    _dg = torch.stack(torch.meshgrid(_o, _o, indexing="ij"), -1).reshape(-1, 2)
    DISK = _dg[(_dg ** 2).sum(-1) <= args.query_radius ** 2]                  # (D,2) filled disk
    N_CENTERS = max(1, args.n_query // DISK.size(0))                          # so C*D ~= n_query budget

    def query_idx(iA):
        """Decoder targets. 'global' = uniform full field; 'neighborhood' = DENSE filled disks (radius
        query_radius) around N_CENTERS input sensors — reconstruct each local PATCH in full. Centres are
        restricted to the INTERIOR so every disk is fully in-bounds (no boundary clamping / edge over-weighting)."""
        if args.query_mode == "global":
            return torch.randperm(NPIX, device=DEVICE)[:args.n_query]
        r, c = iA // RW, iA % RW; rad = args.query_radius
        inb = (r >= rad) & (r < RH - rad) & (c >= rad) & (c < RW - rad)       # sensors whose full disk fits
        pool = iA[inb] if inb.any() else iA
        ctr = pool[torch.randperm(pool.numel(), device=DEVICE)[:N_CENTERS]]   # distinct centre sensors
        cr, cc = (ctr // RW)[:, None], (ctr % RW)[:, None]                    # (C,1)
        qr = (cr + DISK[:, 0][None]).clamp(0, RH - 1)                         # (C,D); clamp = no-op safety (interior)
        qc = (cc + DISK[:, 1][None]).clamp(0, RW - 1)
        return (qr * RW + qc).reshape(-1)                                    # (C*D,) all neighbourhood points

    t0 = time.time()
    for ep in range(args.epochs):
        te = time.time(); model.train(); ag = {"rec": 0.0, "n": 0}
        for clip, _y in loader:
            clip = clip.to(DEVICE, non_blocking=True); B, K = clip.size(0), clip.size(2)
            if temporal:
                delta = torch.full((B,), args.dt_fixed, device=DEVICE) if args.dt_fixed > 0 else torch.randint(1, args.dt_max + 1, (B,), device=DEVICE)
                ts = (torch.rand(B, device=DEVICE) * (K - delta).float()).long(); bidx = torch.arange(B, device=DEVICE)
                fa, fb, dt = clip[bidx, :, ts], clip[bidx, :, ts + delta], delta.float() / args.dt_max
            else:
                fa, fb, dt = clip[:, :, 0], clip[:, :, 0], None
            iA = sensor_idx(n_sensors()); tA = model.encode_tokens(fields_to_tokens(fa, iA), coords[iA])
            iq = query_idx(iA)                                              # global OR neighborhood-of-sensors
            tgt_t, tgt_f = fields_to_tokens(fa, iq), fields_to_tokens(fb, iq)
            loss = F.mse_loss(model.decoder(tA, coords[iq]), tgt_t)          # present recon of view A
            if temporal:                                                    # + future recon (non-collapsible)
                loss = loss + F.mse_loss(model.decoder(predictor(tA, dt), coords[iq]), tgt_f)
            if dual:                                                        # 2nd view, shared targets (invariance)
                iB = sensor_idx(n_sensors()); tB = model.encode_tokens(fields_to_tokens(fa, iB), coords[iB])
                loss = loss + F.mse_loss(model.decoder(tB, coords[iq]), tgt_t)
                if temporal:
                    loss = loss + F.mse_loss(model.decoder(predictor(tB, dt), coords[iq]), tgt_f)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            ag["rec"] += float(loss) * B; ag["n"] += B
        sched.step()
        print(f"ep {ep+1:3d}/{args.epochs} train={time.time()-te:.1f}s", flush=True)   # pure train-epoch (no probe)
        if (ep + 1) % args.eval_every == 0 or ep == 0:
            Zms, Zm, Ytr = embed(model, tr, coords, probe_idx); Vms, Vm, Yva = embed(model, va, coords, probe_idx)
            pr = participation_ratio(Zms); pb = probe2(Zms, Ytr, Vms, Yva); pm = probe2(Zm, Ytr, Vm, Yva)
            psms = " ".join(f"{k}={pb[k]:+.3f}" for k in PARAMS); psm = " ".join(f"{k}={pm[k]:+.3f}" for k in PARAMS)
            print(f"ep {ep+1:3d}/{args.epochs}  rec={ag['rec']/ag['n']:.3e}  PR={pr:.1f}  "
                  f"mean+std {psms} | mean {psm}  ({time.time()-t0:.0f}s)", flush=True)
    if args.save:
        out = args.ckpt_out or f"results/checkpoints/g1/faep_{args.mode}_{args.tag}.pt"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ckpt = {"model": model.state_dict(), "stats": tr.stats, "train_args": vars(args)}
        if predictor is not None: ckpt["predictor"] = predictor.state_dict()
        torch.save(ckpt, out)
        print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
