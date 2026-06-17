"""Reconstruction-based SSL on shear_flow — the task-AGNOSTIC objective (recover the
field, knows nothing about Re/Sc). Encoder is the product; decoder/predictor discarded
at eval. Frozen encoder -> mean+std readout -> linear probe (logRe, Sc) + PR.

modes:
  recon          A: encode sparse x_t -> decode x_t at query coords.            (info-complete AE)
  predict        B: encode sparse x_t -> predictor(Δ) -> decode x_{t+Δ}.        (future-field recon;
                    target = real future pixels = NON-collapsible => forces dynamics)
  predict_vicreg C: B + VICReg between two sparse views of x_t (observation-invariance).
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models import FAE
from src.models.fjepa import TokenPredictor
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens
from src.metrics import lin_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_projector(in_dim, d):
    f = [in_dim, d, d, d]; L = []
    for i in range(len(f) - 2):
        L += [nn.Linear(f[i], f[i + 1]), nn.BatchNorm1d(f[i + 1]), nn.ReLU(True)]
    L.append(nn.Linear(f[-2], f[-1], bias=False)); return nn.Sequential(*L)


def off_diag(x):
    n, m = x.shape; return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg(proj, rA, rB, B):
    xz, yz = proj(rA), proj(rB)
    l_sim = F.mse_loss(xz, yz)
    xz, yz = xz - xz.mean(0), yz - yz.mean(0)
    l_std = (F.relu(1 - torch.sqrt(xz.var(0) + 1e-4)).mean() / 2
             + F.relu(1 - torch.sqrt(yz.var(0) + 1e-4)).mean() / 2)
    d = xz.shape[1]
    l_cov = (off_diag((xz.T @ xz) / (B - 1)).pow_(2).sum() / d
             + off_diag((yz.T @ yz) / (B - 1)).pow_(2).sum() / d)
    return l_sim, l_std, l_cov


def participation_ratio(Z):
    Z = Z - Z.mean(0); e = np.clip(np.linalg.eigvalsh(Z.T @ Z / max(len(Z) - 1, 1)), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-30))


def grad_loss(a, b):
    """L2 on spatial first-differences (edges) — penalizes blur / spectral bias. a,b: (B,C,H,W)."""
    return (F.mse_loss(a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :])
            + F.mse_loss(a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1]))


def fft_loss(a, b, hf=True):
    """L2 in 2D Fourier magnitude, optionally |freq|-weighted to up-weight high frequencies."""
    d = (torch.fft.rfft2(a, norm="ortho") - torch.fft.rfft2(b, norm="ortho")).abs() ** 2   # (B,C,H,Wf)
    if hf:
        fy = torch.fft.fftfreq(a.shape[-2], device=a.device).abs()
        fx = torch.fft.rfftfreq(a.shape[-1], device=a.device)
        d = d * (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()[None, None]
    return d.mean()


@torch.no_grad()
def embed(model, ds, coords, idx, batch=128):
    """frozen encoder on frame 0 -> (mean+std, mean) readouts for single-frame eval."""
    model.eval(); Zms, Zm, Y = [], [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        fa = clip[:, :, 0].to(DEVICE)
        tok = model.encode_tokens(fields_to_tokens(fa, idx), coords[idx])
        Zms.append(torch.cat([tok.mean(1), tok.std(1)], -1).cpu().numpy())
        Zm.append(tok.mean(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Zms), np.concatenate(Zm), np.concatenate(Y)


PARAMS = ["logRe", "Sc"]


def probe2(Ztr, Ytr, Zva, Yva):
    out = {}
    for j, nm in enumerate(PARAMS):
        yt, yv = Ytr[:, j], Yva[:, j]; m, s = yt.mean(), yt.std() + 1e-8
        out[nm] = lin_probe(Ztr, (yt - m) / s, Zva, (yv - m) / s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="predict", choices=[
        "recon",          # A: present recon only
        "predict",        # B: future-field recon via predictor
        "predict_vicreg", # C: B + VICReg (deprecated — hurts)
        "recon_both",     # 3: present recon (anchor to x_t) + future-field recon (dynamics)
        "siam",           # 4: recon_both + latent-match to a diff-sparsity future view (no VICReg)
        "twoview"])       # 4': recon_both with two present views sharing the SAME recon targets
    ap.add_argument("--lam_match", type=float, default=1.0, help="weight on siam latent-match term")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dt_max", type=int, default=4)
    ap.add_argument("--dt_fixed", type=int, default=0,
                    help="0 = random Δ in [1,dt_max] (Δ fed to predictor); >0 = fixed horizon Δ")
    ap.add_argument("--mcnt", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--n_query", type=int, default=1024)
    ap.add_argument("--lam_rec", type=float, default=1.0)
    ap.add_argument("--sim", type=float, default=5.0)
    ap.add_argument("--std", type=float, default=100.0)
    ap.add_argument("--cov", type=float, default=5.0)
    ap.add_argument("--pred_depth", type=int, default=2)
    ap.add_argument("--num_latents", type=int, default=128, help="encoder bottleneck width (capacity)")
    ap.add_argument("--num_iter", type=int, default=4, help="encoder cross/self iterations (depth)")
    ap.add_argument("--emb_dim", type=int, default=320)
    ap.add_argument("--decoder_kind", choices=["senseiver", "cvit"], default="senseiver")
    ap.add_argument("--dec_blocks", type=int, default=2, help="cvit decoder blocks (capacity)")
    ap.add_argument("--proj_dim", type=int, default=4096)
    ap.add_argument("--n_seed", type=int, default=24)
    ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="faep")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--dataset", choices=["shear", "flowbench"], default="shear")
    ap.add_argument("--in_chans", type=int, default=None, help="default 4 (shear) / 3 (flowbench)")
    ap.add_argument("--norm_target", action="store_true", help="per-sample per-channel amplitude-stripped recon target")
    ap.add_argument("--lam_grad", type=float, default=0.0, help="gradient-loss weight (spectral-bias fix)")
    ap.add_argument("--lam_fft", type=float, default=0.0, help="FFT/spectral-loss weight (spectral-bias fix)")
    ap.add_argument("--fft_flat", action="store_true", help="FFT loss WITHOUT high-freq weighting")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    R = args.resolution; NPIX = R * R
    pred_mode = args.mode != "recon"                       # uses (t, t+Δ) clip + predictor
    do_present = args.mode in ("recon", "recon_both", "siam", "twoview")   # recon x_t (anchor)
    do_future = args.mode != "recon"                        # recon x_{t+Δ} via predictor
    use_vic = args.mode == "predict_vicreg"
    clip_len = (max(args.dt_max, args.dt_fixed) + 1) if pred_mode else 2
    print(f"=== FAE-{args.mode} [{args.tag}] dt_max={args.dt_max} mcnt={args.mcnt} "
          f"n_query={args.n_query} res={R} ===", flush=True)

    in_chans = args.in_chans if args.in_chans is not None else (3 if args.dataset == "flowbench" else 4)
    if args.dataset == "flowbench":
        from src.data.flowbench import FlowBenchFPO
        PARAMS[:] = ["Strouhal"]
        tr = FlowBenchFPO("train", side=R, mode="clip", clip_len=clip_len, frame_stride=args.frame_stride)
        va = FlowBenchFPO("valid", side=R, mode="clip", clip_len=clip_len, frame_stride=args.frame_stride, stats=tr.stats)
    else:
        tr = ShearFlowClipDataset("train", n_seed=args.n_seed, frame_stride=args.frame_stride,
                                  clip_len=clip_len, side=R)
        va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=args.frame_stride,
                                  clip_len=clip_len, side=R, stats=tr.stats)
    print(f"  dataset={args.dataset} in_chans={in_chans}  train {len(tr)} valid {len(va)}", flush=True)
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=4, pin_memory=True)
    coords = make_coords_2d(n_side=R, device=DEVICE)
    DS = 64; coords_d = make_coords_2d(n_side=DS, device=DEVICE)   # dense grid for grad/fft spectral losses

    model = FAE(emb_dim=args.emb_dim, num_iter=args.num_iter, depth_per_iter=4,
                num_latents=args.num_latents, num_cross_heads=4, num_self_heads=8,
                n_freq=32, max_freq=32, coord_dim=2, in_chans=in_chans,
                decoder_kind=args.decoder_kind, decoder_num_blocks=args.dec_blocks).to(DEVICE)
    npar = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  num_latents={args.num_latents} num_iter={args.num_iter} dec={args.decoder_kind} "
          f"params={npar:.2f}M", flush=True)
    predictor = TokenPredictor(args.emb_dim, depth=args.pred_depth, heads=8).to(DEVICE) if pred_mode else None
    proj = make_projector(320, args.proj_dim).to(DEVICE) if use_vic else None
    params = list(model.parameters())
    if predictor is not None: params += list(predictor.parameters())
    if proj is not None: params += list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    g0 = torch.Generator(device=DEVICE).manual_seed(0)
    probe_idx = torch.randperm(NPIX, generator=g0, device=DEVICE)[:1024]
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        ag = {"rec": 0, "std": 0, "n": 0}
        for clip, _y in loader:
            clip = clip.to(DEVICE, non_blocking=True); B = clip.size(0); K = clip.size(2)
            bidx = torch.arange(B, device=DEVICE)
            if pred_mode:
                if args.dt_fixed > 0:
                    delta = torch.full((B,), args.dt_fixed, device=DEVICE)
                else:
                    delta = torch.randint(1, args.dt_max + 1, (B,), device=DEVICE)
                ts = (torch.rand(B, device=DEVICE) * (K - delta).float()).long()
                fa = clip[bidx, :, ts]; fb = clip[bidx, :, ts + delta]; dt = delta.float() / args.dt_max
            else:
                fa = clip[:, :, 0]; fb = fa; dt = None
            nA = int(np.random.choice(args.mcnt))
            iA = torch.randperm(NPIX, device=DEVICE)[:nA]
            iq = torch.randperm(NPIX, device=DEVICE)[:args.n_query]
            tgt_t = fields_to_tokens(fa, iq); tgt_f = fields_to_tokens(fb, iq)
            tA = model.encode_tokens(fields_to_tokens(fa, iA), coords[iA])   # latent of INPUT x_t (probed)
            loss = torch.zeros((), device=DEVICE); l_rec = torch.zeros((), device=DEVICE)
            l_std = torch.tensor(0.0)
            if do_present:                                          # anchor latent to the input x_t
                pred_t = model.decoder(tA, coords[iq])
                if args.norm_target:                                # strip per-sample per-channel amplitude
                    sc = fa.std(dim=(2, 3)).unsqueeze(1).clamp_min(0.5)   # (B,1,C); floor: down-weight loud samples, don't amplify flat channels
                    lp = F.mse_loss(pred_t / sc, tgt_t / sc)
                else:
                    lp = F.mse_loss(pred_t, tgt_t)
                loss = loss + lp; l_rec = l_rec + lp
                if args.lam_grad > 0 or args.lam_fft > 0:          # dense decode -> spectral-bias losses
                    Bc, Cc = fa.size(0), fa.size(1)
                    xhat = model.decoder(tA, coords_d).reshape(Bc, DS, DS, Cc).permute(0, 3, 1, 2)
                    tgtd = F.interpolate(fa, size=(DS, DS), mode="bilinear", align_corners=False)
                    if args.lam_grad > 0: loss = loss + args.lam_grad * grad_loss(xhat, tgtd)
                    if args.lam_fft > 0: loss = loss + args.lam_fft * fft_loss(xhat, tgtd, hf=not args.fft_flat)
            tdec = predictor(tA, dt) if do_future else None
            if do_future:                                           # future-field recon (non-collapsible)
                lf = F.mse_loss(model.decoder(tdec, coords[iq]), tgt_f); loss = loss + lf; l_rec = l_rec + lf
            if args.mode == "siam":                                 # invariance via latent-match (no VICReg)
                nB = int(np.random.choice(args.mcnt)); iB = torch.randperm(NPIX, device=DEVICE)[:nB]
                Lb = model.encode_tokens(fields_to_tokens(fb, iB), coords[iB]).detach()   # diff view of future
                loss = loss + args.lam_match * F.mse_loss(tdec, Lb)
            if args.mode == "twoview":                              # invariance via SHARED recon target
                nB = int(np.random.choice(args.mcnt)); iB = torch.randperm(NPIX, device=DEVICE)[:nB]
                tB = model.encode_tokens(fields_to_tokens(fa, iB), coords[iB])            # 2nd view of x_t
                loss = loss + F.mse_loss(model.decoder(tB, coords[iq]), tgt_t)
                loss = loss + F.mse_loss(model.decoder(predictor(tB, dt), coords[iq]), tgt_f)
            if use_vic:
                nB = int(np.random.choice(args.mcnt))
                iB = torch.randperm(NPIX, device=DEVICE)[:nB]
                tB = model.encode_tokens(fields_to_tokens(fa, iB), coords[iB])
                l_sim, l_std, l_cov = vicreg(proj, model.represent(tA), model.represent(tB), B)
                loss = loss + args.sim * l_sim + args.std * l_std + args.cov * l_cov
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            ag["rec"] += float(l_rec) * B; ag["std"] += float(l_std) * B; ag["n"] += B
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            Zms, Zm, Ytr = embed(model, tr, coords, probe_idx)
            Vms, Vm, Yva = embed(model, va, coords, probe_idx)
            pr = participation_ratio(Zms); pb = probe2(Zms, Ytr, Vms, Yva); pm = probe2(Zm, Ytr, Vm, Yva)
            psms = " ".join(f"{k}={pb[k]:+.3f}" for k in PARAMS)
            psm = " ".join(f"{k}={pm[k]:+.3f}" for k in PARAMS)
            vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"ep {ep+1:3d}/{args.epochs}  rec={ag['rec']/ag['n']:.3e}  PR={pr:.1f}  "
                  f"mean+std {psms} | mean {psm}  peakVRAM={vram:.1f}GB  ({time.time()-t0:.0f}s)", flush=True)
    if args.save:
        out = f"results/checkpoints/g1/faep_{args.mode}_{args.tag}.pt"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ckpt = {"model": model.state_dict(), "stats": tr.stats, "train_args": vars(args)}
        if predictor is not None:
            ckpt["predictor"] = predictor.state_dict()      # latent flow, for future-prediction demo
        torch.save(ckpt, out)
        print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
