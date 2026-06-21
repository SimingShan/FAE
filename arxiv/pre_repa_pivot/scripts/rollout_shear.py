"""Shear-flow rollout (The Well). 4-channel state [tracer, pressure, vx, vy]; predict the full state
frame_t -> frame_{t+1} with PDE-Arena UNetmod-64 + AdaGN. Conditioning rows:
  --mode time : baseline
  --mode re   : true log10(Reynolds) (the CLEAN hidden parameter; trivial baselines fail it)
  --mode rep  : frozen FAE-shear rep of first frames -> MLP(bottleneck 1) -> AdaGN
Reports ONE-STEP val MSE and MULTI-STEP (autoregressive, horizon H) val MSE -- Re is a diffusion
coefficient (weak per-step, accumulates over a long rollout), so the multi-step number is the real test."""
import os, sys, glob, re, argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Dataset, DataLoader
from src.cond_unet.twod_unet import Unet
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator
from scripts.rollout_ns import compute_reps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WELL = os.environ.get("THE_WELL_DATA_DIR", os.path.expanduser("~/scratch/the_well_data")) + "/shear_flow/data"
RESERVE, H = 8, 8                                 # first RESERVE frames -> rep ; H-step autoregressive horizon


def re_sc(fn):
    m = re.search(r'Reynolds_([0-9e+-]+)_Schmidt_([0-9e+-]+)', os.path.basename(fn))
    return float(m.group(1)), float(m.group(2))


def build_unet():                                 # 4ch: 2 scalar (tracer,pressure) + 1 vector (velocity)
    return Unet(2, 1, 2, 1, time_history=1, time_future=1, hidden_channels=64, activation="gelu",
               norm=True, ch_mults=(1, 2, 2, 4), is_attn=(False,) * 4, n_blocks=2,
               param_conditioning="scalar", use_scale_shift_norm=True).to(DEVICE)


class ShearStep(Dataset):
    def __init__(self, split, side=224, n_traj=3, fstride=8, stats=None):
        import h5py
        files = sorted(glob.glob(f"{WELL}/{split}/shear_flow_Reynolds_*_Schmidt_*.hdf5"))
        self.traj, self.head, self.re = [], [], []
        for f in files:
            R_, S_ = re_sc(f)
            with h5py.File(f, "r") as h:
                ntr = min(n_traj, h["t0_fields/tracer"].shape[0])
                for tj in range(ntr):
                    tr = h["t0_fields/tracer"][tj, ::fstride]; pr = h["t0_fields/pressure"][tj, ::fstride]
                    vel = h["t1_fields/velocity"][tj, ::fstride]                       # (T',256,512,2)
                    x = np.stack([tr, pr, vel[..., 0], vel[..., 1]], 1).astype(np.float32)  # (T',4,256,512)
                    x = F.interpolate(torch.from_numpy(x), size=(side, side), mode="bilinear", align_corners=False).numpy()
                    self.traj.append(x); self.head.append(x[:RESERVE].copy()); self.re.append(np.log10(R_))
        self.re = np.array(self.re, np.float32)
        if stats is None:
            cat = np.concatenate(self.traj).reshape(-1, 4, side, side)
            self.stats = (cat.mean((0, 2, 3)), cat.std((0, 2, 3)) + 1e-6)
        else:
            self.stats = stats
        m, s = self.stats
        for a in (self.traj, self.head):
            for c in a: c -= m[None, :, None, None]; c /= s[None, :, None, None]
        rm, rs = self.re.mean(), self.re.std() + 1e-8; self.re_n = (self.re - rm) / rs
        self.traj16 = [t.astype(np.float16) for t in self.traj]
        self.pairs = [(i, t) for i in range(len(self.traj16)) for t in range(RESERVE, self.traj16[i].shape[0] - 1)]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, k):
        i, t = self.pairs[k]
        x = torch.from_numpy(self.traj16[i][t].astype(np.float32)).unsqueeze(0)
        y = torch.from_numpy(self.traj16[i][t + 1].astype(np.float32)).unsqueeze(0)
        return x, y, float(t), i, float(self.re_n[i])


@torch.no_grad()
def multistep(net, ds, cond_fn):
    """autoregressive H-step rollout from t=RESERVE on each valid traj; mean MSE over the horizon."""
    net.eval(); tot = n = 0
    for i in range(len(ds.traj16)):
        tr = ds.traj16[i]
        if tr.shape[0] < RESERVE + H + 1: continue
        x = torch.from_numpy(tr[RESERVE].astype(np.float32)).to(DEVICE)[None, None]
        ea, z = cond_fn(i)
        for k in range(H):
            x = net(x, torch.tensor([float(RESERVE + k)], device=DEVICE), z, ea)
            gt = torch.from_numpy(tr[RESERVE + 1 + k].astype(np.float32)).to(DEVICE)[None, None]
            tot += F.mse_loss(x, gt).item(); n += 1
    return 1e3 * tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["time", "re", "rep"], required=True)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_traj", type=int, default=3)
    ap.add_argument("--fae_ckpt", default="results/checkpoints/g1/faep_twoview_fae_shear_s0.pt")
    args = ap.parse_args(); set_seed(args.seed)
    print(f"=== rollout SHEAR mode={args.mode} ep={args.epochs} seed={args.seed} (4ch, Re cond, 1-step + {H}-step) ===", flush=True)
    tr = ShearStep("train", n_traj=args.n_traj); va = ShearStep("valid", n_traj=args.n_traj, stats=tr.stats)
    print(f"  train {len(tr.traj16)} traj / {len(tr)} pairs ; valid {len(va.traj16)} traj / {len(va)} pairs", flush=True)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4)
    net = build_unet(); params = list(net.parameters()); rep_mlp = rtr = rva = None
    if args.mode == "rep":
        rtr = compute_reps(tr.head, args.fae_ckpt); rva = compute_reps(va.head, args.fae_ckpt)
        rep_mlp = nn.Sequential(nn.Linear(rtr.shape[1], 1), nn.GELU(), nn.Linear(1, 256)).to(DEVICE)
        nn.init.zeros_(rep_mlp[-1].weight); nn.init.zeros_(rep_mlp[-1].bias); params += list(rep_mlp.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    warm, total = 0.05 * args.epochs * len(tl), args.epochs * len(tl); step = 0
    cond_va = lambda i: (rep_mlp(rva[i:i+1]) if args.mode == "rep" else None,
                         torch.tensor([va.re_n[i]], device=DEVICE) if args.mode == "re" else None)
    for ep in range(args.epochs):
        net.train()
        for x, y, t, i, r in tl:
            x, y, t, r = x.to(DEVICE), y.to(DEVICE), t.to(DEVICE), r.to(DEVICE); i = i.long().to(DEVICE)
            for g in opt.param_groups: g["lr"] = args.lr * (min(step/max(1,warm),1.) if step<warm else 0.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm))))
            z = r if args.mode == "re" else None
            ea = rep_mlp(rtr[i]) if args.mode == "rep" else None
            loss = F.mse_loss(net(x, t, z, ea), y)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        net.eval(); vt = vn = 0
        with torch.no_grad():
            for x, y, t, i, r in vl:
                x, y, t, r = x.to(DEVICE), y.to(DEVICE), t.to(DEVICE), r.to(DEVICE); i = i.long().to(DEVICE)
                z = r if args.mode == "re" else None
                ea = rep_mlp(rva[i]) if args.mode == "rep" else None
                vt += F.mse_loss(net(x, t, z, ea), y).item() * x.size(0); vn += x.size(0)
        if ep == args.epochs - 1 or ep % 5 == 4:
            ms = multistep(net, va, cond_va)
            print(f"  ep {ep+1:2d}/{args.epochs}  1step_MSE(x1e3)={1e3*vt/vn:.3f}  {H}step_MSE(x1e3)={ms:.3f}", flush=True)
    print(f"DONE mode={args.mode}  1step={1e3*vt/vn:.3f}  {H}step={multistep(net, va, cond_va):.3f}", flush=True)


if __name__ == "__main__":
    main()
