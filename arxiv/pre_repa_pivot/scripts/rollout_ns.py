"""Reproduce SSLForPDEs Table 2 (NS time-stepping) on our data, faithfully.

Solver = PDE-Arena modified UNet (UNetmod-64) with AdaGN conditioning (vendored in src/cond_unet).
One-step prediction: frame_t (+time [+cond]) -> frame_{t+1}; report one-step validation MSE (x1e3).
Conditioning rows (paper App F.3):
  --mode time       : z=None                          (baseline)
  --mode buoyancy   : z=true buoyancy (upper bound)
  --mode rep        : frozen-encoder rep on first 16 frames -> 2-layer MLP(bottleneck=1) -> add to cond emb (ours)
First RESERVE(16) frames are held out of time-stepping (used only for the rep), per the paper.
"""
import os, sys, glob, argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Dataset, DataLoader
from src.cond_unet.twod_unet import Unet
from src.data.ns import buoyancy
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator

NS = os.environ.get("NS_DATA_ROOT", os.path.expanduser("~/scratch/ns_data"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAJLEN, RESERVE, SIDE = 56, 16, 128


class NSStep(Dataset):
    """One-step pairs (frame_t, frame_{t+1}) from frames [RESERVE..TRAJLEN-1]; stores first RESERVE frames for rep."""
    def __init__(self, split, n_traj_per_file, stats=None):
        import h5py
        files = sorted(glob.glob(f"{NS}/*_{split}_*.h5"))
        self.traj, self.head, self.buoy = [], [], []
        for f in files:
            b = buoyancy(f)
            with h5py.File(f, "r") as h:
                g = h[split]; ntr = min(n_traj_per_file, g["u"].shape[0])
                for tj in range(ntr):
                    arr = np.stack([g["u"][tj], g["vx"][tj], g["vy"][tj]], 1).astype(np.float32)  # (T,3,H,W)
                    self.traj.append(arr); self.head.append(arr[:RESERVE].copy()); self.buoy.append(b)
        self.buoy = np.array(self.buoy, np.float32)
        if stats is None:
            cat = np.concatenate(self.traj).reshape(-1, 3, SIDE, SIDE)
            self.stats = (cat.mean((0, 2, 3)), cat.std((0, 2, 3)) + 1e-6)
        else:
            self.stats = stats
        m, s = self.stats
        for a in self.traj: a -= m[None, :, None, None]; a /= s[None, :, None, None]
        for a in self.head: a -= m[None, :, None, None]; a /= s[None, :, None, None]
        self.traj = [t.astype(np.float16) for t in self.traj]
        self.pairs = [(i, t) for i in range(len(self.traj)) for t in range(RESERVE, TRAJLEN - 1)]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, k):
        i, t = self.pairs[k]
        x = torch.from_numpy(self.traj[i][t].astype(np.float32)).unsqueeze(0)        # (1,3,H,W)
        y = torch.from_numpy(self.traj[i][t + 1].astype(np.float32)).unsqueeze(0)    # (1,3,H,W)
        return x, y, float(t), i, float(self.buoy[i])


def build_unet():
    return Unet(1, 1, 1, 1, time_history=1, time_future=1, hidden_channels=64, activation="gelu",
               norm=True, ch_mults=(1, 2, 2, 4), is_attn=(False,) * 4, n_blocks=2,
               param_conditioning="scalar", use_scale_shift_norm=True).to(DEVICE)


@torch.no_grad()
def compute_reps(heads, fae_ckpt):
    """frozen FAE encode of the first RESERVE frames (averaged) -> (N, d) rep per trajectory."""
    from scripts.eval_ns_probe import load_fae
    from src.data.well2d import fields_to_tokens
    model, coords, idx = load_fae(fae_ckpt); model.eval()
    reps = []
    for h in heads:                                  # h: (RESERVE,3,H,W) normalized
        fr = torch.from_numpy(h.astype(np.float32)).to(DEVICE)
        tok = model.encode_tokens(fields_to_tokens(fr, idx), coords[idx])       # (RESERVE, L, d)
        z = torch.cat([tok.mean(1), tok.std(1)], -1).mean(0)                    # avg over frames -> (2d,)
        reps.append(z.cpu().numpy())
    return torch.tensor(np.stack(reps), dtype=torch.float32, device=DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["time", "buoyancy", "rep"], required=True)
    ap.add_argument("--n_traj_per_file", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fae_ckpt", default="results/checkpoints/g1/faep_twoview_fae_ns_tw.pt")
    ap.add_argument("--tag", default="r")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    print(f"=== rollout NS  mode={args.mode}  n_traj/file={args.n_traj_per_file}  lr={args.lr}  ep={args.epochs}  seed={args.seed} ===", flush=True)

    tr = NSStep("train", args.n_traj_per_file)
    va = NSStep("valid", args.n_traj_per_file, stats=tr.stats)
    print(f"  train {len(tr.traj)} traj / {len(tr)} pairs ; valid {len(va.traj)} traj / {len(va)} pairs", flush=True)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4)

    net = build_unet()
    params = list(net.parameters())
    rep_mlp = reps_tr = reps_va = None
    if args.mode == "rep":
        reps_tr = compute_reps(tr.head, args.fae_ckpt); reps_va = compute_reps(va.head, args.fae_ckpt)
        d = reps_tr.shape[1]; emb_dim = 64 * 4
        rep_mlp = nn.Sequential(nn.Linear(d, 1), nn.GELU(), nn.Linear(1, emb_dim)).to(DEVICE)  # bottleneck=1 (crucial)
        nn.init.zeros_(rep_mlp[-1].weight); nn.init.zeros_(rep_mlp[-1].bias)                   # start as no-op -> learn to add signal
        params += list(rep_mlp.parameters())
        print(f"  rep dim={d} -> MLP bottleneck 1 -> emb {emb_dim}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    warm, total = 0.05 * args.epochs * len(tl), args.epochs * len(tl); step = 0

    def cond(net_in, batch_t, batch_i, batch_b):
        z = emb_add = None
        if args.mode == "buoyancy": z = batch_b
        elif args.mode == "rep": emb_add = rep_mlp(reps_tr[batch_i] if net.training else reps_va[batch_i])
        return net(net_in, batch_t, z, emb_add)

    for ep in range(args.epochs):
        net.train(); tot = n = 0
        for x, y, t, i, b in tl:
            x, y, t, b = x.to(DEVICE), y.to(DEVICE), t.to(DEVICE), b.to(DEVICE); i = i.long().to(DEVICE)
            for g in opt.param_groups: g["lr"] = args.lr * (min(step / max(1, warm), 1.0) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            pred = cond(x, t, i, b); loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            tot += loss.item() * x.size(0); n += x.size(0)
        # one-step validation MSE
        net.eval(); vtot = vn = 0
        with torch.no_grad():
            for x, y, t, i, b in vl:
                x, y, t, b = x.to(DEVICE), y.to(DEVICE), t.to(DEVICE), b.to(DEVICE); i = i.long().to(DEVICE)
                z = b if args.mode == "buoyancy" else None
                ea = rep_mlp(reps_va[i]) if args.mode == "rep" else None
                vtot += F.mse_loss(net(x, t, z, ea), y).item() * x.size(0); vn += x.size(0)
        print(f"  ep {ep+1:2d}/{args.epochs}  train_mse={tot/n:.4e}  val_one-step_MSE(x1e3)={1e3*vtot/vn:.3f}", flush=True)
    print(f"DONE mode={args.mode}  final val one-step MSE x1e3 = {1e3*vtot/vn:.3f}", flush=True)
    if args.save:
        out = f"results/checkpoints/g1/rollout_ns_{args.mode}_s{args.seed}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
        ck = {"net": net.state_dict(), "stats": tr.stats, "mode": args.mode, "args": vars(args)}
        if rep_mlp is not None: ck["rep_mlp"] = rep_mlp.state_dict()
        torch.save(ck, out); print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
