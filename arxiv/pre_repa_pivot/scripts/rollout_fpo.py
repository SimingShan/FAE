"""FlowBench FPO rollout (forecasting test on UNSEEN geometries). One-step prediction
frame_t (+time [+cond]) -> frame_{t+1} with the PDE-Arena UNetmod-64 + AdaGN. Conditioning rows:
  --mode time   : baseline
  --mode re     : true Reynolds (scalar)
  --mode rep    : frozen FAE-FPO rep of the first frames -> MLP(bottleneck 1) -> AdaGN
Split by geometry case (valid = unseen shapes). Reports one-step val MSE (x1e3). NOTE: unlike NS
buoyancy, the geometry is VISIBLE in the input, so this tests trajectory-summary conditioning, not
hidden-parameter inference."""
import os, sys, glob, re, argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Dataset, DataLoader
from src.cond_unet.twod_unet import Unet
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator
from scripts.rollout_ns import build_unet, compute_reps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FB = os.environ.get("FLOWBENCH_DIR", os.path.expanduser("~/scratch/flowbench")) + "/FPO_NS_2D_1024x256"
C0, C1, RESERVE = 60, 316, 16


def reynolds(path):
    m = re.search(r'Re_([0-9.eE+-]+)', os.path.basename(path))
    return float(m.group(1)) if m else float("nan")


class FPOStep(Dataset):
    def __init__(self, split, side=224, family="harmonics", stride=8, stats=None):
        sims = sorted(glob.glob(f"{FB}/{family}/*/Re_*.npz"))
        keep = [f for f in sims if (int(os.path.basename(os.path.dirname(f))) % 5 == 0) == (split == "valid")]
        self.traj, self.head, self.re = [], [], []
        for f in keep:
            d = np.load(f)['data'].astype(np.float32)[:, :, C0:C1, :]              # (T,256,256,3)
            solid = (~np.load(os.path.dirname(f) + "/input_geometry.npz")['mask'].astype(bool))[:, C0:C1]
            x = torch.from_numpy(d).permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
            sm = F.interpolate(torch.from_numpy(solid)[None, None].float(), (side, side))[0, 0] > 0.5
            x[:, :, sm] = 0.0
            self.traj.append(x.numpy()); self.head.append(x.numpy()[:RESERVE]); self.re.append(reynolds(f))
        self.re = np.array(self.re, np.float32); self.side = side; self.stride = stride
        if stats is None:
            cat = np.concatenate(self.traj).reshape(-1, 3, side, side)
            self.stats = (cat.mean((0, 2, 3)), cat.std((0, 2, 3)) + 1e-6)
        else:
            self.stats = stats
        m, s = self.stats
        for a in [self.traj, self.head]:
            for c in a: c -= m[None, :, None, None]; c /= s[None, :, None, None]
        rm, rs = self.re.mean(), self.re.std() + 1e-8
        self.re_n = (self.re - rm) / rs                                            # standardized Re for conditioning
        self.traj = [t.astype(np.float16) for t in self.traj]
        self.pairs = [(i, t) for i in range(len(self.traj)) for t in range(RESERVE, self.traj[i].shape[0] - 1, stride)]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, k):
        i, t = self.pairs[k]
        x = torch.from_numpy(self.traj[i][t].astype(np.float32)).unsqueeze(0)
        y = torch.from_numpy(self.traj[i][min(t + self.stride, self.traj[i].shape[0] - 1)].astype(np.float32)).unsqueeze(0)
        return x, y, float(t), i, float(self.re_n[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["time", "re", "rep"], required=True)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fae_ckpt", default="results/checkpoints/g1/faep_twoview_fae_fpo_s0.pt")
    args = ap.parse_args(); set_seed(args.seed)
    print(f"=== rollout FPO mode={args.mode} ep={args.epochs} seed={args.seed} (unseen-geometry valid) ===", flush=True)

    tr = FPOStep("train"); va = FPOStep("valid", stats=tr.stats)
    print(f"  train {len(tr.traj)} sims / {len(tr)} pairs ; valid {len(va.traj)} sims / {len(va)} pairs", flush=True)
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
        print(f"  ep {ep+1:2d}/{args.epochs}  val_one-step_MSE(x1e3)={1e3*vt/vn:.3f}", flush=True)
    print(f"DONE mode={args.mode}  final val one-step MSE x1e3 = {1e3*vt/vn:.3f}", flush=True)


if __name__ == "__main__":
    main()
