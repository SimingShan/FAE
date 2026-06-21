"""Their VICReg baseline on FlowBench FPO (faithful port: VICReg loss + ResNet-18 + crop augmentation).
Two random-resized-crop views per frame -> VICReg(sim/std/cov = 25/25/1). Frozen backbone -> linear
probe of log-shedding-frequency (Strouhal), vs the trivial floor. Logs feature-std (collapse guard) +
participation ratio. VICReg is data-hungry; FlowBench FPO is small -> watch for collapse (featstd<<1)."""
import os, sys, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchvision.models as tvm, torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from src.data.flowbench import FlowBenchFPO
from src.metrics import lin_probe
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def vicreg_loss(x, y, sim=25., std=25., cov=1.):
    rep = F.mse_loss(x, y)
    x = x - x.mean(0); y = y - y.mean(0)
    sx = torch.sqrt(x.var(0) + 1e-4); sy = torch.sqrt(y.var(0) + 1e-4)
    sl = (F.relu(1 - sx).mean() + F.relu(1 - sy).mean()) / 2
    n, d = x.shape
    cx = (x.T @ x) / (n - 1); cy = (y.T @ y) / (n - 1)
    off = lambda m: m.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
    cl = (off(cx).pow(2).sum() + off(cy).pow(2).sum()) / d
    return sim * rep + std * sl + cov * cl, sx.mean().item()


def backbone(in_chans=3):
    m = tvm.resnet18(weights=None); m.fc = nn.Identity()
    m.conv1 = nn.Conv2d(in_chans, 64, 7, 2, 3, bias=False)
    return m


def projector(d=512, w=512):
    return nn.Sequential(nn.Linear(d, w), nn.BatchNorm1d(w), nn.ReLU(True),
                         nn.Linear(w, w), nn.BatchNorm1d(w), nn.ReLU(True), nn.Linear(w, w, bias=False))


@torch.no_grad()
def embed(bb, ds, R):
    bb.eval(); Z, Y = [], []
    for clip, y in DataLoader(ds, batch_size=128):
        x = clip[:, :, 0].to(DEVICE)
        Z.append(bb(x).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="vicreg_fpo_s0")
    args = ap.parse_args(); set_seed(args.seed); R = args.resolution
    print(f"=== VICReg(theirs) FlowBench-FPO [{args.tag}] res={R} seed={args.seed} ===", flush=True)

    tr = FlowBenchFPO("train", side=R, mode="clip", clip_len=2, frame_stride=8)
    va = FlowBenchFPO("valid", side=R, mode="clip", clip_len=2, frame_stride=8)
    print(f"  train {len(tr)} clips ; valid {len(va)} clips", flush=True)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    aug = T.Compose([T.RandomResizedCrop(R, scale=(0.4, 1.0), antialias=True), T.RandomHorizontalFlip()])

    bb = backbone().to(DEVICE); pr = projector().to(DEVICE)
    opt = torch.optim.AdamW(list(bb.parameters()) + list(pr.parameters()), lr=args.lr, weight_decay=1e-6)
    for ep in range(args.epochs):
        bb.train(); pr.train(); tot = fs = n = 0
        for clip, _ in tl:
            x = clip[:, :, 0].to(DEVICE)
            v1, v2 = aug(x), aug(x)
            loss, featstd = vicreg_loss(pr(bb(v1)), pr(bb(v2)))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); fs += featstd; n += 1
        if ep % 10 == 0 or ep == args.epochs - 1:
            Ztr, ytr = embed(bb, tr, R); Zva, yva = embed(bb, va, R)
            m, s = ytr.mean(), ytr.std() + 1e-8
            r2 = lin_probe(Ztr, (ytr - m) / s, Zva, (yva - m) / s)
            C = np.cov(Ztr.T); pr_ratio = (np.trace(C) ** 2) / (np.sum(C * C) + 1e-9)
            print(f"  ep {ep+1:3d}/{args.epochs} loss={tot/n:.2f} featstd={fs/n:.3f} "
                  f"Strouhal_R2={r2:+.3f} PR={pr_ratio:.1f}", flush=True)
    out = f"results/checkpoints/g1/vicreg_{args.tag}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"backbone": bb.state_dict(), "args": vars(args)}, out)
    print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
