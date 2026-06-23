"""DINO-FAE: authentic DINO self-distillation on the functional encoder (replaces my collapsing cosine
version). DINOHead + DINOLoss are copied verbatim from facebookresearch/dino (external/dino), single-GPU.

Anti-collapse = teacher CENTERING (subtract EMA mean) + SHARPENING (teacher low temp) + EMA teacher — the
exact mechanism my hand-rolled cosine lacked. Two views = two sparsity sensor subsets (our 'multi-crop').
Decoder trained on DETACHED latents so it stays usable for the REPA per-patch readout. Geometric Fourier.
Encoder is the product; heads/teacher discarded at eval.
"""
import os, sys, argparse, copy, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.init import trunc_normal_
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from src.models.fae import FAE
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---- copied from facebookresearch/dino (vision_transformer.py) ----
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer: self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x); x = F.normalize(x, dim=-1, p=2); return self.last_layer(x)


# ---- copied from facebookresearch/dino (main_dino.py), single-GPU (no dist) ----
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp; self.center_momentum = center_momentum; self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(nepochs - warmup_teacher_temp_epochs, 1)) * teacher_temp))

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1).detach().chunk(2)
        total, n = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: continue
                total += torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1).mean(); n += 1
        self.update_center(teacher_output)
        return total / n

    @torch.no_grad()
    def update_center(self, teacher_output):
        bc = torch.sum(teacher_output, dim=0, keepdim=True) / len(teacher_output)
        self.center = self.center * self.center_momentum + bc * (1 - self.center_momentum)


def ridge_r2(Ztr, Ytr, Zva, Yva, lam=1.0):
    mu, sd = Ztr.mean(0), Ztr.std(0).clamp_min(1e-6)                         # standardize BOTH by TRAIN stats
    Ztr = (Ztr - mu) / sd; Zva = (Zva - mu) / sd
    W = torch.linalg.solve(Ztr.T @ Ztr + lam * torch.eye(Ztr.size(1), device=Ztr.device), Ztr.T @ Ytr)
    ss = ((Yva - Zva @ W) ** 2).sum(0); tot = ((Yva - Yva.mean(0)) ** 2).sum(0).clamp_min(1e-6)
    return (1 - ss / tot).mean().item()


def participation_ratio(Z):
    ev = torch.linalg.eigvalsh(torch.cov((Z - Z.mean(0)).T)).clamp_min(0)
    return (ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12)).item()


@torch.no_grad()
def embed(model, ds, coords, idx, n=512):
    xs = torch.from_numpy(np.stack([ds[i][0].numpy() for i in range(min(len(ds), n))])).to(DEVICE)
    ys = torch.tensor(np.stack([np.atleast_1d(ds[i][1]) for i in range(min(len(ds), n))]), dtype=torch.float32, device=DEVICE)
    return model.encode_tokens(fields_to_tokens(xs, idx), coords[idx]).mean(1), ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ns"); ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mcnt", type=int, nargs="+", default=[256, 512], help="per-view sensor counts (sampled)")
    ap.add_argument("--n_query", type=int, default=1024); ap.add_argument("--lam_dec", type=float, default=1.0)
    ap.add_argument("--out_dim", type=int, default=4096); ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--teacher_temp", type=float, default=0.04); ap.add_argument("--student_temp", type=float, default=0.1)
    ap.add_argument("--emb_dim", type=int, default=320); ap.add_argument("--num_latents", type=int, default=128)
    ap.add_argument("--fourier_geometric", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tag", default="fae_dino"); ap.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(); set_seed(args.seed); R = args.resolution; C = 3 if args.dataset == "ns" else 4
    print(f"=== DINO-FAE [{args.tag}] geo={args.fourier_geometric} out_dim={args.out_dim} Tt={args.teacher_temp} "
          f"ema={args.ema} mcnt={args.mcnt} res={R} ===", flush=True)

    tr = NSDataset("train", side=R, mode="single", clip_len=2, frame_stride=4, n_traj=12)
    va = NSDataset("valid", side=R, mode="single", clip_len=2, frame_stride=4, n_traj=8, stats=tr.stats)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    model = FAE(emb_dim=args.emb_dim, num_latents=args.num_latents, coord_dim=2, in_chans=C,
                fourier_geometric=args.fourier_geometric).to(DEVICE)
    s_head = DINOHead(args.emb_dim, args.out_dim).to(DEVICE)
    t_enc = copy.deepcopy(model.encoder).eval()
    t_head = DINOHead(args.emb_dim, args.out_dim).to(DEVICE); t_head.load_state_dict(s_head.state_dict()); t_head.eval()
    for p in list(t_enc.parameters()) + list(t_head.parameters()): p.requires_grad_(False)
    dino = DINOLoss(args.out_dim, 2, args.teacher_temp, args.teacher_temp, 0, args.epochs, args.student_temp).to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(s_head.parameters()), lr=args.lr)
    coords = make_coords_2d(n_side=R, device=DEVICE); NPIX = R * R
    g = torch.Generator(device=DEVICE).manual_seed(0); pidx = torch.randperm(NPIX, generator=g, device=DEVICE)[:1024]

    for ep in range(args.epochs):
        model.train(); s_head.train(); ag = {"d": 0.0, "r": 0.0, "n": 0}; t0 = time.time()
        for x, _ in tl:
            x = x.to(DEVICE); n = x.size(0)
            iA = torch.randperm(NPIX, device=DEVICE)[:int(np.random.choice(args.mcnt))]
            iB = torch.randperm(NPIX, device=DEVICE)[:int(np.random.choice(args.mcnt))]
            iq = torch.randperm(NPIX, device=DEVICE)[:args.n_query]
            sA = model.encode_tokens(fields_to_tokens(x, iA), coords[iA]).mean(1)
            sB = model.encode_tokens(fields_to_tokens(x, iB), coords[iB]).mean(1)
            s_out = s_head(torch.cat([sA, sB], 0))
            with torch.no_grad():
                tA = t_enc(fields_to_tokens(x, iA), coords[iA]).mean(1)
                tB = t_enc(fields_to_tokens(x, iB), coords[iB]).mean(1)
                t_out = t_head(torch.cat([tA, tB], 0))
            distill = dino(s_out, t_out, ep)
            full = model.encode_tokens(fields_to_tokens(x, iA), coords[iA])
            rec = F.mse_loss(model.decoder(full, coords[iq]), fields_to_tokens(x, iq))   # recon ANCHORS+bootstraps encoder
            loss = distill + args.lam_dec * rec
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pt, pe in zip(t_enc.parameters(), model.encoder.parameters()): pt.mul_(args.ema).add_(pe, alpha=1 - args.ema)
                for pt, pe in zip(t_head.parameters(), s_head.parameters()): pt.mul_(args.ema).add_(pe, alpha=1 - args.ema)
            ag["d"] += distill.item() * n; ag["r"] += rec.item() * n; ag["n"] += n
        if ep % 10 == 9 or ep == args.epochs - 1:
            model.eval(); Ztr, Ytr = embed(model, tr, coords, pidx); Zva, Yva = embed(model, va, coords, pidx)
            print(f"ep {ep+1:3d}/{args.epochs}  distill={ag['d']/ag['n']:.3f}  rec={ag['r']/ag['n']:.3e}  "
                  f"PR={participation_ratio(Ztr):.1f}  probe R2={ridge_r2(Ztr, Ytr, Zva, Yva):+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if args.save:
        out = f"results/checkpoints/g1/faep_dino_{args.tag}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"model": model.state_dict(), "stats": tr.stats, "train_args": vars(args)}, out); print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
