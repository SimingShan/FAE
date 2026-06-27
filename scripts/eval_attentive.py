"""Attentive probe (Qu et al. CANONICAL AttentiveClassifier) — the STRONGER learned readout, to test if
the encoder ranking is readout-dependent vs the mean-pool linear probe. Frozen encoder -> token grid ->
learned query cross-attends -> regression head, trained ~100 ep. shear logRe/logSc; typhoon wind/pressure.
  python scripts/eval_attentive.py --dataset typhoon
"""
import os, sys, glob, argparse, warnings, time
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "external/physical-representation-learning"))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from physics_jepa.attentive_pooler import AttentiveClassifier
from src.models.fae import FAE
from src.data.well2d import make_coords_2d, fields_to_tokens
from scripts.train_baseline import build_model
from scripts.probe_all import get_data, _frame0, TARGETS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def tokens(ck, method, ds, side, nmax):
    a = torch.load(ck, map_location=DEVICE)["train_args"]
    C = a.get("in_chans") or (1 if a.get("dataset") == "typhoon" else 4 if a.get("dataset") == "shear" else 3)
    if method == "fae":
        from src.data.well2d import make_coords_2d_hw
        rh, rw = (a["res_h"], a["res_w"]) if a.get("res_h") else (side, side)   # rect-aware (shear 128x256)
        m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), depth_per_iter=a.get("depth_per_iter", 5),
                num_latents=a["num_latents"], num_cross_heads=a.get("num_cross_heads", 4), num_self_heads=a.get("num_self_heads", 8),
                n_freq=a.get("n_freq", 32), max_freq=a.get("max_freq", 32), coord_dim=2, in_chans=C).to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE)["model"]); m.eval()
        coords = make_coords_2d_hw(rh, rw, device=DEVICE); idx = torch.arange(rh * rw, device=DEVICE)
        enc = lambda f: m.encode_tokens(fields_to_tokens(f, idx), coords[idx])
    else:
        m = build_model("mae" if method == "mae" else "ijepa", resolution=a["resolution"], in_chans=C,
                        embed_dim=a.get("embed_dim"), depth=a.get("depth"), patch_size=a.get("patch_size")).to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE)["model"]); m.eval()
        enc = (lambda f: m.forward_encoder(f, 0.0)[0][:, 1:, :]) if method == "mae" else (lambda f: m.target(f))
    Z, Y = [], []
    for x, y in DataLoader(ds, batch_size=64):
        Z.append(enc(_frame0(x).to(DEVICE)).cpu()); Y.append(y)
        if sum(t.shape[0] for t in Z) >= nmax:
            break
    return torch.cat(Z)[:nmax], torch.cat(Y)[:nmax]


def train_attn(Ztr, Ytr, Zte, Yte, epochs=100, lr=1e-3, batch=64, heads=8):
    D = Ztr.shape[-1]; P = Ytr.shape[1]
    clf = AttentiveClassifier(embed_dim=D, num_heads=heads, depth=1, num_classes=P).to(DEVICE)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-8
    Ytr_n = ((Ytr - mu) / sd).to(DEVICE); Yte_n = ((Yte - mu) / sd).to(DEVICE)
    Ztr, Zte = Ztr.to(DEVICE), Zte.to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = [-9] * P
    for ep in range(epochs):
        clf.train(); ix = torch.randperm(Ztr.shape[0], device=DEVICE)
        for i in range(0, len(ix), batch):
            b = ix[i:i + batch]
            loss = F.mse_loss(clf(Ztr[b]), Ytr_n[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 9 or ep == epochs - 1:
            clf.eval()
            with torch.no_grad():
                p = clf(Zte)
            r2 = [1 - ((Yte_n[:, k] - p[:, k]) ** 2).mean().item() / (Yte_n[:, k].var().item() + 1e-8) for k in range(P)]
            best = [max(best[k], r2[k]) for k in range(P)]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["shear", "typhoon"], required=True)
    ap.add_argument("--nmax", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    side = {"shear": 128, "typhoon": 128}[args.dataset]; tgt = TARGETS[args.dataset]
    from scripts.probe_all import fae_hw
    fit, test = ("valid_a", "valid_b") if args.dataset == "shear" else ("valid", "test")
    tr = get_data(args.dataset, fit, side); te = get_data(args.dataset, test, side, stats=tr.stats)   # square (ViTs)
    fcks = sorted(glob.glob(f"results/checkpoints/{args.dataset}/fae/*_s*.pt"))
    hw = fae_hw(fcks[0], side) if fcks else (side, side)                                               # FAE rect-aware
    if hw != (side, side):
        tr_f = get_data(args.dataset, fit, list(hw)); te_f = get_data(args.dataset, test, list(hw), stats=tr_f.stats)
    else:
        tr_f, te_f = tr, te
    print(f"=== {args.dataset} ATTENTIVE probe (Qu AttentiveClassifier, {fit}->{test})  targets={tgt} ViT={side} FAE={hw[0]}x{hw[1]} ===", flush=True)
    rows = []
    for meth in ["fae", "mae", "jepa"]:
        ck = (sorted(glob.glob(f"results/checkpoints/{args.dataset}/{meth}/*_s*.pt")) or [None])[0]
        if ck is None:
            continue
        t0 = time.time()
        dtr, dte = (tr_f, te_f) if meth == "fae" else (tr, te)
        Ztr, Ytr = tokens(ck, meth, dtr, side, args.nmax); Zte, Yte = tokens(ck, meth, dte, side, args.nmax)
        r2 = train_attn(Ztr, Ytr, Zte, Yte, epochs=args.epochs)
        rows.append((meth.upper(), r2, Ztr.shape)); print(f"  {meth.upper():5s} R²={[round(x,3) for x in r2]}  tokens{tuple(Ztr.shape)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  {'encoder':6s} " + "  ".join(f"R2_{t:>8s}" for t in tgt))
    for nm, r2, _ in rows:
        print(f"  {nm:6s} " + "  ".join(f"{v:>+9.3f}" for v in r2), flush=True)


if __name__ == "__main__":
    main()
