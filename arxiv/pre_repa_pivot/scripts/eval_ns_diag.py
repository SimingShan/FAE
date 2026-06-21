"""Diagnostic: is the ours-vs-theirs buoyancy gap real, or a fixed-ridge-alpha artifact on the
small (19-buoyancy) valid->test split? Embed BOTH frozen reps, then sweep ridge alpha + RidgeCV,
report valid->test R². Also: best single-feature |corr| with buoyancy (is the signal even there?)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/SSLForPDEs"))
import torch.nn as nn, torchvision.models as tvm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from scripts.eval_ns_probe import load_fae, embed_fae
from src.data.ns import NSDataset
from utils import get_eval_loader_ns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA = os.path.expanduser("~/scratch/ns_data")
ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]


def sweep(name, Ztr, ytr, Zte, yte):
    sc = StandardScaler().fit(Ztr); Xtr, Xte = sc.transform(Ztr), sc.transform(Zte)
    m, s = ytr.mean(), ytr.std() + 1e-8; yt, ye = (ytr - m) / s, (yte - m) / s
    # best single standardized feature corr with buoyancy
    corr = np.nan_to_num([np.corrcoef(Xtr[:, j], yt)[0, 1] for j in range(Xtr.shape[1])])
    line = f"  {name:14s} d={Ztr.shape[1]:4d} max|corr|={np.abs(corr).max():.3f} | "
    for a in ALPHAS:
        r2 = r2_score(ye, Ridge(a).fit(Xtr, yt).predict(Xte))
        line += f"a{a:g}:{r2:+.2f} "
    rcv = RidgeCV(alphas=ALPHAS).fit(Xtr, yt)
    line += f"|| RidgeCV(a={rcv.alpha_:g}):{r2_score(ye, rcv.predict(Xte)):+.3f}"
    print(line, flush=True)


@torch.no_grad()
def embed_theirs(ckpt, loader):
    ck = torch.load(ckpt, map_location=DEVICE); sd = ck["model"]
    bb = tvm.resnet18(weights=None); bb.fc = nn.Identity()
    bb.conv1 = nn.Conv2d(80, 64, 7, 2, 3, bias=False)
    bb.load_state_dict({k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=False)
    bb = bb.to(DEVICE).eval()
    Z, Y = [], []
    for x, c in loader:
        Z.append(bb(x.to(DEVICE)).cpu().numpy()); Y.append(np.asarray(c).ravel())
    return np.concatenate(Z), np.concatenate(Y)


def main():
    print("=== NS buoyancy probe DIAGNOSTIC (valid->test) — alpha sweep + RidgeCV ===", flush=True)
    # OURS
    va = NSDataset("valid", side=128, mode="clip", clip_len=2, frame_stride=4, n_traj=16)
    te = NSDataset("test", side=128, mode="clip", clip_len=2, frame_stride=4, n_traj=16, stats=va.stats)
    mdl, coords, idx = load_fae("results/checkpoints/g1/faep_twoview_fae_ns_tw.pt")
    Ztr, ytr = embed_fae(mdl, va, coords, idx); Zte, yte = embed_fae(mdl, te, coords, idx)
    sweep("OURS base", Ztr, ytr, Zte, yte)
    # THEIRS (their exact eval pipeline)
    vl = get_eval_loader_ns(DATA, 64, 4, mode="val", crop_size=(16, 128, 128))
    tl = get_eval_loader_ns(DATA, 64, 4, mode="test", crop_size=(16, 128, 128))
    Ztr2, ytr2 = embed_theirs("results/ssl_logs/ssl_full/model.pth", vl)
    Zte2, yte2 = embed_theirs("results/ssl_logs/ssl_full/model.pth", tl)
    sweep("THEIRS VICReg", Ztr2, ytr2, Zte2, yte2)
    print("\n  interpretation: if THEIRS jumps positive at high alpha/RidgeCV => fixed-alpha artifact;\n"
          "  if it stays <=0 even at best alpha with low max|corr| => rep genuinely lacks a linear buoyancy axis.")


if __name__ == "__main__":
    main()
