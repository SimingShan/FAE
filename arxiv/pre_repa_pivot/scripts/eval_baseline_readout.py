"""Re-probe saved baseline checkpoints with BOTH mean and mean+std readouts, so the
comparison to our twoview is on a matched readout. Token access per method:
  mae/ae/videomae : forward_encoder(x,0)[0][:,1:]   (drop cls)
  ijepa/stjepa    : target(x)
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader
import scripts.train_baseline as tb
from src.metrics import lin_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tb.DEVICE = DEVICE


@torch.no_grad()
def tokens(model, method, x):
    if method in ("mae", "ae", "videomae"):
        lat, _, _ = model.forward_encoder(x, mask_ratio=0.0)
        return lat[:, 1:, :]
    return model.target(x)                      # ijepa / stjepa


@torch.no_grad()
def embed(model, method, ds, batch=64):
    model.eval(); Zm, Zms, Y = [], [], []
    for f, y in DataLoader(ds, batch_size=batch):
        t = tokens(model, method, f.to(DEVICE)).float()
        Zm.append(t.mean(1).cpu().numpy())
        Zms.append(torch.cat([t.mean(1), t.std(1)], -1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Zm), np.concatenate(Zms), np.concatenate(Y)


def probe(Ztr, Ytr, Zva, Yva):
    out = []
    for j in range(2):
        yt, yv = Ytr[:, j], Yva[:, j]; m, s = yt.mean(), yt.std() + 1e-8
        out.append(lin_probe(Ztr, (yt - m) / s, Zva, (yv - m) / s))
    return out


def run(path):
    ck = torch.load(path, map_location="cpu"); ta = ck["train_args"]
    method = ta["method"]; nf = ta.get("n_frames", 1); R = ta["resolution"]
    model = tb.build_model(method, R, nf, ta.get("tubelet", 2))
    model.load_state_dict(ck["model"]); model.eval()
    from src.data.well2d import ShearFlowSnapshotDataset, ShearFlowWindowDataset
    if method in ("videomae", "stjepa"):
        tr = ShearFlowWindowDataset("train", n_seed=ta["n_seed"], n_frames=nf, side=R)
        va = ShearFlowWindowDataset("valid", n_seed=8, n_frames=nf, side=R, stats=tr.stats)
    else:
        tr = ShearFlowSnapshotDataset("train", n_seed=ta["n_seed"], frame_stride=12, side=R)
        va = ShearFlowSnapshotDataset("valid", n_seed=8, frame_stride=12, side=R, stats=tr.stats)
    Ztr_m, Ztr_ms, Ytr = embed(model, method, tr); Zva_m, Zva_ms, Yva = embed(model, method, va)
    rm = probe(Ztr_m, Ytr, Zva_m, Yva); rms = probe(Ztr_ms, Ytr, Zva_ms, Yva)
    print(f"{method:9s} nf={nf:2d}  mean[logRe={rm[0]:+.3f} Sc={rm[1]:+.3f}]  "
          f"mean+std[logRe={rms[0]:+.3f} Sc={rms[1]:+.3f}]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", nargs="+", required=True)
    for p in ap.parse_args().ckpt:
        try: run(p)
        except Exception as e: print(f"FAIL {p}: {e}", flush=True)
