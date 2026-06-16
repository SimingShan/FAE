"""Battery probe for baselines (MAE/I-JEPA/etc.) — same physical-quantity panel as
eval_battery.py, but using the baseline's native encode (full image). Generality
head-to-head vs ours."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader
import scripts.train_baseline as tb
from scripts.eval_battery import quantities, NAMES
from src.data.well2d import ShearFlowSnapshotDataset
from src.metrics import lin_probe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; tb.DEVICE = DEVICE


@torch.no_grad()
def embed(model, ds, batch=64):
    model.eval(); Z, Q, Y = [], [], []
    for f, y in DataLoader(ds, batch_size=batch):
        f = f.to(DEVICE)
        Z.append(model.encode(f).float().cpu().numpy())
        Q.append(quantities(f).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Q), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/checkpoints/g1/mae_shear_mae_nf1.pt")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu"); ta = ck["train_args"]; R = ta["resolution"]
    model = tb.build_model(ta["method"], R, ta.get("n_frames", 1), ta.get("tubelet", 2))
    model.load_state_dict(ck["model"]); model.eval()
    tr = ShearFlowSnapshotDataset("train", n_seed=12, frame_stride=12, side=R)
    va = ShearFlowSnapshotDataset("valid", n_seed=8, frame_stride=12, side=R, stats=ck["stats"])
    Ztr, Qtr, Ytr = embed(model, tr); Zva, Qva, Yva = embed(model, va)
    Ttr = np.concatenate([Ytr, Qtr], 1); Tva = np.concatenate([Yva, Qva], 1)
    print(f"=== battery [{ta['method']} / {os.path.basename(args.ckpt)}] (full-image encode) ===")
    for j, nm in enumerate(NAMES):
        yt, yv = Ttr[:, j], Tva[:, j]; mn, s = yt.mean(), yt.std() + 1e-8
        r2 = lin_probe(Ztr, (yt - mn) / s, Zva, (yv - mn) / s)
        print(f"  {nm:12s}  R2={r2:+.3f}", flush=True)


if __name__ == "__main__":
    main()
