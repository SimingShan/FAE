"""Ridge-probe THEIR frozen VICReg ResNet-18 on NS buoyancy, valid->test, SAME probe + split as ours.
Uses their exact eval input pipeline (get_eval_loader_ns: 16-frame x 5-field -> 80ch ResNet input),
but replaces their noisy from-scratch MLP head with the closed-form ridge probe we use for ours.
=> apples-to-apples vs ours 0.668. Reports R2=1-MSE/var and standardized MSE."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/SSLForPDEs"))
import torch.nn as nn, torchvision.models as tvm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from utils import get_eval_loader_ns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA = os.path.expanduser("~/scratch/ns_data")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "results/ssl_logs/ssl_full/model.pth"


def probe(Ztr, ytr, Zte, yte, alpha=1.0):
    sc = StandardScaler().fit(Ztr); m, s = ytr.mean(), ytr.std() + 1e-8
    reg = Ridge(alpha).fit(sc.transform(Ztr), (ytr - m) / s)
    pred = reg.predict(sc.transform(Zte)); yte_s = (yte - m) / s
    return r2_score(yte_s, pred), float(np.mean((yte_s - pred) ** 2))


@torch.no_grad()
def embed(bb, loader):
    Z, Y = [], []
    for x, coeffs in loader:
        Z.append(bb(x.to(DEVICE)).cpu().numpy()); Y.append(np.asarray(coeffs).ravel())
    return np.concatenate(Z), np.concatenate(Y)


def main():
    ck = torch.load(CKPT, map_location=DEVICE); sd = ck["model"]
    bb = tvm.resnet18(weights=None); bb.fc = nn.Identity()
    bb.conv1 = nn.Conv2d(16 * 5, 64, 7, 2, 3, bias=False)
    bbsd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
    info = bb.load_state_dict(bbsd, strict=False); bb = bb.to(DEVICE).eval()
    print(f"loaded their backbone @ epoch {ck.get('epoch','?')}  (missing {len(info.missing_keys)}, unexpected {len(info.unexpected_keys)})", flush=True)
    vl = get_eval_loader_ns(DATA, 64, 4, mode="val", crop_size=(16, 128, 128))
    tl = get_eval_loader_ns(DATA, 64, 4, mode="test", crop_size=(16, 128, 128))
    Ztr, ytr = embed(bb, vl); Zte, yte = embed(bb, tl)
    print(f"embedded: val {Ztr.shape} ({len(set(np.round(ytr,4)))} buoy) / test {Zte.shape} ({len(set(np.round(yte,4)))} buoy)", flush=True)
    r2, mse = probe(Ztr, ytr, Zte, yte)
    print(f"\nTHEIRS VICReg ResNet-18 (RIDGE, valid->test, ep{ck.get('epoch','?')}): R2(buoy)={r2:+.3f}  MSE_std={mse:.3f}")
    print(f"  vs OURS FAE 0.668 (single-frame+sparse) and FLOOR -0.42.  NOTE theirs = 16-frame temporal + full field.")


if __name__ == "__main__":
    main()
