"""Unified FAIR linear-probe head-to-head on NS buoyancy (the SSLForPDEs benchmark).

Embed valid (probe-train) + test (probe-test) NS clips with each FROZEN encoder, then the project's
standard ridge LINEAR probe of standardized buoyancy -> R2 and standardized-label MSE (trivial
predictor -> MSE ~ 1.0). Same data / same split / same probe across methods = apples-to-apples.

  ours   : our FAE checkpoint(s) (results/checkpoints/g1/faep_twoview_fae_ns_*.pt)
  theirs : VICReg ResNet-18 backbone (results/ssl_logs/<exp>/model.pth) -- best-effort; their own
           MLP-head number is also printed in their training log.
  floor  : ridge on raw per-clip channel mean+std (the trivial baseline, ~0.117 R2 expected).
"""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.data.ns import NSDataset
from src.models import FAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NTRAJ = 16   # trajectories/file for the probe set (denser than training's 8)


def probe(Ztr, ytr, Zte, yte, alpha=1.0):
    """ridge on standardized features; report R2 and standardized-label MSE (trivial -> ~1.0)."""
    sc = StandardScaler().fit(Ztr)
    m, s = ytr.mean(), ytr.std() + 1e-8
    reg = Ridge(alpha).fit(sc.transform(Ztr), (ytr - m) / s)
    pred = reg.predict(sc.transform(Zte)); yte_s = (yte - m) / s
    return r2_score(yte_s, pred), float(np.mean((yte_s - pred) ** 2))


@torch.no_grad()
def embed_fae(model, ds, coords, idx, batch=128):
    model.eval(); Z, Y = [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        fa = clip[:, :, 0].to(DEVICE)
        tok = model.encode_tokens(fields_to_tokens(fa, idx), coords[idx])
        Z.append(torch.cat([tok.mean(1), tok.std(1)], -1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y).ravel()


def channel_stats(ds):
    """trivial floor feature: per-clip frame-0 channel mean+std."""
    X, Y = [], []
    for clip, y in DataLoader(ds, batch_size=256):
        f0 = clip[:, :, 0]                                   # (B,C,H,W)
        X.append(torch.cat([f0.mean((2, 3)), f0.std((2, 3))], -1).numpy()); Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y).ravel()


def load_fae(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    a = ck["train_args"]; R = a["resolution"]; NPIX = R * R
    inc = a.get("in_chans") or (3 if a.get("dataset") in ("ns", "flowbench") else 4)
    model = FAE(emb_dim=a["emb_dim"], num_iter=a["num_iter"], depth_per_iter=4,
                num_latents=a["num_latents"], num_cross_heads=4, num_self_heads=8,
                n_freq=32, max_freq=32, coord_dim=2, in_chans=inc).to(DEVICE)
    model.load_state_dict(ck["model"])
    coords = make_coords_2d(n_side=R, device=DEVICE)
    g0 = torch.Generator(device=DEVICE).manual_seed(0)
    idx = torch.randperm(NPIX, generator=g0, device=DEVICE)[:1024]
    return model, coords, idx


def main():
    print(f"=== NS buoyancy: unified linear-probe head-to-head (valid->test, ridge, standardized) ===")
    print("loading NS valid(probe-train) + test(probe-test) clips ...", flush=True)
    va = NSDataset("valid", side=128, mode="clip", clip_len=2, frame_stride=4, n_traj=NTRAJ)
    te = NSDataset("test", side=128, mode="clip", clip_len=2, frame_stride=4, n_traj=NTRAJ, stats=va.stats)
    print(f"  probe-train {len(va)} clips ({len(set(np.round(va.labels,4)))} buoyancies), "
          f"probe-test {len(te)} clips ({len(set(np.round(te.labels,4)))} buoyancies)\n", flush=True)
    rows = []

    # FLOOR
    Xtr, ytr = channel_stats(va); Xte, yte = channel_stats(te)
    r2, mse = probe(Xtr, ytr, Xte, yte)
    rows.append(("FLOOR channel mean+std", r2, mse, "-"))

    # OURS
    for ck in sorted(glob.glob("results/checkpoints/g1/faep_*fae_ns*.pt")):
        try:
            model, coords, idx = load_fae(ck)
            Ztr, ytr = embed_fae(model, va, coords, idx); Zte, yte = embed_fae(model, te, coords, idx)
            r2, mse = probe(Ztr, ytr, Zte, yte)
            rows.append((f"OURS {os.path.basename(ck).replace('faep_','').replace('.pt','')}",
                         r2, mse, f"d={Ztr.shape[1]}"))
        except Exception as e:
            rows.append((f"OURS {os.path.basename(ck)} FAILED", float('nan'), float('nan'), str(e)[:40]))

    # THEIRS (best-effort)
    try:
        import importlib.util
        rows += theirs_rows(va, te)
    except Exception as e:
        rows.append(("THEIRS (see their training log)", float('nan'), float('nan'), str(e)[:40]))

    print(f"  {'method':32s} {'R2(buoy)':>9s} {'MSE_std':>9s}   note")
    print("  " + "-" * 70)
    for nm, r2, mse, note in rows:
        print(f"  {nm:32s} {r2:>+9.3f} {mse:>9.3f}   {note}")
    print("\n  (trivial predictor -> MSE_std ~1.0; their floor-vet earlier gave R2 0.117)")


def theirs_rows(va, te):
    """Load VICReg ResNet-18 backbone, embed via their NSDatasetEval pipeline. Best-effort."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/SSLForPDEs"))
    import torch.nn as nn, torchvision.models as tvm
    cks = sorted(glob.glob("results/ssl_logs/*/model.pth"))
    if not cks:
        raise FileNotFoundError("no their model.pth yet")
    out = []
    for ckp in cks:
        ck = torch.load(ckp, map_location=DEVICE)
        sd = ck.get("model", ck) if isinstance(ck, dict) else ck
        n_time = 16  # crop_t default
        bb = tvm.resnet18(weights=None); bb.fc = nn.Identity()
        c1 = bb.conv1
        bb.conv1 = nn.Conv2d(n_time * 5, c1.out_channels, c1.kernel_size, c1.stride, c1.padding, bias=False)
        bbsd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
        bb.load_state_dict(bbsd, strict=False); bb = bb.to(DEVICE).eval()
        out.append((f"THEIRS {os.path.basename(os.path.dirname(ckp))}", float('nan'), float('nan'),
                    "backbone loaded; needs their NSDatasetEval input -> use their log MSE"))
    return out


if __name__ == "__main__":
    main()
