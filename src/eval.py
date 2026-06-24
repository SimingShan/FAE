"""Config-driven NS-buoyancy linear probe (valid->test, RidgeCV, standardized).

SIDE, sensor count, and ridge alphas all come from the merged config, so the probe can NEVER drift
from the training resolution (the SIDE=64-at-128 bug). Same mean+std pooling for FAE and the ViTs.
"""
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ridge(Ztr, ytr, Zte, yte, alphas):
    sc = StandardScaler().fit(Ztr); m, s = ytr.mean(), ytr.std() + 1e-8
    reg = RidgeCV(alphas=alphas).fit(sc.transform(Ztr), (ytr - m) / s)
    p = reg.predict(sc.transform(Zte)); ys = (yte - m) / s
    return float(r2_score(ys, p)), float(np.mean((ys - p) ** 2)), float(reg.alpha_)


def _pr(Z):
    Z = Z - Z.mean(0); e = np.clip(np.linalg.eigvalsh(np.cov(Z.T)), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-30))


@torch.no_grad()
def _embed_fae(model, ds, coords, idx, batch=32):
    Z, Y = [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        fa = clip[:, :, 0].to(DEVICE)
        tok = model.encode_tokens(fields_to_tokens(fa, idx), coords[idx])
        Z.append(torch.cat([tok.mean(1), tok.std(1)], -1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y).ravel()


@torch.no_grad()
def _embed_vit(m, method, ds, batch=128):
    Z, Y = [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        x = clip[:, :, 0].to(DEVICE)
        tok = m.forward_encoder(x, 0.0)[0][:, 1:] if method == "mae" else m.target(x)
        Z.append(torch.cat([tok.mean(1), tok.std(1)], -1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y).ravel()


def _chstats(ds):
    X, Y = [], []
    for clip, y in DataLoader(ds, batch_size=256):
        f0 = clip[:, :, 0]
        X.append(torch.cat([f0.mean((2, 3)), f0.std((2, 3))], -1).numpy()); Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y).ravel()


def probe(cfg, ckpt_path):
    """Returns dict(r2, mse, floor_r2, pr, alpha, n_train, n_test). Resolution-locked to cfg.resolution."""
    SIDE = cfg.resolution; NPIX = SIDE * SIDE; alphas = list(cfg.ridge_alphas)
    va = NSDataset("valid", side=SIDE, mode="clip", clip_len=2, frame_stride=cfg.frame_stride, n_traj=cfg.eval_n_traj)
    te = NSDataset("test", side=SIDE, mode="clip", clip_len=2, frame_stride=cfg.frame_stride, n_traj=cfg.eval_n_traj, stats=va.stats)
    fr2, fmse, _ = _ridge(*_chstats(va), *_chstats(te), alphas)            # trivial floor

    ck = torch.load(ckpt_path, map_location=DEVICE); a = ck.get("train_args", {}); method = a.get("method")
    if method in ("mae", "ijepa"):
        from scripts.train_baseline import build_model
        m = build_model("mae" if method == "mae" else "ijepa", resolution=SIDE, in_chans=cfg.in_chans,
                        embed_dim=a.get("embed_dim"), depth=a.get("depth"), patch_size=a.get("patch_size")).to(DEVICE)
        m.load_state_dict(ck["model"]); m.eval()
        meth = "mae" if method == "mae" else "jepa"
        Ztr, ytr = _embed_vit(m, meth, va); Zte, yte = _embed_vit(m, meth, te)
    else:                                                                   # FAE
        from src.models.fae import FAE
        m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), num_latents=a["num_latents"],
                in_chans=cfg.in_chans, coord_dim=2).to(DEVICE)
        m.load_state_dict(ck["model"]); m.eval()
        coords = make_coords_2d(n_side=SIDE, device=DEVICE)
        if cfg.eval_fae_full_grid:
            idx = torch.arange(NPIX, device=DEVICE)                         # full grid -> matched info w/ ViTs
        else:
            g0 = torch.Generator(device=DEVICE).manual_seed(0)
            idx = torch.randperm(NPIX, generator=g0, device=DEVICE)[:cfg.eval_n_sensors]
        Ztr, ytr = _embed_fae(m, va, coords, idx); Zte, yte = _embed_fae(m, te, coords, idx)

    r2, mse, alpha = _ridge(Ztr, ytr, Zte, yte, alphas)
    return dict(r2=r2, mse=mse, floor_r2=fr2, pr=_pr(Ztr), alpha=alpha, n_train=len(ytr), n_test=len(yte))
