"""Clean post-hoc eval of the 2x2 anchored-FAE ablation (matched ep80 checkpoints).

Per arm reports: (1) SCALE-NORMALIZED convergence ‖z_n−z_dense‖/‖z_dense‖ (removes the magnitude
confound), (2) sparse-recon relL2 vs sensor count, (3) NS-buoyancy linear probe (valid->test, ridge,
standardized) + the channel floor. Decides whether anchoring / disc-consistency earn their keep, and
whether either changes the probe.
"""
import os, sys, glob, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIDE = 64; NPIX = SIDE * SIDE; COUNTS = [16, 32, 64, 128, 256, 512, 1024, 2048]


def load_arm(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE); a = ck["train_args"]; M = a["n_anchor"] ** 2
    if a.get("vanilla"):
        from src.models.fae import FAE
        m = FAE(emb_dim=320, num_latents=M, in_chans=3, coord_dim=2).to(DEVICE)
        enc, dec = (lambda u, c: m.encode_tokens(u, c)), (lambda z, c: m.decoder(z, c))
    else:
        from src.models.anchored_fae import AnchoredFAE
        m = AnchoredFAE(emb_dim=320, n_anchor_side=a["n_anchor"], in_chans=3).to(DEVICE)
        enc, dec = (lambda u, c: m.encode(u, c)), (lambda z, c: m.decode(z, c))
    m.load_state_dict(ck["model"]); m.eval()
    arm = ("vanilla " if a.get("vanilla") else "anchored") + ("+disc " if a["lam_disc"] > 0 else "+nodisc")
    return m, enc, dec, arm


@torch.no_grad()
def norm_conv(enc, x, coords, ref_n=4096):
    iref = torch.randperm(NPIX, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)[:ref_n]
    zref = enc(fields_to_tokens(x, iref), coords[iref]); zn = zref.flatten(1).norm(dim=1).clamp_min(1e-6)
    return [(n, ((enc(fields_to_tokens(x, (i := torch.randperm(NPIX, device=DEVICE)[:n])), coords[i]) - zref)
                 .flatten(1).norm(dim=1) / zn).mean().item()) for n in COUNTS]


@torch.no_grad()
def sparse_recon(enc, dec, x, coords, C=3):
    out = []
    for n in COUNTS:
        i = torch.randperm(NPIX, device=DEVICE)[:n]
        pred = dec(enc(fields_to_tokens(x, i), coords[i]), coords).reshape(x.size(0), SIDE, SIDE, C).permute(0, 3, 1, 2)
        out.append((n, (torch.linalg.norm((pred - x).flatten(1), dim=1) /
                        torch.linalg.norm(x.flatten(1), dim=1).clamp_min(1e-6)).mean().item()))
    return out


@torch.no_grad()
def embed(enc, ds, coords, idx, batch=128):
    Z, Y = [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        fa = clip[:, :, 0].to(DEVICE)
        tok = enc(fields_to_tokens(fa, idx), coords[idx])
        Z.append(torch.cat([tok.mean(1), tok.std(1)], -1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y).ravel()


def probe(Ztr, ytr, Zte, yte):
    sc = StandardScaler().fit(Ztr); m, s = ytr.mean(), ytr.std() + 1e-8
    r = Ridge(1.0).fit(sc.transform(Ztr), (ytr - m) / s)
    p = r.predict(sc.transform(Zte)); ys = (yte - m) / s
    return r2_score(ys, p), float(np.mean((ys - p) ** 2))


def main():
    coords = make_coords_2d(n_side=SIDE, device=DEVICE)
    va = NSDataset("valid", side=SIDE, mode="clip", clip_len=2, frame_stride=4, n_traj=16)
    te = NSDataset("test", side=SIDE, mode="clip", clip_len=2, frame_stride=4, n_traj=16, stats=va.stats)
    xb = torch.from_numpy(np.stack([va[i][0][:, 0].numpy() for i in range(64)])).to(DEVICE)   # 64 fields for curves
    idx = torch.randperm(NPIX, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)[:1024]
    # floor
    def chstats(ds):
        X, Y = [], []
        for clip, y in DataLoader(ds, batch_size=256):
            f0 = clip[:, :, 0]; X.append(torch.cat([f0.mean((2, 3)), f0.std((2, 3))], -1).numpy()); Y.append(y.numpy())
        return np.concatenate(X), np.concatenate(Y).ravel()
    fr2, fmse = probe(*chstats(va), *chstats(te))
    print(f"=== 2x2 anchored-FAE: convergence / sparse-recon / buoyancy probe ===", flush=True)
    print(f"FLOOR buoyancy R2={fr2:+.3f} MSE={fmse:.3f}\n", flush=True)
    for ck in sorted(glob.glob("results/checkpoints/g1/anchored_fae_*.pt")):
        try:
            m, enc, dec, arm = load_arm(ck)
            nc = norm_conv(enc, xb, coords); sr = sparse_recon(enc, dec, xb, coords)
            r2, mse = probe(*embed(enc, va, coords, idx), *embed(enc, te, coords, idx))
            print(f"[{arm}] probe R2={r2:+.3f} MSE={mse:.3f}", flush=True)
            print(f"   nconv  " + " ".join(f"{n}:{d:.3f}" for n, d in nc), flush=True)
            print(f"   sparse " + " ".join(f"{n}:{d:.3f}" for n, d in sr), flush=True)
        except Exception as e:
            print(f"[{os.path.basename(ck)}] FAIL {str(e)[:60]}", flush=True)


if __name__ == "__main__":
    main()
