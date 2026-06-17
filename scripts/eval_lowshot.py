"""Low-shot linear probe: fit ridge on k labeled frames PER (Re,Sc) combo, evaluate on the
full valid set. Reports R²(logRe), R²(Sc) vs k for ours (FAE), MAE, and the channel-mean+std
trivial floor (recomputed at each k). Motivated eval: a good rep exposes the parameter from
FEW labels — avoids the 11k-label 'supervised overkill'."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from src.models import FAE
from benchmarks.mae.mae import mae_physics
from src.data.well2d import ShearFlowSnapshotDataset, make_coords_2d, fields_to_tokens
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KS = [1, 3, 5, 10, 20, 50]


def t4(s):
    m, sd = s
    return (torch.tensor(np.asarray(m), dtype=torch.float32).view(1, 4, 1, 1),
            torch.tensor(np.asarray(sd), dtype=torch.float32).view(1, 4, 1, 1))


@torch.no_grad()
def extract(split, fae, mae, fm, fs, mm, ms, R, n_seed):
    st4 = (np.asarray(fm).reshape(1, 4, 1, 1), np.asarray(fs).reshape(1, 4, 1, 1))  # FAE stats -> 4D snapshot
    ds = ShearFlowSnapshotDataset(split, n_seed=n_seed, frame_stride=12, side=R, stats=st4)
    coords = make_coords_2d(R, DEVICE); g = torch.Generator().manual_seed(0)
    iA = torch.randperm(R * R, generator=g)[:512].to(DEVICE)
    Zf, Zm, Cf, lab = [], [], [], []
    for x, _ in DataLoader(ds, batch_size=128):
        x = x.to(DEVICE); phys = x * fs.to(DEVICE) + fm.to(DEVICE)
        Zf.append(fae.encode_tokens(fields_to_tokens(x, iA), coords[iA]).mean(1).cpu().numpy())
        xm = (phys - mm.to(DEVICE)) / ms.to(DEVICE)
        Zm.append(mae.encode(xm).cpu().numpy())
        Cf.append(torch.cat([phys.mean((2, 3)), phys.std((2, 3))], 1).cpu().numpy())  # channel mean+std floor
    Y = np.stack([np.asarray(ds.logRe), np.log10(np.asarray(ds.Sc))], 1)
    return np.concatenate(Zf), np.concatenate(Zm), np.concatenate(Cf), Y


def main():
    cko = torch.load("results/checkpoints/g1/faep_twoview_tvb_s0.pt", map_location="cpu", weights_only=False)
    R = cko["train_args"]["resolution"]
    fae = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128, num_cross_heads=4,
              num_self_heads=8, n_freq=32, max_freq=32, coord_dim=2, in_chans=4).to(DEVICE)
    fae.load_state_dict(cko["model"]); fae.eval(); fm, fs = t4(cko["stats"])
    ckm = torch.load("results/checkpoints/g1/mae_shear_mae_s1.pt", map_location="cpu", weights_only=False)
    mae = mae_physics(img_size=R).to(DEVICE); mae.load_state_dict(ckm["model"]); mae.eval(); mm, ms = t4(ckm["stats"])

    Zf, Zm, Cf, Ytr = extract("train", fae, mae, fm, fs, mm, ms, R, n_seed=6)
    Vf, Vm, Vc, Yva = extract("valid", fae, mae, fm, fs, mm, ms, R, n_seed=6)
    combo = np.array([hash((round(a, 3), round(b, 3))) for a, b in Ytr])
    ucombo = np.unique(combo)
    print(f"train {len(Ytr)} frames / {len(ucombo)} combos | valid {len(Yva)} frames")

    def probe(Xtr, Xva, j, idx):
        ym, ys = Ytr[idx, j].mean(), Ytr[idx, j].std() + 1e-8
        r = Ridge(alpha=10.0).fit(Xtr[idx], (Ytr[idx, j] - ym) / ys)
        return r2_score((Yva[:, j] - ym) / ys, r.predict(Xva))

    print(f"\n{'k/combo':8} {'OURS Re':9} {'MAE Re':9} {'floor Re':9} | {'OURS Sc':9} {'MAE Sc':9} {'floor Sc':9}")
    rng = np.random.default_rng(0)
    for k in KS:
        accf = {n: [[], []] for n in ("f", "m", "c")}
        for _ in range(5):                                   # 5 random low-shot draws
            idx = np.concatenate([rng.choice(np.where(combo == c)[0], min(k, (combo == c).sum()), replace=False) for c in ucombo])
            for j in (0, 1):
                accf["f"][j].append(probe(Zf, Vf, j, idx)); accf["m"][j].append(probe(Zm, Vm, j, idx)); accf["c"][j].append(probe(Cf, Vc, j, idx))
        m = {n: [np.mean(accf[n][j]) for j in (0, 1)] for n in accf}
        print(f"{k:<8} {m['f'][0]:<9.3f} {m['m'][0]:<9.3f} {m['c'][0]:<9.3f} | {m['f'][1]:<9.3f} {m['m'][1]:<9.3f} {m['c'][1]:<9.3f}")
    print("=> a good rep keeps Re high at small k. floor = channel mean+std (the real trivial baseline).")


if __name__ == "__main__":
    main()
