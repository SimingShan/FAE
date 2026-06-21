"""Trivial-floor vet for PDE-Arena NavierStokes-2D-conditioned BUOYANCY (SSLForPDEs benchmark).
Each file = one buoyancy value (constant within file). Download ~24 files spanning the buoyancy
range, read u/vx/vy at a developed frame for a few trajectories, fit channel-mean/std + random-proj
to predict buoyancy (combo-split by file = unseen buoyancy). Low floor => valid hard target."""
import re, os, numpy as np, h5py
from huggingface_hub import HfFileSystem, hf_hub_download
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
REPO = "pdearena/NavierStokes-2D-conditoned"
NS = os.path.expanduser("~/scratch/ns_data")
FRAME, TRAJ, NPICK = 40, [0, 8, 16, 24], 24


def buoyancy(fn): return float(re.search(r'_train_\d+_([0-9.]+)_32', fn).group(1))


def main():
    fs = HfFileSystem()
    allf = sorted((f.split("/")[-1] for f in fs.ls("datasets/" + REPO, detail=False)
                   if "_train_" in f and f.endswith(".h5")), key=buoyancy)
    pick = allf[::max(1, len(allf) // NPICK)][:NPICK]
    print(f"{len(allf)} train files; vetting {len(pick)} spanning buoyancy {buoyancy(pick[0]):.3f}-{buoyancy(pick[-1]):.3f}")
    Xcm, Xcms, Xrp, B, fid = [], [], [], [], []
    rng = np.random.default_rng(0); P = None
    for fi, fn in enumerate(pick):
        lp = hf_hub_download(REPO, fn, repo_type="dataset", local_dir=NS)
        with h5py.File(lp) as h:
            buo = float(h["train/buo_y"][0])
            for tj in TRAJ:
                snap = np.stack([h["train/u"][tj, FRAME], h["train/vx"][tj, FRAME], h["train/vy"][tj, FRAME]], -1).astype(np.float32)
                cm = snap.mean((0, 1)); cs = snap.std((0, 1))
                Xcm.append(cm); Xcms.append(np.concatenate([cm, cs]))
                flat = snap[::8, ::8].reshape(-1)
                if P is None: P = rng.standard_normal((flat.size, 128)).astype(np.float32)
                Xrp.append(flat @ P); B.append(buo); fid.append(fi)
        os.remove(lp)                                              # free disk after reading
    Xcm, Xcms, Xrp, B, fid = map(np.array, (Xcm, Xcms, Xrp, B, fid))
    print(f"{len(B)} samples, {len(set(np.round(B,4)))} distinct buoyancies")
    cperm = rng.permutation(len(pick)); tc = set(cperm[:max(5, len(pick) // 5)].tolist())
    te = np.array([c in tc for c in fid]); tr = ~te
    def floor(X, nm):
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8; Xs = (X - m) / s
        print(f"  TRIVIAL {nm:16s} R2(buoyancy)={r2_score(B[te], Ridge(1.0).fit(Xs[tr], B[tr]).predict(Xs[te])):+.3f}")
    print("=== buoyancy trivial floor (combo-split by file) ===")
    floor(Xcm, "channel-mean"); floor(Xcms, "channel mean+std"); floor(Xrp, "random-proj(128)")
    print("=> low floor => buoyancy is a HARD target (valid SSLForPDEs head-to-head benchmark).")


if __name__ == "__main__":
    main()
