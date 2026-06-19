"""Download the PDE-Arena NavierStokes-2D-conditioned dataset (the SSLForPDEs benchmark).

  python scripts/download_ns.py                 # full train+valid+test into $NS_DATA_ROOT (~77GB)
  python scripts/download_ns.py --splits train  # one split
  python scripts/download_ns.py --subset 8      # first 8 files/split (quick smoke / CI)

Verifies every file opens (HF transfers occasionally truncate) and re-fetches/removes broken ones,
so the on-disk dataset is always usable. Default dir: $NS_DATA_ROOT or ~/scratch/ns_data.

NOTE: this released HF dataset is a ~10x SUBSET of the paper's full pretraining set (26,624
trajectories). It is enough for the rollout/probe evals, but NOT for reproducing their VICReg
pretraining (which collapses on this little data). See docs/NS_EXPERIMENTS.md.
"""
import os, glob, argparse, h5py
from huggingface_hub import snapshot_download

REPO = "pdearena/NavierStokes-2D-conditoned"
NS = os.environ.get("NS_DATA_ROOT", os.path.expanduser("~/scratch/ns_data"))


def verify(root):
    bad = 0
    for f in sorted(glob.glob(f"{root}/*.h5")):
        try:
            with h5py.File(f, "r") as h:
                _ = list(h.keys())
        except Exception:
            bad += 1
            real = os.path.realpath(f)
            os.remove(f)
            if os.path.exists(real):
                os.remove(real)
            print(f"  removed truncated/corrupt: {os.path.basename(f)}", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"], choices=["train", "valid", "test"])
    ap.add_argument("--subset", type=int, default=0, help="0 = all; else first N files per split")
    ap.add_argument("--root", default=NS)
    ap.add_argument("--max_redownload", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)
    patterns = [f"*_{s}_*.h5" for s in args.splits] + ["*.yaml"]
    print(f"downloading {args.splits} -> {args.root}", flush=True)
    for attempt in range(args.max_redownload + 1):
        snapshot_download(REPO, repo_type="dataset", local_dir=args.root,
                          allow_patterns=patterns, max_workers=8)
        if verify(args.root) == 0:
            break
        print(f"  re-fetching after integrity failures (attempt {attempt+1})", flush=True)
    if args.subset:
        for s in args.splits:
            for f in sorted(glob.glob(f"{args.root}/*_{s}_*.h5"))[args.subset:]:
                os.remove(f)
    for s in args.splits:
        print(f"  {s}: {len(glob.glob(f'{args.root}/*_{s}_*.h5'))} files", flush=True)
    print(f"DONE -> {args.root}", flush=True)


if __name__ == "__main__":
    main()
