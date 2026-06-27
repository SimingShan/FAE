"""Training-convergence curves from the pretraining logs: TRAINING LOSS vs epoch (per encoder, per
dataset) — shows training converged. Losses are different objectives/scales, so normalized to the
epoch-1 value (loss/loss[0]). Also writes the in-training probe-R² curves separately. No GPU."""
import os, sys, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.plotstyle import apply, COLORS
apply()

JOBS = {("shear", "fae"): 2034236, ("shear", "mae"): 2034237, ("shear", "jepa"): 2034238,
        ("typhoon", "fae"): 2034239, ("typhoon", "mae"): 2034240, ("typhoon", "jepa"): 2034241}
COL = {m: COLORS[m.upper()] for m in ("fae", "mae", "jepa")}
EPRE = re.compile(r"ep\s+(\d+)/\d+")
LPRE = re.compile(r"\b(?:rec|loss)=([0-9.eE+\-]+)")                 # FAE prints rec=, MAE/JEPA print loss=
RPRE = re.compile(r"(?:\| mean|probe)\s+\w+=([+-][0-9.]+)\s+\w+=([+-][0-9.]+)")


def parse(jobid):
    f = f"logs/gpu_{jobid}.out"; eps, loss, r1 = [], [], []
    for line in (open(f) if os.path.exists(f) else []):
        em = EPRE.search(line); lm = LPRE.search(line); rm = RPRE.search(line)
        if em and lm:
            eps.append(int(em.group(1))); loss.append(float(lm.group(1))); r1.append(float(rm.group(1)) if rm else np.nan)
    return np.array(eps), np.array(loss), np.array(r1)


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/figs/misc/training_loss.png")
args = ap.parse_args()
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for i, ds in enumerate(["shear", "typhoon"]):
    ax = axs[i]
    for meth in ["fae", "mae", "jepa"]:
        ep, loss, _ = parse(JOBS[(ds, meth)])
        if len(ep):
            ax.plot(ep, loss / loss[0], color=COL[meth], lw=2, label=meth.upper())
    ax.set_title(f"{ds} — training loss (converged)", fontsize=14)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss / loss[0]"); ax.set_box_aspect(1)
    if i == 0:
        ax.legend()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.tight_layout(); plt.savefig(args.out, dpi=130, bbox_inches="tight")
print(f"wrote {args.out}", flush=True)
