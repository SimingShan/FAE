"""Paper fig 2 — training convergence: LEFT training loss (loss/loss[0]) vs epoch, RIGHT in-loop probe-R2
vs epoch, for FAE/MAE/JEPA. Auto-discovers each method's log for the dataset. plotstyle.
  python scripts/figs/fig2_training.py --dataset typhoon
"""
import os, sys, re, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
from src.plotstyle import apply, COLORS, panels
apply()

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
args = ap.parse_args()
EPRE = re.compile(r"ep\s+(\d+)/\d+")
LOSS = re.compile(r"\b(?:rec|loss)=([0-9.eE+\-]+)")
PROBE = re.compile(r"(?:\|\s*mean|probe)\s+\w+=([+\-][0-9.]+)")  # FAE prints '| mean X=', baselines print 'probe X='
DS = re.compile(r"dataset=(\w+)")
METH = re.compile(r"=== FAE-|method[=\s'\"]+(mae|ijepa)|\b(mae|ijepa)\b")


def method_of(text):
    if "FAE-" in text or "encode_tokens" in text: return "fae"
    if re.search(r"\bmae\b|method=mae", text): return "mae"
    if re.search(r"ijepa|jepa", text): return "jepa"
    return None


def parse(path):
    txt = open(path, errors="ignore").read()
    m = DS.search(txt)
    if not m or m.group(1) != args.dataset: return None
    meth = method_of(txt[:4000])
    eps, loss, pr = [], [], []
    for ln in txt.splitlines():
        e = EPRE.search(ln); l = LOSS.search(ln)
        if e and l:
            ei = int(e.group(1))
            if eps and ei <= eps[-1]: break          # multi-seed log -> stop at first seed boundary (ep resets to 1)
            eps.append(ei); loss.append(float(l.group(1)))
            p = PROBE.search(ln); pr.append(float(p.group(1)) if p else np.nan)
    return meth, np.array(eps), np.array(loss), np.array(pr)


runs = {}                                                        # method -> (ep, loss, probe), keep longest
for f in glob.glob("logs/gpu_*.out") + glob.glob("logs/run_*.out"):
    r = parse(f)
    if not r or r[0] is None or len(r[1]) == 0: continue
    meth, ep, loss, pr = r
    if meth not in runs or len(ep) > len(runs[meth][0]):
        runs[meth] = (ep, loss, pr)

apply(); fig, (a0, a1) = panels(2, side=5.0)
for meth in ["fae", "mae", "jepa"]:
    if meth not in runs: continue
    ep, loss, pr = runs[meth]; c = COLORS[meth.upper()]
    a0.plot(ep, loss / loss[0], color=c, lw=2.2, label=meth.upper())
    m = ~np.isnan(pr)
    if m.any(): a1.plot(ep[m], pr[m], color=c, lw=2.2, marker="o", ms=4, label=meth.upper())
a0.set_xlabel("epoch"); a0.set_ylabel("training loss  (loss / loss$_0$)"); a0.set_title("convergence")
a1.set_xlabel("epoch"); a1.set_ylabel("probe R²  (mean-pool)"); a1.set_title("representation quality")
a0.legend(); a1.legend()
fig.suptitle(f"{args.dataset} — training", fontsize=15)
out = f"results/figs/{args.dataset}/fig2_training.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=200)
print(f"wrote {out}  (methods: {list(runs)})", flush=True)
