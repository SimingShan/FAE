"""Paper fig 4 — observation-invariance: ICC vs # sensors (log2 x-axis, 2^x) for the FAE ablation cells
(Senseiver / +temporal / +dual-view / full). Hypothesis: dual-view drives invariance. plotstyle.
Consumes results/sweeps/sensor_sweep_<ds>.json (icc section).
  python scripts/figs/fig4_icc_sensors.py --dataset typhoon
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
from src.plotstyle import apply, COLORS, NPG, panels
apply()

CCOL = {"Senseiver": NPG[5], "+temporal": NPG[3], "+dual-view": NPG[2], "full FAE": COLORS["FAE"]}
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
args = ap.parse_args()
J = json.load(open(f"results/sweeps/sensor_sweep_{args.dataset}.json"))
sens = np.array(J["sensors"])

fig, ax = panels(1, side=5.5)
for cell in ["Senseiver", "+temporal", "+dual-view", "full FAE"]:
    if cell in J.get("icc", {}):
        ax.plot(sens, J["icc"][cell], "-o", color=CCOL[cell], label=cell)
ax.set_xscale("log", base=2); ax.set_xticks(sens); ax.set_xticklabels([str(s) for s in sens])
ax.set_xlabel("# sensors"); ax.set_ylabel("invariance ICC"); ax.set_ylim(0, 1.02)
ax.set_title(f"{args.dataset}: observation-invariance vs sensors"); ax.legend()
out = f"results/figs/{args.dataset}/fig4_icc_sensors.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200); print(f"wrote {out}", flush=True)
