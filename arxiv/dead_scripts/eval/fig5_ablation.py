"""Paper fig 5 — ablation study per dataset: probe-R2 for the 4 FAE cells (Senseiver / +temporal /
+dual-view / full) + MAE + JEPA, vs the trivial floor. Consumes results/probes/<ds>.json. plotstyle.
  python scripts/figs/fig5_ablation.py --dataset typhoon
"""
import os, sys, json, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
from src.plotstyle import apply, COLORS, NPG, panels
apply()

MODE2CELL = {"recon": "Senseiver", "recon_both": "+temporal", "twoview_present": "+dual-view", "twoview": "full FAE"}
CCOL = {"Senseiver": NPG[5], "+temporal": NPG[3], "+dual-view": NPG[2], "full FAE": COLORS["FAE"],
        "MAE": COLORS["MAE"], "JEPA": COLORS["JEPA"]}
ORDER = ["Senseiver", "+temporal", "+dual-view", "full FAE", "MAE", "JEPA"]

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
args = ap.parse_args()
J = json.load(open(f"results/probes/{args.dataset}.json"))
tgt = J["targets"][0]

vals = defaultdict(list)                                          # cell/method -> [r2 over seeds]
for e in J["encoders"]:
    if "r2" not in e or e["r2"][0] != e["r2"][0]:                # skip nan/failed
        continue
    key = MODE2CELL.get(e.get("mode")) if e["method"] == "fae" else e["method"].upper()
    if key: vals[key].append(e["r2"][0])
means = [np.mean(vals[c]) if vals[c] else np.nan for c in ORDER]

fig, ax = panels(1, side=6.0)
bars = ax.bar(range(len(ORDER)), means, color=[CCOL[c] for c in ORDER], width=0.72,
              edgecolor="white", linewidth=0.8)
flo = J["floor"][0]; ymax = np.nanmax(means); ylo = min(0.0, flo) - 0.02; yhi = ymax * 1.24
ax.set_ylim(ylo, yhi)
ax.axhline(flo, ls="--", color="0.4", lw=1.5)                     # floor annotated on the line itself
ax.text(len(ORDER) - 0.45, flo + 0.01 * (yhi - ylo), f"floor {flo:+.2f}", fontsize=10, color="0.4", ha="right", va="bottom")
for i in range(len(ORDER)):                                      # value labels above bars
    if not np.isnan(means[i]):
        ax.text(i, means[i] + 0.02 * (yhi - ylo), f"{means[i]:.2f}", ha="center", va="bottom", fontsize=11)
ax.axvline(3.5, color="0.85", lw=1.2)                            # divider: FAE cells | grid ViTs
ax.text(1.5, ymax * 1.15, "FAE cells", ha="center", fontsize=11, color="0.5")
ax.text(4.5, ymax * 1.15, "grid ViTs", ha="center", fontsize=11, color="0.5")
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=28, ha="right", fontsize=12)
ax.set_ylabel(f"probe R²  ({tgt})"); ax.set_box_aspect(0.72)
ax.set_title(f"{args.dataset} — ablation  ({J['split']})")
out = f"results/figs/{args.dataset}/fig5_ablation.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=200)
print(f"wrote {out}  ({dict((c, round(np.mean(vals[c]),3)) for c in ORDER if vals[c])})", flush=True)
