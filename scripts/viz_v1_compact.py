"""One compact summary plot of all 5 v1 methods."""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results/probes/g1/g1_all.json")) as f:
    R = json.load(f)

methods = ["fae_recon", "fae_vicreg", "mlp", "cnn", "mae"]
labels = {"fae_recon":"V3-recon", "fae_vicreg":"V3+VICReg", "mlp":"MLP",
          "cnn":"CNN", "mae":"MAE"}
colors = {"fae_recon":"#1f77b4", "fae_vicreg":"#0d690d", "mlp":"#ff7f0e",
          "cnn":"#9467bd", "mae":"#e377c2"}

def get(m, k):
    r = R.get(m, {})
    if "." in k:
        a, b = k.split(".")
        v = r.get(a, {})
        return v.get(b, np.nan) if isinstance(v, dict) else np.nan
    return r.get(k, np.nan)

metrics = ["probe_heat_nu", "probe_burgers_nu",
           "classification.logreg", "classification.knn",
           "classification.adv_f1", "consistency.cos_sim"]
mnames = ["Heat ν R²", "Burg ν R²", "LogReg", "kNN-15", "Adv F1", "Cons. cos"]

fig, ax = plt.subplots(figsize=(11, 4.5))
width = 0.16; x = np.arange(len(metrics))
for i, m in enumerate(methods):
    vals = [get(m, k) for k in metrics]
    vals = [0 if v is None else v for v in vals]
    bars = ax.bar(x + i*width - 0.4, vals, width, color=colors[m], label=labels[m])
ax.axhline(0, color="gray", linewidth=0.5)
ax.axhline(-0.019, color="red", linestyle="--", alpha=0.5,
           label="random baseline")
ax.set_xticks(x); ax.set_xticklabels(mnames, fontsize=10)
ax.set_ylabel("score"); ax.set_ylim(-0.1, 1.05)
ax.set_title("G1 (1D 4-PDE) — all 5 methods @ ~7M params", fontsize=12)
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="lower right", fontsize=9, ncol=2)
plt.tight_layout()
out = os.path.join(ROOT, "results/probes/g1/summary_v1.png")
fig.savefig(out, dpi=90, bbox_inches="tight")  # low dpi for tiny file
print(f"saved {out}", flush=True)
print(f"size: {os.path.getsize(out)/1024:.0f} KB")
