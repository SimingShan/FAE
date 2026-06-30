"""Parse train_fae logs -> plot rec-loss and buoyancy-probe vs epoch (global vs neighborhood, ns).
Answers: is the neighborhood objective still converging (needs more epochs) or plateaued lower?"""
import os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

A40 = "/tmp/claude-28660/-gpfs-radev-scratch-lu-lu-ss5235-WFAE/2da58c04-683b-4dd9-9d02-1575582e10cc/scratchpad"
RUNS = [   # (label, logpath, color)
    ("global  in=64  (recon_m64)",       "logs/reconm64_ns_2041066.out",   "#1f77b4"),
    ("global  in=1024 (recon_m1024)",    "logs/reconm1024_ns_2041067.out", "#2ca02c"),
    ("global  in=4096 (recon_m4096)",    "logs/reconm4096_ns_2041187.out", "#17becf"),
    ("nbhd    in=64  (recon_nbhd_m64)",  f"{A40}/nbhd_m64_a40.log",         "#d62728"),
    ("nbhd    in=1024 (recon_nbhd_m1024)","logs/nbhdm1024_ns_2041352.out",  "#ff7f0e"),
    ("nbhd    in=mixed (recon_nbhd)",    "logs/nbhd_ns_2041349.out",        "#9467bd"),
]
pat = re.compile(r"ep\s+(\d+)/\s*\d+\s+rec=([\d.eE+-]+).*?mean buoyancy=([+-][\d.]+)")

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
for label, path, col in RUNS:
    if not os.path.exists(path):
        print(f"skip (no log): {path}"); continue
    eps, rec, buo = [], [], []
    for ln in open(path):
        m = pat.search(ln)
        if m:
            eps.append(int(m.group(1))); rec.append(float(m.group(2))); buo.append(float(m.group(3)))
    if not eps:
        print(f"no probe lines: {path}"); continue
    ls = "--" if label.startswith("global") else "-"
    ax[0].plot(eps, rec, ls, c=col, marker="o", ms=3, label=label)
    ax[1].plot(eps, buo, ls, c=col, marker="o", ms=3, label=label)
    print(f"{label:34s} last ep{eps[-1]:>3d}  rec={rec[-1]:.3e}  buoyancy={buo[-1]:+.3f}")

ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train rec-loss (log)")
ax[0].set_title("Reconstruction loss vs epoch\n(scales differ: global=full-field, nbhd=local patch)", fontsize=10)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("buoyancy probe R2 (mean-pool, in-log @1024 sensors)")
ax[1].set_title("Probe vs epoch — is nbhd still climbing?", fontsize=10)
ax[1].axhline(0.88, c="gray", ls=":", lw=1, label="global recon final (0.88)")
for a in ax:
    a.legend(fontsize=7.5, loc="best"); a.grid(alpha=0.3)
plt.suptitle("ns: global vs neighborhood — loss & probe convergence (single-seed, in-log self-probe)", fontsize=12)
plt.tight_layout()
out = "results/figures/loss_curves_ns.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
