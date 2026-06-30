"""Re-render every line figure from SAVED data/logs with the uniform paper style (src/plotstyle.py).
CPU-only (no model, no GPU): parses training/eval logs + JSON, re-emits the figures. Re-run any time
the style changes. Figures: convergence, sensor_sweep, view_invariance_sweep, view_invariance(bars).
"""
import os, re, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
ROOT = "/gpfs/radev/scratch/lu_lu/ss5235/WFAE"; sys.path.insert(0, ROOT)
from src.plotstyle import apply, COLORS, panels
apply()
OUT = os.path.join(ROOT, "results/figures")
def latest(pat):
    f = sorted(glob.glob(os.path.join(ROOT, pat))); return f[-1] if f else None


# ---------- 1. convergence (parse the 200-ep training logs) ----------
def conv():
    # logs hold all 3 seeds (ep 1..200 repeated) -> group by epoch and average -> one clean curve/method.
    logs = {"FAE": latest("logs/run_fae_*.out"), "MAE": latest("logs/run_mae_*.out"), "JEPA": latest("logs/run_jepa_*.out")}
    rx = re.compile(r"ep\s+(\d+)/\d+\s+(?:rec|loss)=([0-9.eE+-]+).*?buoyancy=([+\-0-9.]+)")
    fig, (a0, a1) = panels(2)
    for name, lp in logs.items():
        if not lp: continue
        R, L = {}, {}
        for ln in open(lp):
            m = rx.match(ln)
            if m:
                e = int(m[1]); L.setdefault(e, []).append(float(m[2])); R.setdefault(e, []).append(float(m[3]))
        ep = sorted(R)
        rm = np.array([np.mean(R[e]) for e in ep]); rs = np.array([np.std(R[e]) for e in ep])
        lm = np.array([np.mean(L[e]) for e in ep])
        a0.plot(ep, rm, "-o", color=COLORS[name], label=name, markersize=4)
        a0.fill_between(ep, rm - rs, rm + rs, color=COLORS[name], alpha=0.15, lw=0)
        a1.semilogy(ep, lm, "-o", color=COLORS[name], label=name, markersize=4)
    a0.set(xlabel="epoch", ylabel="probe $R^2$ (buoyancy)", title="Probe $R^2$ vs epoch"); a0.set_ylim(-0.05, 1.0)
    a1.set(xlabel="epoch", ylabel="training loss", title="Training loss vs epoch")
    a0.legend(); a1.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "convergence.png")); print("convergence.png (3-seed mean±std)")


# ---------- 2. sensor_sweep (from saved JSON) ----------
def sweep():
    d = json.load(open(os.path.join(OUT, "sensor_sweep.json"))); res, floor = d["res"], d["floor"]
    counts = [64, 128, 256, 512, 1024]; FULL = 128 * 128
    mean = [np.mean(res[str(c)]) for c in counts]; std = [np.std(res[str(c)]) for c in counts]
    fm, fs = np.mean(res["full"]), np.std(res["full"])
    fig, ax = panels(1, side=5.5)
    ax.errorbar(counts, mean, yerr=std, fmt="-o", color=COLORS["FAE"], capsize=3, label="FAE (sparse sensors)")
    ax.errorbar([FULL], [fm], yerr=[fs], fmt="D", color=COLORS["FAE"], markersize=9, capsize=3, label=f"FAE (full grid) {fm:.3f}")
    ax.scatter([FULL], [0.905], marker="s", s=80, color=COLORS["MAE"], zorder=5, label="MAE (full grid only) 0.905")
    ax.scatter([FULL], [0.656], marker="^", s=80, color=COLORS["JEPA"], zorder=5, label="JEPA (full grid only) 0.656")
    ax.axhline(floor, color=COLORS["floor"], ls=":", label=f"trivial floor {floor:.2f}")
    ax.set_xscale("log", base=2); ax.set_xticks(counts + [FULL]); ax.set_xticklabels([str(c) for c in counts] + ["full"])
    ax.set(xlabel="# sensors fed to encoder", ylabel="buoyancy probe $R^2$", title="Eval-time sensor sweep")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "sensor_sweep.png")); print("sensor_sweep.png")


# ---------- 3. view_invariance_sweep (parse ICC sweep log) ----------
def inv_sweep():
    lp = latest("logs/viewinvsw_*.out");  rx = re.compile(r"seed\d\s+(\w+)\s+budget=\s*(\d+)\s+ICC=([0-9.]+)\s+same=([0-9.]+)")
    cur = {}
    for ln in open(lp):
        m = rx.search(ln)
        if m:
            n, b = m[1], int(m[2]); cur.setdefault(n, {}).setdefault(b, {"icc": [], "same": []})
            cur[n][b]["icc"].append(float(m[3])); cur[n][b]["same"].append(float(m[4]))
    fig, (a0, a1) = panels(2)
    for name in ["FAE", "MAE", "JEPA"]:
        bs = sorted(cur.get(name, {}))
        if not bs: continue
        icm = [np.mean(cur[name][b]["icc"]) for b in bs]; ics = [np.std(cur[name][b]["icc"]) for b in bs]
        sam = [np.mean(cur[name][b]["same"]) for b in bs]
        a0.errorbar(bs, icm, yerr=ics, fmt="-o" if name == "FAE" else "--s", color=COLORS[name], capsize=3, label=name)
        a1.plot(bs, sam, "-o" if name == "FAE" else "--s", color=COLORS[name], label=name)
    for a, yl, ttl in [(a0, "ICC (between / (between+within))", "Invariance vs budget"),
                       (a1, "same-field cosine", "Same-field cosine vs budget")]:
        a.set_xscale("log", base=2); a.set_xlabel("# sensors / observed pixels"); a.set_ylabel(yl); a.set_title(ttl); a.legend()
    a0.set_ylim(0, 1.03)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "view_invariance_sweep.png")); print("view_invariance_sweep.png")


# ---------- 4. view_invariance (bars, restyled for uniformity) ----------
def inv_bar():
    lp = latest("logs/viewinv_2*.out"); rx = re.compile(r"seed\d\s+(\w+)\s+same=([0-9.]+)\s+diff=([0-9.]+)\s+gap=\S+\s+ICC=([0-9.]+)")
    r = {n: {"same": [], "diff": [], "icc": []} for n in ["FAE", "MAE", "JEPA"]}
    for ln in open(lp):
        m = rx.search(ln)
        if m and m[1] in r:
            r[m[1]]["same"].append(float(m[2])); r[m[1]]["diff"].append(float(m[3])); r[m[1]]["icc"].append(float(m[4]))
    names = ["FAE", "MAE", "JEPA"]; x = np.arange(3)
    fig, (a0, a1) = panels(2)
    sm = [np.mean(r[n]["same"]) for n in names]; dm = [np.mean(r[n]["diff"]) for n in names]
    a0.bar(x - 0.2, sm, 0.4, color=[COLORS[n] for n in names], label="same field")
    a0.bar(x + 0.2, dm, 0.4, color=[COLORS[n] for n in names], alpha=0.45, label="different fields")
    a0.set_xticks(x); a0.set_xticklabels(names); a0.set_ylabel("cosine similarity"); a0.set_title("View-invariance @ 25%"); a0.legend()
    im = [np.mean(r[n]["icc"]) for n in names]; ist = [np.std(r[n]["icc"]) for n in names]
    a1.bar(x, im, 0.55, yerr=ist, capsize=4, color=[COLORS[n] for n in names])
    for i in range(3): a1.text(i, im[i] + ist[i] + 0.01, f"{im[i]:.3f}", ha="center", fontsize=13)
    a1.set_xticks(x); a1.set_xticklabels(names); a1.set_ylim(0, 1.05); a1.set_ylabel("invariance ratio (ICC)"); a1.set_title("Variance that is field, not view")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "view_invariance.png")); print("view_invariance.png")


for f in (conv, sweep, inv_sweep, inv_bar):
    try: f()
    except Exception as e: print(f"SKIP {f.__name__}: {e}")
