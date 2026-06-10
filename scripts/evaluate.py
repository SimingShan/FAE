"""Evaluate all trained G1 methods on held-out data.

For every checkpoint found in results/checkpoints/g1/ (see src/models/zoo.py):
  - per-system coefficient linear probe (heat nu, advection beta, burgers nu,
    reaction_diffusion D)
  - 4-class PDE classification (LogReg, kNN-15, advection-vs-rest F1)
  - consistency under partial observation (sparse-input methods only;
    dense methods are structurally unable to do this and get n/a)

Outputs:
  results/probes/g1/g1_all.json          full metrics
  results/probes/g1/emb_<method>.npz     val embeddings + class labels
  results/probes/g1/{probe_bars, tsne_grid, consistency}.png
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.models import zoo
from src.data.g1 import (load_g1_system, PDE_NAMES, PDE_CLASS, COEFF_NAME,
                           make_coords_1d, train_val_split)
from src.metrics import (probe_all_coefficients, classification_metrics,
                          random_baseline_probe,
                          two_subset_agreement, variance_across_subsets)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "results", "checkpoints", "g1")
OUT  = os.path.join(ROOT, "results", "probes", "g1")
os.makedirs(OUT, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Consistency is only meaningful for natively sparse-input methods. Dense
# methods (cnn, mae, jepa_vit) get n/a — do NOT fake it with zero-fill.
SPARSE_TIER = {"fae_recon", "fae_vicreg", "fae_spatiotemporal", "mlp", "jepa_perceiver"}

COLORS = {
    "fae_recon":          "#1f77b4",
    "fae_vicreg":         "#0d690d",
    "fae_spatiotemporal": "#2ca02c",
    "mlp":                "#ff7f0e",
    "cnn":                "#9467bd",
    "mae":                "#e377c2",
    "jepa_perceiver":     "#17becf",
    "jepa_vit":           "#8c564b",
}
SYSTEM_LABELS = ["Heat", "Advection", "Burgers", "Reaction-Diff (AC)"]


def encode_factory(model, kind, full_coords):
    """encode_fn(u_traj, sensor_idx, seed) -> (N, D); mid-frame per trajectory."""
    @torch.no_grad()
    def encode_fn(u_traj, sensor_idx=None, seed=0, batch: int = 64):
        N_traj, T, X = u_traj.shape
        mid = T // 2
        idx = None
        if sensor_idx is not None:
            idx = torch.as_tensor(sensor_idx, device=device, dtype=torch.long)
        out = []
        for i0 in range(0, N_traj, batch):
            x = torch.from_numpy(u_traj[i0:i0+batch, mid]).to(device).float()
            out.append(zoo.encode(model, kind, x, full_coords, idx=idx).cpu())
        return torch.cat(out, 0).numpy()
    return encode_fn


def main():
    t0 = time.time()
    print("Loading G1 val data (per system, shuffle-then-split)...", flush=True)
    val_per_system = {}
    for n in PDE_NAMES:
        d = load_g1_system(n)
        s = train_val_split(d["u"], d["coeff"], val_frac=0.2, seed=0)
        val_per_system[n] = {"u": s["u_val"], "coeff": s["coeff_val"],
                              "train_u": s["u_train"], "train_coeff": s["coeff_train"],
                              "pde_class": d["pde_class"]}
    print(f"  loaded {[(n, v['u'].shape) for n, v in val_per_system.items()]}", flush=True)

    train_u = np.concatenate([v["train_u"] for v in val_per_system.values()], axis=0)
    val_u   = np.concatenate([v["u"]       for v in val_per_system.values()], axis=0)
    train_cls = np.concatenate([np.full(v["train_u"].shape[0], v["pde_class"], dtype=np.int64)
                                  for v in val_per_system.values()])
    val_cls   = np.concatenate([np.full(v["u"].shape[0], v["pde_class"], dtype=np.int64)
                                  for v in val_per_system.values()])
    print(f"  train: {train_u.shape}, val: {val_u.shape}", flush=True)

    full_coords = make_coords_1d(device, N=val_u.shape[-1])
    n_pix = full_coords.shape[0]

    results = {}
    for spec in zoo.METHODS:
        print(f"\n=== {spec.name} ===", flush=True)
        model, _ = zoo.load_method(spec.name, CKPT, device)
        if model is None:
            print("  SKIP (no checkpoint)")
            continue
        tier = "sparse" if spec.name in SPARSE_TIER else "dense"
        n_par = sum(p.numel() for p in model.parameters())
        print(f"  loaded ({n_par/1e6:.2f}M params, {tier})", flush=True)
        encode_fn = encode_factory(model, spec.kind, full_coords)

        Z_train = encode_fn(train_u)
        Z_val   = encode_fn(val_u)
        print(f"  Z_train={Z_train.shape}  Z_val={Z_val.shape}", flush=True)

        method_res = {"label": spec.label, "params_M": n_par/1e6, "tier": tier}

        # Per-system coefficient probes
        for system in PDE_NAMES:
            key = COEFF_NAME[system]
            n_train = val_per_system[system]["train_u"].shape[0]
            n_val_s = val_per_system[system]["u"].shape[0]
            start_tr = sum(val_per_system[n]["train_u"].shape[0]
                            for n in PDE_NAMES[:PDE_CLASS[system]])
            start_va = sum(val_per_system[n]["u"].shape[0]
                            for n in PDE_NAMES[:PDE_CLASS[system]])
            Ztr_s = Z_train[start_tr:start_tr + n_train]
            Zva_s = Z_val[start_va:start_va + n_val_s]
            ctr = {key: val_per_system[system]["train_coeff"]}
            cva = {key: val_per_system[system]["coeff"]}
            r = probe_all_coefficients(Ztr_s, ctr, Zva_s, cva, coeffs=[key])
            method_res[f"probe_{system}_{key}"] = r[key]

        # 4-class PDE classification
        try:
            method_res["classification"] = classification_metrics(
                Z_train, train_cls, Z_val, val_cls)
        except Exception as e:
            print(f"  classification error: {e}", flush=True)
            method_res["classification"] = {"error": str(e)}

        # Consistency (sparse tier only)
        if tier == "sparse":
            u_sample = val_u[:50]
            _, _, cos_avg, l2_avg = two_subset_agreement(
                encode_fn, u_sample, n_pix=n_pix, n_sensors=128,
                seed_pair=(0, 1), device=device)
            method_res["consistency"] = {"cos_sim": cos_avg, "l2_diff": l2_avg,
                                          "n_sensors": 128}
            var = variance_across_subsets(encode_fn, u_sample, n_pix=n_pix,
                                            n_sensors=128, n_subsets=5)
            method_res["variance_across_subsets_mean"] = float(var.mean())
        else:
            method_res["consistency"] = None
            method_res["variance_across_subsets_mean"] = None

        results[spec.name] = method_res
        np.savez(os.path.join(OUT, f"emb_{spec.name}.npz"),
                  Z_train=Z_train, Z_val=Z_val,
                  train_cls=train_cls, val_cls=val_cls)
        print(f"  done; metrics: {method_res}", flush=True)
        del model
        torch.cuda.empty_cache()

    # Random-feature baseline (probe floor)
    print("\n=== random baseline ===", flush=True)
    n_train_h = val_per_system["heat"]["train_u"].shape[0]
    n_val_h = val_per_system["heat"]["u"].shape[0]
    rb = random_baseline_probe(
        {"nu": val_per_system["heat"]["train_coeff"]},
        {"nu": val_per_system["heat"]["coeff"]},
        n_train=n_train_h, n_val=n_val_h, dim=320, seed=42, coeffs=["nu"])
    results["_random_baseline"] = {"probe_heat_nu": rb["nu"], "tier": "baseline"}
    print(f"  random R² (heat nu): {rb['nu']:.3f}", flush=True)

    json.dump(results, open(os.path.join(OUT, "g1_all.json"), "w"), indent=2)
    print(f"\nsaved {OUT}/g1_all.json", flush=True)

    plot_probe_bars(results)
    plot_tsne_grid(results)
    plot_consistency(results)
    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)


def plot_probe_bars(results):
    methods = [m.name for m in zoo.METHODS if m.name in results]
    metrics = ["probe_heat_nu", "probe_advection_beta", "probe_burgers_nu",
                "probe_reaction_diffusion_D",
                "classification.logreg", "classification.knn", "classification.adv_f1"]
    metric_names = ["Heat ν", "Adv β", "Burg ν", "AC D",
                     "LogReg 4-class", "kNN-15", "Adv F1"]

    def get(method, key):
        r = results[method]
        if "." in key:
            a, b = key.split(".")
            return r.get(a, {}).get(b, np.nan) if isinstance(r.get(a), dict) else np.nan
        return r.get(key, np.nan)

    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.85 / max(len(methods), 1)
    x = np.arange(len(metrics))
    for i, m in enumerate(methods):
        vals = [get(m, k) for k in metrics]
        ax.bar(x + i*width - 0.4, vals, width, label=results[m]["label"],
                color=COLORS.get(m, "gray"))
    if "_random_baseline" in results:
        v = results["_random_baseline"]["probe_heat_nu"]
        ax.axhline(v, color="red", linestyle="--", alpha=0.5,
                    label=f"random baseline ({v:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=10, rotation=15)
    ax.set_ylabel("score")
    ax.set_title("G1 multi-PDE: probe + classification metrics", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(min(-0.1, ax.get_ylim()[0]), 1.05)
    plt.tight_layout()
    out = os.path.join(OUT, "probe_bars.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}", flush=True)


def plot_tsne_grid(results):
    methods = [m.name for m in zoo.METHODS if m.name in results]
    n = len(methods)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.5))
    if n == 1: axes = [axes]
    pal = ["tab:red", "tab:blue", "tab:green", "tab:gray"]
    for ax, m in zip(axes, methods):
        emb_path = os.path.join(OUT, f"emb_{m}.npz")
        if not os.path.exists(emb_path): continue
        d = np.load(emb_path)
        Z = d["Z_val"]
        cls = d["val_cls"]
        try:
            pca = PCA(n_components=min(50, Z.shape[1]))
            Z50 = pca.fit_transform(Z)
            tsne = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(Z50)
        except Exception as e:
            ax.text(0.5, 0.5, f"err: {e}", ha="center", va="center",
                      transform=ax.transAxes)
            continue
        for cid in range(4):
            mask = cls == cid
            ax.scatter(tsne[mask, 0], tsne[mask, 1], s=14, c=pal[cid],
                        label=SYSTEM_LABELS[cid], alpha=0.6, edgecolors="none")
        d_res = results[m]
        r = d_res.get("classification", {})
        title = f"{d_res['label']}\n"
        title += f"R² heat={d_res.get('probe_heat_nu', float('nan')):.2f} "
        title += f"adv={d_res.get('probe_advection_beta', float('nan')):.2f} "
        title += f"burg={d_res.get('probe_burgers_nu', float('nan')):.2f}\n"
        if r and "logreg" in r:
            title += f"LR={r.get('logreg',0):.2f}  kNN={r.get('knn',0):.2f}  AdvF1={r.get('adv_f1',0):.2f}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if ax is axes[0]:
            ax.legend(fontsize=9, markerscale=1.5, loc="best")
    plt.tight_layout()
    out = os.path.join(OUT, "tsne_grid.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}", flush=True)


def plot_consistency(results):
    methods = [m.name for m in zoo.METHODS
                if m.name in results and results[m.name]["tier"] == "sparse"]
    if not methods: return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = [results[m]["label"] for m in methods]
    cos = [results[m].get("consistency", {}).get("cos_sim", 0) for m in methods]
    var = [results[m].get("variance_across_subsets_mean", 0) for m in methods]
    axes[0].bar(names, cos, color=[COLORS.get(m, "gray") for m in methods])
    axes[0].set_ylabel("cos similarity")
    axes[0].set_title("Two-subset latent agreement (M=128, higher = more consistent)")
    axes[0].set_ylim(-0.1, 1.05); axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(names, var, color=[COLORS.get(m, "gray") for m in methods])
    axes[1].set_ylabel("mean variance across 5 subsets")
    axes[1].set_title("Variance across 5 sensor subsets (M=128, lower = more consistent)")
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(OUT, "consistency.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
