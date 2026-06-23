"""Append experiment results to a local CSV (results/experiments.csv) — lossless auto-capture.
The human curates/promotes/flags rows manually (results/ is gitignored, so it stays local).

Usage in an eval script:
    from src.utils.reslog import log_result
    log_result(category="gen_uncond", name="fae", dataset="ns", metric="spectrum_dist",
               value=0.276, std=0.005, seeds=3, n_samples=2048, lam=0.5, status="valid")
Never raises (a logging failure must not kill an eval).
"""
import os, csv, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "results", "experiments.csv")
FIELDS = ["tier", "date", "category", "name", "dataset", "res", "objective", "emb_dim", "patch", "lam",
          "epochs", "seeds", "n_samples", "metric", "value", "std", "floor", "ckpt", "job", "status", "notes"]


def log_result(**kw):
    try:
        kw.setdefault("date", datetime.date.today().isoformat())
        kw.setdefault("tier", "auto")                          # auto-captured; human promotes to HEADLINE/valid/WASTED
        kw.setdefault("status", "auto")
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        new = not os.path.exists(PATH)
        with open(PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: kw.get(k, "") for k in FIELDS})
    except Exception as e:
        print(f"  [reslog] skipped ({str(e)[:50]})", flush=True)
