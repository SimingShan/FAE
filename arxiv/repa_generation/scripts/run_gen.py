"""REPA-generation runner — config in, fair gen result out (the gen analogue of run.py).

  python scripts/run_gen.py configs/ns/gen/fae.yaml            # full campaign (all seeds)
  python scripts/run_gen.py configs/ns/gen/none.yaml --smoke   # 2-ep / seed-0 wiring test

For each seed: assert the frozen alignment encoder EXISTS (no silent fallback) -> train the SiT
(generate.py, ALL knobs from the config) -> read the authoritative n_samples metric stored in the
checkpoint -> aggregate mean±std across seeds -> HEADLINE row in results/experiments.csv.
spectrum_dist is NOISY (hard rule #3) so the ranking is ONLY trustworthy multi-seed + many-sample.
"""
import os, sys, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src.config import load_config, ckpt_file
from src.utils.reslog import log_result

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENC_TAG = lambda cfg: {"fae": ("fae", cfg.fae_tag), "fae_set": ("fae", cfg.fae_tag), "fae_set2": ("fae", cfg.fae_tag),
                       "mae": ("mae", cfg.mae_tag), "jepa": ("jepa", cfg.jepa_tag)}


def gen_ckpt(cfg, seed):
    return os.path.join(cfg.ckpt_dir, cfg.dataset, "gen", f"{cfg.tag}_{cfg.mode}_s{seed}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true", help="2 ep / seed 0 / tiny data — wiring test")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if cfg.align != "none":                                    # assert the alignment encoder exists — no fallback
        m, tag = ENC_TAG(cfg)[cfg.align]
        enc = ckpt_file(m, tag, 0)
        assert os.path.exists(enc), f"alignment encoder missing: {enc} (train it first)"
    seeds = [0] if args.smoke else list(cfg.seeds)
    print(f"=== run_gen [{cfg.tag}] mode={cfg.mode} align={cfg.align} res={cfg.resolution} "
          f"sit={cfg.sit_size} ep={cfg.gen_epochs} n_samples={cfg.n_samples} seeds={seeds} git={cfg.git} ===", flush=True)

    primary = "recon_relL2" if cfg.mode == "sparse" else "spectrum_dist"
    vals = {}
    for seed in seeds:
        ckpt = gen_ckpt(cfg, seed)
        cmd = [PY, "scripts/generate.py", "--config", args.config, "--seed", str(seed), "--ckpt_out", ckpt]
        if args.smoke:
            cmd.append("--smoke")
        print(f"\n--- gen seed {seed} ---\n  " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        m = torch.load(ckpt, map_location="cpu")["metric"]
        for k, v in m.items():
            vals.setdefault(k, []).append(v)
        print(f"  [seed {seed}] {m}", flush=True)
        if args.smoke:                                        # wiring test -> never pollute the curated CSV
            continue
        log_result(category="gen", name=f"{cfg.align}_s{seed}", dataset=cfg.dataset, res=cfg.resolution,
                   objective=f"REPA-{cfg.mode}", lam=cfg.lam, epochs=cfg.gen_epochs, seeds=1,
                   n_samples=cfg.n_samples, metric=primary, value=round(m[primary], 4),
                   ckpt=os.path.basename(ckpt), job=cfg.git, notes=f"sit={cfg.sit_size}")

    if args.smoke:
        print("\n=== SMOKE OK (no CSV row written) ===", flush=True); return
    arr = vals[primary]; mn, sd = float(np.mean(arr)), float(np.std(arr))
    log_result(tier="HEADLINE", category="gen", name=f"REPA-{cfg.align}", dataset=cfg.dataset,
               res=cfg.resolution, objective=f"REPA-{cfg.mode}", lam=cfg.lam, epochs=cfg.gen_epochs,
               seeds=len(seeds), n_samples=cfg.n_samples, metric=primary, value=round(mn, 4),
               std=round(sd, 4), status="valid", notes=f"git={cfg.git} sit={cfg.sit_size}")
    print(f"\n=== {cfg.align} [{cfg.mode}] {primary} = {mn:.4f} ± {sd:.4f}  over {len(seeds)} seeds ===", flush=True)


if __name__ == "__main__":
    main()
