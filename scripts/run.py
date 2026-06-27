"""The ONE runner — config in, fair result out.

  python scripts/run.py configs/ns/fae/default.yaml             # full campaign (all seeds, full epochs)
  python scripts/run.py configs/ns/mae/default.yaml --smoke     # 2-epoch / 1-seed / tiny-data smoke
  python scripts/run.py configs/ns/fae/ablation/FAE_no_dual_no_temp.yaml  # an ablation cell
  python scripts/run.py configs/ns/fae/test/sensor_sweep_disc_64.yaml     # a sensor-config variant

Each config's unique `tag` field names its own checkpoint -> setups NEVER overwrite each other.

For each seed: assert param budget -> train (the right trainer, ALL knobs from the config) ->
probe (src.eval, resolution-locked, RidgeCV, full-grid) -> log to results/experiments.csv with
provenance (git hash). Aggregates mean±std across seeds into one HEADLINE row.
"""
import os, sys, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from omegaconf import OmegaConf
from src.config import load_config, assert_budget, ckpt_path
from src import eval as ev
from src.utils.reslog import log_result

PY = sys.executable


def fae_cmd(cfg, seed, tag, ckpt):
    c = [PY, "scripts/train_fae.py", "--mode", cfg.mode, "--dataset", cfg.dataset,
         "--resolution", str(cfg.resolution), "--epochs", str(cfg.epochs), "--batch", str(cfg.batch),
         "--lr", str(cfg.lr), "--weight_decay", str(cfg.weight_decay), "--warmup_frac", str(cfg.warmup_frac),
         "--betas", str(cfg.betas[0]), str(cfg.betas[1]), "--frame_stride", str(cfg.frame_stride),
         "--emb_dim", str(cfg.emb_dim), "--num_latents", str(cfg.num_latents),
         "--num_iter", str(cfg.num_iter), "--depth_per_iter", str(cfg.depth_per_iter),
         "--n_traj", str(cfg.n_traj), "--dt_max", str(cfg.dt_max),
         "--sensor_pattern", cfg.get("sensor_pattern", "discrete"),
         "--n_seed", str(cfg.n_seed), "--n_query", str(cfg.n_query),                  # config-driven (no trainer defaults)
         "--num_cross_heads", str(cfg.num_cross_heads), "--num_self_heads", str(cfg.num_self_heads),
         "--n_freq", str(cfg.n_freq), "--max_freq", str(cfg.max_freq),
         "--pred_depth", str(cfg.pred_depth), "--dt_fixed", str(cfg.dt_fixed),
         "--seed", str(seed), "--tag", tag, "--ckpt_out", ckpt, "--save"]
    if cfg.get("mcnt_range"):
        c += ["--mcnt_range", str(cfg.mcnt_range[0]), str(cfg.mcnt_range[1])]
    elif cfg.get("mcnt"):
        c += ["--mcnt"] + [str(x) for x in cfg.mcnt]
    if cfg.get("res_h"):                                  # non-square FAE (native aspect; ViTs stay square)
        c += ["--res_h", str(cfg.res_h), "--res_w", str(cfg.res_w)]
    return c


def baseline_cmd(cfg, seed, tag, ckpt):
    meth = "mae" if cfg.method == "mae" else "ijepa"
    c = [PY, "scripts/train_baseline.py", "--method", meth, "--dataset", cfg.dataset,
         "--resolution", str(cfg.resolution), "--in_chans", str(cfg.in_chans), "--embed_dim", str(cfg.emb_dim),
         "--depth", str(cfg.depth), "--patch_size", str(cfg.patch_size), "--epochs", str(cfg.epochs),
         "--batch", str(cfg.batch), "--lr", str(cfg.lr), "--wd", str(cfg.weight_decay),
         "--betas", str(cfg.betas[0]), str(cfg.betas[1]), "--warmup_frac", str(cfg.warmup_frac),
         "--frame_stride", str(cfg.frame_stride), "--num_heads", str(cfg.num_heads),
         "--n_seed", str(cfg.n_seed), "--n_frames", str(cfg.n_frames),                # config-driven (no trainer defaults)
         "--n_ctx", str(cfg.n_ctx), "--n_tgt", str(cfg.n_tgt), "--ctx_frac", str(cfg.ctx_frac),
         "--tgt_frac", str(cfg.tgt_frac), "--tubelet", str(cfg.tubelet),
         "--ema_start", str(cfg.ema_start), "--ema_end", str(cfg.ema_end),
         "--n_traj", str(cfg.n_traj), "--seed", str(seed), "--tag", tag, "--ckpt_out", ckpt, "--amp"]
    if cfg.get("mask_ratio") is not None:
        c += ["--mask_ratio", str(cfg.mask_ratio)]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true", help="2 ep / seed 0 / tiny data — wiring test")
    ap.add_argument("--no_eval", action="store_true", help="train+save only; skip the coupled probe (probe separately)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg.epochs = 2; cfg.seeds = [0]; cfg.n_traj = 4
    obj = cfg.get("mode") or f"patch{cfg.get('patch_size')}"
    got = assert_budget(cfg)
    print(f"=== run [{cfg.method}/{obj}] res={cfg.resolution} ep={cfg.epochs} n_traj={cfg.n_traj} "
          f"seeds={list(cfg.seeds)} enc={got:.2f}M git={cfg.git} ===", flush=True)

    r2s = []
    for seed in cfg.seeds:
        tag = f"{cfg.tag}_s{seed}"        # unique per config -> no checkpoint collisions across setups
        ckpt = ckpt_path(cfg, seed)       # results/checkpoints/<dataset>/<method>/<tag>_s<seed>.pt
        cmd = (fae_cmd if cfg.method == "fae" else baseline_cmd)(cfg, seed, tag, ckpt)
        print(f"\n--- train {cfg.method} seed {seed} ---\n  " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if args.no_eval:
            print(f"  [seed {seed}] trained + saved {ckpt} (--no_eval: probe separately)", flush=True); continue
        r = ev.probe(cfg, ckpt)
        r2s.append(r["r2"])
        print(f"  [seed {seed}] R2={r['r2']:+.3f} MSE={r['mse']:.3f} floor={r['floor_r2']:+.3f} "
              f"PR={r['pr']:.1f} alpha={r['alpha']:.1e}", flush=True)
        log_result(tier="auto", category="probe", name=f"{cfg.method}_s{seed}", dataset=cfg.dataset,
                   res=cfg.resolution, objective=obj, emb_dim=cfg.emb_dim, epochs=cfg.epochs, seeds=1,
                   metric="R2_buoy", value=round(r["r2"], 4), floor=round(r["floor_r2"], 4),
                   ckpt=os.path.basename(ckpt), job=cfg.git, notes=f"PR={r['pr']:.1f}")

    if args.no_eval:
        print(f"\n=== {cfg.method} trained ({len(cfg.seeds)} seeds), eval skipped (--no_eval) ===", flush=True); return
    m, s = float(np.mean(r2s)), float(np.std(r2s))
    log_result(tier="HEADLINE", category="probe", name=cfg.method, dataset=cfg.dataset, res=cfg.resolution,
               objective=obj, emb_dim=cfg.emb_dim, epochs=cfg.epochs, seeds=len(cfg.seeds), metric="R2_buoy",
               value=round(m, 4), std=round(s, 4), floor=round(r["floor_r2"], 4), status="valid",
               notes=f"git={cfg.git} enc={got:.1f}M")
    print(f"\n=== {cfg.method} [{obj}] R2 = {m:+.3f} ± {s:.3f}  over {len(cfg.seeds)} seeds "
          f"(floor {r['floor_r2']:+.3f}) ===", flush=True)


if __name__ == "__main__":
    main()
