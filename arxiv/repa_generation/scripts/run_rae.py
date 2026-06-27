"""RAE runner — config in, two-axis (rFID + gFID) result out (the RAE analogue of run_gen.py).

  python scripts/run_rae.py configs/ns/rae/fae.yaml            # full campaign (all seeds)
  python scripts/run_rae.py configs/ns/rae/fae.yaml --smoke    # tiny wiring test

Per seed: assert encoder exists -> stage1 (train decoder) -> stage2 (latent DiT) -> rae_eval (rFID+gFID)
-> aggregate mean±std -> HEADLINE rows. RAE diffuses IN the frozen latent, so this is the test that
actually USES the representation (unlike REPA). gen_spectrum_dist is NOISY -> trust only multi-seed.
"""
import os, sys, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src.config import load_config, ckpt_file
from src.rae_eval import rae_eval, ENC_TAG
from src.utils.reslog import log_result

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def stage_ckpt(cfg, stage, seed):
    return os.path.join(cfg.ckpt_dir, cfg.dataset, "rae", f"{cfg.tag}_{stage}_s{seed}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="pilot ONE seed at full epochs (no HEADLINE row)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    enc = cfg.encoder
    assert os.path.exists(ckpt_file(enc, cfg[ENC_TAG[enc]], 0)), f"frozen encoder missing for {enc}"
    pilot = args.seed is not None
    seeds = [0] if args.smoke else [args.seed] if pilot else list(cfg.seeds)
    smk = ["--smoke"] if args.smoke else []
    print(f"=== run_rae [{cfg.tag}] enc={enc} res={cfg.resolution} s1ep={cfg.s1_epochs} s2ep={cfg.s2_epochs} "
          f"seeds={seeds} git={cfg.git} ===", flush=True)

    rels, gsds = [], []
    for seed in seeds:
        s1 = stage_ckpt(cfg, "stage1", seed); s2 = stage_ckpt(cfg, "stage2", seed)
        c1 = [PY, "scripts/train_rae_stage1.py", "--config", args.config, "--seed", str(seed), "--ckpt_out", s1] + smk
        c2 = [PY, "scripts/train_rae_stage2.py", "--config", args.config, "--seed", str(seed), "--ckpt_out", s2, "--stage1_ckpt", s1] + smk
        print(f"\n--- seed {seed}: stage1 ---\n  " + " ".join(c1), flush=True); subprocess.run(c1, check=True, cwd=ROOT)
        print(f"--- seed {seed}: stage2 ---\n  " + " ".join(c2), flush=True); subprocess.run(c2, check=True, cwd=ROOT)
        m = rae_eval(cfg, s1, s2, DEV)
        rels.append(m["recon_relL2"]); gsds.append(m["gen_spectrum_dist"])
        print(f"  [seed {seed}] rFID(relL2={m['recon_relL2']:.4f} sd={m['recon_spectrum_dist']:.4f})  gFID(sd={m['gen_spectrum_dist']:.4f})", flush=True)
        if args.smoke or pilot:
            continue
        log_result(category="rae", name=f"{enc}_s{seed}", dataset=cfg.dataset, res=cfg.resolution,
                   objective="RAE recon", epochs=cfg.s1_epochs, seeds=1, metric="recon_relL2",
                   value=round(m["recon_relL2"], 4), ckpt=os.path.basename(s1), job=cfg.git)
        log_result(category="rae", name=f"{enc}_s{seed}", dataset=cfg.dataset, res=cfg.resolution,
                   objective="RAE gen", epochs=cfg.s2_epochs, seeds=1, n_samples=cfg.n_samples,
                   metric="gen_spectrum_dist", value=round(m["gen_spectrum_dist"], 4), ckpt=os.path.basename(s2), job=cfg.git)

    if args.smoke:
        print("\n=== SMOKE OK (no CSV row) ===", flush=True); return
    if pilot:
        print(f"\n=== PILOT {enc} seed {args.seed}: rFID relL2={rels[0]:.4f}  gFID sd={gsds[0]:.4f} ===", flush=True); return
    for metric, arr in [("recon_relL2", rels), ("gen_spectrum_dist", gsds)]:
        mn, sd = float(np.mean(arr)), float(np.std(arr))
        log_result(tier="HEADLINE", category="rae", name=f"RAE-{enc}", dataset=cfg.dataset, res=cfg.resolution,
                   objective="RAE", epochs=cfg.s2_epochs, seeds=len(seeds), n_samples=cfg.n_samples,
                   metric=metric, value=round(mn, 4), std=round(sd, 4), status="valid", notes=f"git={cfg.git}")
        print(f"=== RAE-{enc}  {metric} = {mn:.4f} ± {sd:.4f}  over {len(seeds)} seeds ===", flush=True)


if __name__ == "__main__":
    main()
