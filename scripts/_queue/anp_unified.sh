#!/bin/bash
# Experiment (a) — FLAGSHIP NP candidate: ANP dual-path (fixes recon) +
# unified alignment terms (keeps z linearly organized / fixes probes).
# Union of the two stable runs: np3 stabilizers + ANP det-path.
set -uo pipefail
cd "$(dirname "$0")/../.."
name=fae_anp_unified
python scripts/train_fae_np.py --gpu 1 --epochs 20 \
    --det_path --det_drop 0.25 \
    --beta 1e-3 --anchor 1e-3 \
    --align_kl 1e-2 --proj_sim 25 --proj_std 25 --proj_cov 1 \
    --beta_warmup_epochs 6 \
    --out results/checkpoints/g1/${name}.pt > logs/${name}.log 2>&1
EVAL_DEVICE=cuda:1 python scripts/eval_fae_np.py            --np_ckpt ${name}.pt > logs/${name}_eval.log 2>&1
EVAL_DEVICE=cuda:1 python scripts/eval_fae_np_ensemble.py   --np_ckpt ${name}.pt > logs/${name}_ensemble.log 2>&1
EVAL_DEVICE=cuda:1 python scripts/eval_fae_np_generation.py --np_ckpt ${name}.pt > logs/${name}_generation.log 2>&1
touch logs/.anp_unified_done
echo "anp_unified queue done $(date)"
