#!/bin/bash
# Unified FAE: NP ELBO + anchor + independent-view symmetric-KL alignment
# + projected VICReg on posterior means. One model for invariance,
# linearization, uncertainty, and generation.
set -uo pipefail
cd "$(dirname "$0")/../.."
name=fae_np3_unified
python scripts/train_fae_np.py --gpu 1 --epochs 20 \
    --beta 1e-3 --anchor 1e-3 \
    --align_kl 1e-2 --proj_sim 25 --proj_std 25 --proj_cov 1 --beta_warmup_epochs 6 \
    --out results/checkpoints/g1/${name}.pt > logs/${name}.log 2>&1
EVAL_DEVICE=cuda:1 python scripts/eval_fae_np.py            --np_ckpt ${name}.pt > logs/${name}_eval.log 2>&1
EVAL_DEVICE=cuda:1 python scripts/eval_fae_np_ensemble.py   --np_ckpt ${name}.pt > logs/${name}_ensemble.log 2>&1
EVAL_DEVICE=cuda:1 python scripts/eval_fae_np_generation.py --np_ckpt ${name}.pt > logs/${name}_generation.log 2>&1
VIZ_DEVICE=cuda:1  python scripts/viz_fae_np_ensemble.py    --np_ckpt ${name}.pt > logs/${name}_viz.log 2>&1
touch logs/.np3_done
echo "np3 unified queue done $(date)"
