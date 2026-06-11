#!/bin/bash
# FAE-NP v2 retrain queue — GPU 1 (~11-16h total)
# Run 1: hybrid — NP ELBO + VICReg var/cov on mu (targets the linear-probe gap)
# Run 2: stronger anchor at moderate beta (prior-quality ablation)
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results/checkpoints/g1

run_and_eval () {  # $1 = ckpt name (no .pt), rest = extra train args
  local name=$1; shift
  echo "=== [gpu1] TRAIN $name $(date) ==="
  python scripts/train_fae_np.py --gpu 1 --epochs 20 \
      --out results/checkpoints/g1/${name}.pt "$@" \
      > logs/${name}.log 2>&1
  echo "=== [gpu1] EVAL $name $(date) ==="
  EVAL_DEVICE=cuda:1 python scripts/eval_fae_np.py          --np_ckpt ${name}.pt > logs/${name}_eval.log 2>&1
  EVAL_DEVICE=cuda:1 python scripts/eval_fae_np_ensemble.py --np_ckpt ${name}.pt > logs/${name}_ensemble.log 2>&1
  EVAL_DEVICE=cuda:1 python scripts/eval_fae_np_generation.py --np_ckpt ${name}.pt > logs/${name}_generation.log 2>&1
}

run_and_eval fae_np2_b1e-3_a1e-3_vicmu --beta 1e-3 --anchor 1e-3 --vicreg_mu_std 10 --vicreg_mu_cov 1
run_and_eval fae_np2_b1e-3_a1e-2       --beta 1e-3 --anchor 1e-2

touch logs/.np2_gpu1_done
echo "=== [gpu1] queue done $(date) ==="
