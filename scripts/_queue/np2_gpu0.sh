#!/bin/bash
# FAE-NP v2 retrain queue — GPU 0 (~11-16h total)
# Run 1: main fixed model (sigmoid sigma + anchor, moderate regularization)
# Run 2: strong regularization (generation-oriented)
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results/checkpoints/g1

run_and_eval () {  # $1 = ckpt name (no .pt), rest = extra train args
  local name=$1; shift
  echo "=== [gpu0] TRAIN $name $(date) ==="
  python scripts/train_fae_np.py --gpu 0 --epochs 20 \
      --out results/checkpoints/g1/${name}.pt "$@" \
      > logs/${name}.log 2>&1
  echo "=== [gpu0] EVAL $name $(date) ==="
  EVAL_DEVICE=cuda:0 python scripts/eval_fae_np.py          --np_ckpt ${name}.pt > logs/${name}_eval.log 2>&1
  EVAL_DEVICE=cuda:0 python scripts/eval_fae_np_ensemble.py --np_ckpt ${name}.pt > logs/${name}_ensemble.log 2>&1
  EVAL_DEVICE=cuda:0 python scripts/eval_fae_np_generation.py --np_ckpt ${name}.pt > logs/${name}_generation.log 2>&1
}

run_and_eval fae_np2_b1e-3_a1e-3 --beta 1e-3 --anchor 1e-3
run_and_eval fae_np2_b1e-2_a1e-2 --beta 1e-2 --anchor 1e-2

touch logs/.np2_gpu0_done
echo "=== [gpu0] queue done $(date) ==="
