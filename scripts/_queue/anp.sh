#!/bin/bash
# FAE-ANP: dual-path (deterministic context tokens + global latent z),
# ANP-style — fixes the NP recon bottleneck exposed by the pooling diagnosis.
# Proven np2 settings (beta/anchor 1e-3), no projector/align terms: isolates
# the det-path effect.
set -uo pipefail
cd "$(dirname "$0")/../.."
name=fae_anp
python scripts/train_fae_np.py --gpu 1 --epochs 20 \
    --beta 1e-3 --anchor 1e-3 --det_path --det_drop 0.25 \
    --out results/checkpoints/g1/${name}.pt > logs/${name}.log 2>&1
EVAL_DEVICE=cuda:1 python scripts/eval_fae_np.py            --np_ckpt ${name}.pt > logs/${name}_eval.log 2>&1
EVAL_DEVICE=cuda:1 python scripts/eval_fae_np_ensemble.py   --np_ckpt ${name}.pt > logs/${name}_ensemble.log 2>&1
EVAL_DEVICE=cuda:1 python scripts/eval_fae_np_generation.py --np_ckpt ${name}.pt > logs/${name}_generation.log 2>&1
touch logs/.anp_done
echo "anp queue done $(date)"
