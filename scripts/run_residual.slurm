#!/bin/bash
#SBATCH --job-name=resid
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --output=logs/resid_%j.out
#SBATCH --error=logs/resid_%j.err
set -e
source "$(dirname "$0")/env.sh"
C=results/checkpoints/g1
python scripts/eval_residual.py --dt 4 --ckpt \
  $C/fjepa_vicreg_dt4_r6_tok_dt4_mc256_s0.pt \
  $C/fjepa_vicreg_dt4_r6_pool_dt4_mc256_s0.pt \
  $C/fjepa_vicreg_dt1_r6_tok_dt1_mc256_s0.pt
echo "DONE"
