#!/bin/bash
#SBATCH --job-name=sf_matrix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/sf_%A_%a.out
#SBATCH --error=logs/sf_%A_%a.err
# Matched single-frame matrix: array index -> method, each trains (batch 256,
# 100 epochs, n_seed 24, res 224) then self-evals with the frozen linear probe.
# Same dataset + batch + epochs => matched optimizer steps (Hard-Rule #4).
set -e
source "$(dirname "$0")/env.sh"

TAG=align224
METHODS=(fae ae mae ijepa)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "[$(date)] === method=$M (task $SLURM_ARRAY_TASK_ID) ==="

if [ "$M" = "fae" ]; then
    python scripts/train_fae_shear.py --epochs 100 --batch 256 --n_seed 24 --tag $TAG
    python scripts/eval_linear_probe.py --method fae \
        --ckpt results/checkpoints/g1/fae_vicreg_shear_${TAG}.pt --n_seed 24
else
    python scripts/train_baseline.py --method $M --epochs 100 --batch 256 \
        --n_seed 24 --workers 8 --tag $TAG
    python scripts/eval_linear_probe.py --method $M \
        --ckpt results/checkpoints/g1/${M}_shear_${TAG}.pt --n_seed 24
fi
echo "[$(date)] === done $M ==="
