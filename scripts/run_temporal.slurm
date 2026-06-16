#!/bin/bash
#SBATCH --job-name=fae_temporal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/temporal_%j.out
#SBATCH --error=logs/temporal_%j.err
# FAE spatio-temporal: 16-frame windows (paper context), coord_dim=3, res 224.
# The trainer runs its own 3D linear probe each eval (eval_linear_probe is
# single-frame-only, so it is NOT called here).
set -e
source "$(dirname "$0")/env.sh"
python scripts/train_fae_shear.py --temporal --n_frames 16 --epochs 100 \
    --batch 64 --n_seed 24 --tag align224t
