#!/bin/bash
#SBATCH --job-name=sup
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sup_%j.out
#SBATCH --error=logs/sup_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
# args: SEED FRAMES TAG  (readout fixed meanstd, dt_max 4)
python scripts/train_supervised.py --seed $1 --frames $2 --dt_max 4 --readout meanstd --epochs 80 --tag $3
