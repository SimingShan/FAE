#!/bin/bash
#SBATCH --job-name=mae_tgt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/maetgt_%j.out
#SBATCH --error=logs/maetgt_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/train_baseline.py --method mae --n_frames 1 --batch 64 --epochs 80 --amp --resolution 224 --tag "$1" --seed 0 $2
