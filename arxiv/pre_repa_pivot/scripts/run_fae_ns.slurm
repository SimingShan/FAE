#!/bin/bash
#SBATCH --job-name=fae_ns
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/faens_%j.out
#SBATCH --error=logs/faens_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/train_fae_predict.py --dataset ns --resolution 128 --mode twoview \
  --dt_max 2 --dt_fixed 2 --frame_stride 4 --num_iter 4 --mcnt 512 1024 --n_query 2048 \
  --epochs "${2:-80}" --tag "${1:-fae_ns}" --save $3
