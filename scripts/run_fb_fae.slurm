#!/bin/bash
#SBATCH --job-name=fb_fae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/fbscr_%j.out
#SBATCH --error=logs/fbscr_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/train_fae_predict.py --dataset flowbench --mode "$1" --epochs "$2" \
  --batch "$3" --tag "$4" --resolution 224 --dt_max 2 --num_iter 4 --mcnt 256 512 --n_query 1024
