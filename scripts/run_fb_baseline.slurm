#!/bin/bash
#SBATCH --job-name=fb_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/fbscr_%j.out
#SBATCH --error=logs/fbscr_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/train_baseline.py --dataset flowbench --method "$1" --epochs "$2" \
  --batch "$3" --n_frames "${4:-1}" --tag "$5" --resolution 224 --amp --workers 4
