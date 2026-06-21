#!/bin/bash
#SBATCH --job-name=videomae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/videomae_%j.out
#SBATCH --error=logs/videomae_%j.err
# Video MAE pretrained FROM SCRATCH (tube-mask 0.9) on 16-frame clips, frozen
# LINEAR probe — the fair temporal pixel baseline, same pipeline/probe as
# FAE-temporal / AE / MAE / I-JEPA. 256G for the 16-frame window load.
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/train_baseline.py --method videomae --epochs 100 --batch 64 \
    --mask_ratio 0.9 --resolution 224 --n_seed 24 --workers 8 --tag align224t
