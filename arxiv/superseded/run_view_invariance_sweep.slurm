#!/bin/bash
#SBATCH --job-name=view_inv_sw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viewinvsw_%j.out
#SBATCH --error=logs/viewinvsw_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/view_invariance_sweep.py
