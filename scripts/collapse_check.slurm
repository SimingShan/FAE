#!/bin/bash
#SBATCH --job-name=collapse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/collapse_%j.out
#SBATCH --error=logs/collapse_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/collapse_check.py
