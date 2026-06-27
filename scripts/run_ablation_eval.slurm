#!/bin/bash
#SBATCH --job-name=ablation_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ablationeval_%j.out
#SBATCH --error=logs/ablationeval_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/ablation_eval.py
