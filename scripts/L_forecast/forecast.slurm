#!/bin/bash
#SBATCH --job-name=forecast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/forecast_%j.out
#SBATCH --error=logs/forecast_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/train_forecast.py "$@"
