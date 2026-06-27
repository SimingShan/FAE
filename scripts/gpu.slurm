#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gpu_%j.out
#SBATCH --error=logs/gpu_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python "$@"
