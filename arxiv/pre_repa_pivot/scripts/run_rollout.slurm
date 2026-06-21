#!/bin/bash
#SBATCH --job-name=rollout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/rollout_%j.out
#SBATCH --error=logs/rollout_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/run_rollout.py
