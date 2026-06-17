#!/bin/bash
#SBATCH --job-name=attn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/attn_%j.out
#SBATCH --error=logs/attn_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/eval_attentive.py
