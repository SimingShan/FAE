#!/bin/bash
#SBATCH --job-name=fpo_pre
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/fpo_%x_%j.out
#SBATCH --error=logs/fpo_%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
"$@"
