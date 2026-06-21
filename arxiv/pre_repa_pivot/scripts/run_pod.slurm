#!/bin/bash
#SBATCH --job-name=pod
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pod_%j.out
#SBATCH --error=logs/pod_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/eval_pod_rank.py
