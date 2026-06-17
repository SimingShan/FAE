#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
for c in "$@"; do echo "########## $c"; python scripts/diag_time_invariance.py --ckpt "$c"; done
