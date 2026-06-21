#!/bin/bash
#SBATCH --job-name=floor224
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/floor_%j.out
#SBATCH --error=logs/floor_%j.err
# The gate (Hard-Rule #1): trivial linear-probe floor at the aligned config
# (224 res, raw Schmidt). Re-measures the now-stale 128²/log-Sc floor. CPU only.
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/eval_linear_probe.py --method trivial --resolution 224 --n_seed 24
