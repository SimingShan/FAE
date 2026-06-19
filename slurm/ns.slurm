#!/bin/bash
# Run any NS stage on SLURM:  sbatch slurm/ns.slurm <stage> [args...]
#   sbatch slurm/ns.slurm train-fae --seed 0
#   sbatch slurm/ns.slurm rollout --cond rep --seed 0
#SBATCH --job-name=ns
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ns_%j.out
#SBATCH --error=logs/ns_%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/run_ns.py "$@"
