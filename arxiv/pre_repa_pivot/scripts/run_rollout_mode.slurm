#!/bin/bash
#SBATCH --job-name=rollout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rollout_%x_%j.out
#SBATCH --error=logs/rollout_%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/rollout_ns.py --mode "$1" --n_traj_per_file 4 --epochs 20
echo "ROLLOUT $1 DONE"
