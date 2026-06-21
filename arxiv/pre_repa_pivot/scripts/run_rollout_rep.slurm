#!/bin/bash
#SBATCH --job-name=rollout_rep
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rollout_rep_%j.out
#SBATCH --error=logs/rollout_rep_%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/rollout_ns.py --mode rep --n_traj_per_file 4 --epochs 20
echo "REP DONE"
