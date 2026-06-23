#!/bin/bash
#SBATCH --job-name=roll_save
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/roll_save_%j.out
#SBATCH --error=logs/roll_save_%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
for m in time rep; do python scripts/rollout_ns.py --mode $m --n_traj_per_file 4 --epochs 15 --save --seed 0; done
echo ALLDONE
