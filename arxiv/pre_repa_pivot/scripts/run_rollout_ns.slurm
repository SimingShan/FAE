#!/bin/bash
#SBATCH --job-name=rollout_ns
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/rollout_all_%j.out
#SBATCH --error=logs/rollout_all_%j.out
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
for m in time buoyancy rep; do
  echo "########## MODE $m ##########"
  python scripts/rollout_ns.py --mode $m --n_traj_per_file 4 --epochs 20
done
echo "ALL ROLLOUT MODES DONE"
