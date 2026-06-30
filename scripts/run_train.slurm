#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Generic single-run launcher: sbatch --time=H:MM:SS -J <name> scripts/run_train.slurm <pyscript> <args...>
cd /gpfs/radev/scratch/lu_lu/ss5235/WFAE
exec /gpfs/radev/project/lu_lu/ss5235/envs/wfae/bin/python -u "$@"
