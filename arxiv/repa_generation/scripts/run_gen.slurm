#!/bin/bash
# REPA-gen campaign for ONE config: sbatch -J <name> scripts/run_gen.slurm <config.yaml> [--smoke]
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
echo "CONFIG=$1 $2  host=$(hostname)"
python scripts/run_gen.py "$1" $2
