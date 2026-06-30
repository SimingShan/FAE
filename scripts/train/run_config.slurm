#!/bin/bash
# Generic launcher: sbatch -J <name> scripts/run_config.slurm <config.yaml>
# Runs the full campaign for ONE config (assert budget -> train all seeds -> probe -> CSV).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
echo "CONFIG=$1  host=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
python scripts/train/run.py "$1"
