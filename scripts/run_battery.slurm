#!/bin/bash
#SBATCH --job-name=batt
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/batt_%j.out
#SBATCH --error=logs/batt_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/eval_battery.py --ckpt results/checkpoints/g1/faep_twoview_tvb_s0.pt
