#!/bin/bash
#SBATCH --job-name=faep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=logs/faep_%j.out
#SBATCH --error=logs/faep_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
# args: MODE SEED TAG "EXTRA"
python scripts/train_fae_predict.py --mode $1 --seed $2 --epochs 80 --tag $3 $4
