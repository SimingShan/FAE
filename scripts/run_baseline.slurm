#!/bin/bash
#SBATCH --job-name=base
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/base_%j.out
#SBATCH --error=logs/base_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
# args: METHOD NFRAMES BATCH EPOCHS TAG
python scripts/train_baseline.py --method $1 --n_frames $2 --batch $3 --epochs $4 --amp --resolution 224 --tag $5 --seed $6
