#!/bin/bash
#SBATCH --job-name=fbcrop
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/fbcrop_%j.out
#SBATCH --error=logs/fbcrop_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/eval_flowbench_crop.py "$@"
