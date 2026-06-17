#!/bin/bash
#SBATCH --job-name=surfcd
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/surfcd_%j.out
#SBATCH --error=logs/surfcd_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/eval_flowbench_surfcd.py
