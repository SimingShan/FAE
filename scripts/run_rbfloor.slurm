#!/bin/bash
#SBATCH --job-name=rbfloor
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rbfloor_%j.out
#SBATCH --error=logs/rbfloor_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/eval_rb_floor.py
