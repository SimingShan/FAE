#!/bin/bash
#SBATCH --job-name=nsfloor
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/nsfloor_%j.out
#SBATCH --error=logs/nsfloor_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/eval_ns_floor.py
