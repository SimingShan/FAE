#!/bin/bash
#SBATCH --job-name=sensor_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/sensorsweep_%j.out
#SBATCH --error=logs/sensorsweep_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/sweep_eval_sensors.py
