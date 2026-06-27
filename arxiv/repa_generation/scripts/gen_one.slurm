#!/bin/bash
# Single (config, seed) gen run: sbatch -J <name> scripts/gen_one.slurm <config.yaml> <seed>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
python scripts/generate.py --config "$1" --seed "$2" \
  --ckpt_out "results/checkpoints/ns/gen/gen_$(basename "$1" .yaml)_uncond_s$2.pt"
