#!/bin/bash
#SBATCH --job-name=gen_pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
# pilot: none, seed 0, full gen_epochs -> the in-loop spectrum_dist trace shows the plateau
python scripts/generate.py --config configs/ns/gen/none.yaml --seed 0 \
  --ckpt_out results/checkpoints/ns/gen/gen_none_uncond_s0.pt
