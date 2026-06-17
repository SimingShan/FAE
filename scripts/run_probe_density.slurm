#!/bin/bash
#SBATCH --job-name=pdens
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=logs/pdens_%j.out
#SBATCH --error=logs/pdens_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
echo "##### TWOVIEW (full train) #####"
python scripts/eval_probe_density.py --ckpt results/checkpoints/g1/faep_twoview_tvb_s0.pt --n_seed 24 --stride 4 2>/dev/null
echo "##### PURE RECON (full train) #####"
python scripts/eval_probe_density.py --ckpt results/checkpoints/g1/faep_recon_ronly_s0.pt --n_seed 24 --stride 4 2>/dev/null
echo DONE
