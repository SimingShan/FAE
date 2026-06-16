#!/bin/bash
#SBATCH --job-name=fae_tmode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=logs/faetm_%j.out
#SBATCH --error=logs/faetm_%j.err
# FAE-temporal time-mode / objective ablation.
# args: TIME_MODE  SIM  STD  COV  TAG
#   coord recon-only:     coord     0 0 0    t_coord_recon
#   aggregate recon-only: aggregate 0 0 0    t_agg_recon
#   coord + vicreg:       coord     5 100 5  t_coord_vicreg
# batch 32 (matched across all; safe for aggregate's per-frame B*T encode).
set -e
source "$(dirname "$0")/env.sh"
echo "[$(date)] time_mode=$1 sim=$2 std=$3 cov=$4 tag=$5"
python scripts/train_fae_shear.py --temporal --n_frames 16 --time_mode "$1" \
    --sim "$2" --std "$3" --cov "$4" --resolution 224 \
    --epochs 100 --batch 32 --n_seed 24 --tag "$5"
echo "[$(date)] done $5"
