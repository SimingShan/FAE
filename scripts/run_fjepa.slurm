#!/bin/bash
#SBATCH --job-name=fjepa
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fjepa_%j.out
#SBATCH --error=logs/fjepa_%j.err
# args: SEED EPOCHS "EXTRA-FLAGS" TAG   (EXTRA is the full flag bundle, e.g.
#   "--loss vicreg --dt_max 4 --pred_type mlp --pred_depth 2 --sim 25 --std 25 --cov 1 --mcnt 256 512")
set -e
source "$(dirname "$0")/env.sh"
echo "[$(date)] seed=$1 epochs=$2 tag=$4 extra: $3"
python scripts/train_fjepa.py --seed "$1" --epochs "$2" $3 \
    --resolution 224 --n_seed 24 --frame_stride 4 --eval_every 25 --batch 128 --tag "$4"
echo "[$(date)] done $4"
