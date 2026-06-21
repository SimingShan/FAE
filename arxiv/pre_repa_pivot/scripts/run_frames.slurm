#!/bin/bash
#SBATCH --job-name=frames
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/frames_%j.out
#SBATCH --error=logs/frames_%j.err
# args: SEED NFRAMES EPOCHS TAG   (NFRAMES<=1 => non-temporal single-frame anchor)
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
NF=$2
if [ "$NF" -le 1 ]; then MODE=""; B=256; else MODE="--temporal --n_frames $NF"; if [ "$NF" -ge 8 ]; then B=64; else B=128; fi; fi
echo "[$(date)] seed=$1 nframes=$NF batch=$B tag=$4 mode=$MODE"
python scripts/train_fae_shear.py $MODE --epochs $3 --batch $B --n_seed 24 \
    --resolution 224 --mcnt 256 512 --seed $1 --tag $4
echo "[$(date)] done $4"
