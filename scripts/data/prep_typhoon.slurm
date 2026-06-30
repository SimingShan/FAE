#!/bin/bash
#SBATCH --job-name=typhoon_cache
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/typhooncache_%j.out
#SBATCH --error=logs/typhooncache_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
EXT=$HOME/scratch/digital_typhoon_ext
export TYPHOON_IMG_ROOT=$EXT/WP/image
export TYPHOON_META_ROOT=$EXT/WP/metadata
export TYPHOON_CACHE=$HOME/scratch/typhoon_cache
export TYPHOON_STRIDE_H=6 TYPHOON_SIDE=128
echo "=== prep (manifest + cache + floor) on already-extracted data ==="
python scripts/prep_typhoon.py
