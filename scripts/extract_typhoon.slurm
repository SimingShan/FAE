#!/bin/bash
#SBATCH --job-name=typhoon_prep
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/typhoonprep_%j.out
#SBATCH --error=logs/typhoonprep_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
DT=$HOME/scratch/digital_typhoon
EXT=$HOME/scratch/digital_typhoon_ext
mkdir -p "$EXT"
echo "=== extracting WP tar -> $EXT (images + metadata) ==="
cat "$DT"/WP.tar.gzaa "$DT"/WP.tar.gzab | tar xzf - -C "$EXT"
echo "=== extracted; layout ==="; ls "$EXT/WP/"; echo "n_typhoon_dirs=$(ls $EXT/WP/image | wc -l)"; ls "$EXT/WP/metadata" 2>/dev/null | head -2
export TYPHOON_IMG_ROOT=$EXT/WP/image
export TYPHOON_META_ROOT=$EXT/WP/metadata
export TYPHOON_CACHE=$HOME/scratch/typhoon_cache
export TYPHOON_STRIDE_H=6 TYPHOON_SIDE=128
echo "=== prep (manifest + cache + floor) ==="
python scripts/prep_typhoon.py
