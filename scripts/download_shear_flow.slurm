#!/bin/bash
#SBATCH --job-name=dl_shear
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dl_shear_%j.out
#SBATCH --error=logs/dl_shear_%j.err
# Download The Well shear_flow FULL (~440 GB) from HuggingFace into $THE_WELL_DATA_DIR.
# RESUMABLE: if the time limit hits, just resubmit — snapshot_download skips completed files.
# Portable: no cluster paths here; set them in scripts/env.sh (or export
# THE_WELL_DATA_DIR / WFAE_ENV / WFAE_REPO). For speed also: export HF_TOKEN=hf_xxx
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
mkdir -p "$THE_WELL_DATA_DIR"
echo "[$(date)] downloading shear_flow -> $THE_WELL_DATA_DIR/shear_flow"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('polymathic-ai/shear_flow', repo_type='dataset',
                  local_dir='$THE_WELL_DATA_DIR/shear_flow',
                  allow_patterns=['data/*'], max_workers=8)
print('DONE')
"
echo "[$(date)] done. footprint:"; du -sh "$THE_WELL_DATA_DIR/shear_flow"
