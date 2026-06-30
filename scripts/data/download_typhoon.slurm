#!/bin/bash
#SBATCH --job-name=typhoon_dl
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/typhoondl_%j.out
#SBATCH --error=logs/typhoondl_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
DT=${TYPHOON_DATA_ROOT:-$HOME/scratch/digital_typhoon}
mkdir -p "$DT"
echo "downloading torchgeo/digital_typhoon (~58GB) -> $DT"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('torchgeo/digital_typhoon', repo_type='dataset',
                  local_dir='$DT', max_workers=8)
print('DONE')
"
du -sh "$DT"
ls -la "$DT" | head