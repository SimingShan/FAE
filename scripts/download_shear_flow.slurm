#!/bin/bash
#SBATCH --job-name=dl_shear
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dl_shear_%j.out
#SBATCH --error=logs/dl_shear_%j.err
# Download The Well shear_flow FULL (~440 GB) to scratch.
# snapshot_download is RESUMABLE: if the 24h limit hits, just resubmit this
# script and it picks up where it left off (hf-xet skips completed files).

set -e
export THE_WELL_DATA_DIR=/gpfs/radev/home/ss5235/scratch/the_well_data
export ENV_PREFIX=/gpfs/radev/home/ss5235/project/envs/wfae

module load miniconda/24.3.0
eval "$(conda shell.bash hook)"
conda activate $ENV_PREFIX

mkdir -p $THE_WELL_DATA_DIR
echo "[$(date)] starting download to $THE_WELL_DATA_DIR/shear_flow"

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'polymathic-ai/shear_flow',
    repo_type='dataset',
    local_dir='$THE_WELL_DATA_DIR/shear_flow',
    allow_patterns=['data/*'],
    max_workers=8,
)
print('DONE')
"
echo "[$(date)] finished. footprint:"
du -sh $THE_WELL_DATA_DIR/shear_flow
