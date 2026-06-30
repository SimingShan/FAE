#!/bin/bash
#SBATCH --job-name=ns_dl
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nsdl_%j.out
#SBATCH --error=logs/nsdl_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
NS=${NS_DATA_ROOT:-$HOME/scratch/ns_data}
python -c "
from huggingface_hub import snapshot_download
snapshot_download('pdearena/NavierStokes-2D-conditoned', repo_type='dataset',
                  local_dir='$NS', allow_patterns=['*_train_*.h5','*_valid_*.h5','*.yaml'], max_workers=8)
print('DONE')
"
du -sh $NS
