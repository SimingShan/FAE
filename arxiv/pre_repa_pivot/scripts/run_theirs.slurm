#!/bin/bash
#SBATCH --job-name=ssl_theirs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/theirs_%j.out
#SBATCH --error=logs/theirs_%j.err
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
cd external/SSLForPDEs
python navier_stokes.py --data-root $HOME/scratch/ns_data \
  --logging-folder $SLURM_SUBMIT_DIR/results/ssl_logs --exp-name "${1:-ssl_full}" \
  --epochs "${2:-100}" --num-workers 8
