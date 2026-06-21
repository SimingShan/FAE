#!/bin/bash
#SBATCH --job-name=smk_tmode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=logs/smk_tmode_%j.out
#SBATCH --error=logs/smk_tmode_%j.err
# Fast GPU gate: run the real temporal training path for BOTH time modes,
# tiny (n_seed 1, 1 epoch). If both print a probe line + save, the 3 real
# experiments are safe to launch.
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
for tm in coord aggregate; do
    echo "######## SMOKE time_mode=$tm ########"
    python scripts/train_fae_shear.py --temporal --n_frames 16 --time_mode $tm \
        --sim 0 --std 0 --cov 0 --epochs 1 --n_seed 1 --batch 4 --tag smk_$tm \
        && echo "SMOKE_OK $tm" || echo "SMOKE_FAIL $tm"
done
echo "ALL_SMOKE_DONE"
