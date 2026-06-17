#!/bin/bash
#SBATCH --job-name=reprobe
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/reprobe_%j.out
#SBATCH --error=logs/reprobe_%j.err
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
C=results/checkpoints/g1
python scripts/eval_baseline_readout.py --ckpt \
  $C/mae_shear_mae_nf1.pt $C/ijepa_shear_ijepa_nf1.pt \
  $C/videomae_shear_vmae_nf4.pt $C/stjepa_shear_stj_nf4.pt \
  $C/videomae_shear_vmae_nf8.pt $C/stjepa_shear_stj_nf8.pt \
  $C/videomae_shear_vmae_nf16.pt $C/stjepa_shear_stj_nf16.pt
echo DONE
