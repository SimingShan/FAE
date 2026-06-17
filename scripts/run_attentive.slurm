#!/bin/bash
#SBATCH --job-name=attn_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/attn_%j.out
#SBATCH --error=logs/attn_%j.err
# Cross-paper parity: Qu et al. attentive probe on our frozen encoders.
# Headline (FAE-temporal) first, then single-frame methods. No set -e so one
# failure doesn't kill the rest. 256G for the 16-frame window load.
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
CK=results/checkpoints/g1

echo "######## FAE temporal 16f (headline) ########"
python scripts/eval_attentive_probe.py --method fae --temporal \
    --ckpt $CK/fae_vicreg_shear_align224t.pt || echo "FAE-temporal FAILED"
echo "######## FAE single ########"
python scripts/eval_attentive_probe.py --method fae \
    --ckpt $CK/fae_vicreg_shear_align224.pt || echo "FAE-single FAILED"
for M in mae ae ijepa; do
    echo "######## $M ########"
    python scripts/eval_attentive_probe.py --method $M \
        --ckpt $CK/${M}_shear_align224.pt || echo "$M FAILED"
done
echo "[$(date)] attentive sweep done"
