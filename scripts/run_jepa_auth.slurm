#!/bin/bash
#SBATCH --job-name=jepa_auth
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jepa_auth_%j.out
#SBATCH --error=logs/jepa_auth_%j.err
# AUTHENTIC JEPA via the helenqu harness (their exact pipeline): conv-JEPA
# pretrain (16 frames, 6 epochs, VICReg 2/40/2) -> attentive-pooling finetune
# probe of (Re, Sc). Target = their Table 1 JEPA MSE 0.38.
# NOTE: 4xa100 -> global batch 32 (paper used 8 GPUs -> 64); bump if the number
# is off. shear_flow is natively supported in their data.py (no patch needed).
set -e
export THE_WELL_DATA_DIR=/gpfs/radev/home/ss5235/scratch/the_well_data
export WANDB_MODE=disabled
module load miniconda/24.3.0
eval "$(conda shell.bash hook)"
conda activate /gpfs/radev/home/ss5235/project/envs/wfae
cd /gpfs/radev/home/ss5235/scratch/WFAE/external/physical-representation-learning

echo "[$(date)] ===== JEPA pretrain (16f / 6ep / 4xa100) ====="
torchrun --nproc_per_node=4 --standalone \
    -m physics_jepa.train_jepa configs/train_shearflow_small.yaml

CKPT=$(find checkpoints -name "ConvEncoder_*.pth" | sort -V | tail -1)
echo "[$(date)] ===== pretrain done. encoder = $CKPT ====="
[ -z "$CKPT" ] && { echo "NO ENCODER CHECKPOINT FOUND"; exit 1; }

echo "[$(date)] ===== JEPA finetune: attentive probe of (Re, Sc) ====="
python -m physics_jepa.finetune configs/train_shearflow_small.yaml \
    --trained_model_path "$CKPT" --seed 42
echo "[$(date)] ===== done — compare reported MSE to paper JEPA 0.38 ====="
