#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/../../external/physical-representation-learning"
export THE_WELL_DATA_DIR=/mnt/crucial/shansiming/project/the_well_data
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/mnt/crucial/shansiming/project/WFAE/logs
echo "=== [gpu0] JEPA shear pretrain $(date) ==="
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m physics_jepa.train_jepa configs/train_shear_auth.yaml > $L/shear_jepa_train.log 2>&1
CK=$(ls -dt checkpoints/*shear-auth20* | head -1); ENC=$(ls -v $CK/ConvEncoder_*.pth | tail -1)
echo "=== [gpu0] JEPA shear finetune $(date) ==="
CUDA_VISIBLE_DEVICES=0 python -m physics_jepa.finetune \
    configs/train_shear_auth.yaml --trained_model_path "$ENC" > $L/shear_jepa_ft.log 2>&1
touch $L/.shear_jepa_done; echo "=== [gpu0] done $(date) ==="
