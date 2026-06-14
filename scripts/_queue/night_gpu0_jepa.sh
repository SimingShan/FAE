#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/../../external/physical-representation-learning"
export THE_WELL_DATA_DIR=${THE_WELL_DATA_DIR:-/mnt/crucial/shansiming/project/the_well_data}
export WANDB_MODE=disabled
L=$(cd "$(dirname "$0")/../.." && pwd)/logs
echo "=== [gpu0] JEPA authentic pretrain (40ep) $(date) ==="
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m physics_jepa.train_jepa configs/train_trl2d_auth.yaml > $L/night_jepa_auth_train.log 2>&1
CK=$(ls -dt checkpoints/*auth40* | head -1)
ENC=$(ls -v $CK/ConvEncoder_*.pth | tail -1)
echo "=== [gpu0] JEPA finetune $(date) enc=$ENC ==="
CUDA_VISIBLE_DEVICES=0 python -m physics_jepa.finetune \
    configs/train_trl2d_auth.yaml --trained_model_path "$ENC" > $L/night_jepa_auth_ft.log 2>&1
touch $L/.night_jepa_done
echo "=== [gpu0] JEPA done $(date) ==="
