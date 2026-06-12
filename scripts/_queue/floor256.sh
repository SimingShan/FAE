#!/bin/bash
# Invariance-capacity trade-off test: VICReg with sensing floor raised 64->256.
# Prediction: preserved dimension (calibration sweep) rises; low-N invariance drops.
set -uo pipefail
cd "$(dirname "$0")/../.."
python scripts/train_fae.py --method fae_vicreg --gpu 0 \
    --mcnt_choices 256 512 1024 \
    --out results/checkpoints/g1/fae_vicreg_floor256.pt \
    > logs/fae_vicreg_floor256.log 2>&1
DIAG_DEVICE=cuda:0 python scripts/diag_dimension.py > logs/diag_dimension_floor256.log 2>&1
DIAG_DEVICE=cuda:0 python scripts/diag_offgrid.py   > logs/diag_offgrid_floor256.log 2>&1
touch logs/.floor256_done
echo "floor256 queue done $(date)"
