#!/bin/bash
# Capacity-cap suspect #2: VICReg alignment strength (sim 25 -> 5).
set -uo pipefail
cd "$(dirname "$0")/../.."
python scripts/train_fae.py --method fae_vicreg --gpu 0 --sim_coeff 5 \
    --out results/checkpoints/g1/fae_vicreg_sim5.pt > logs/fae_vicreg_sim5.log 2>&1
DIAG_DEVICE=cuda:0 python scripts/diag_dimension.py > logs/diag_dimension_sim5.log 2>&1
DIAG_DEVICE=cuda:0 python scripts/diag_offgrid.py   > logs/diag_offgrid_sim5.log 2>&1
touch logs/.sim5_done
