#!/bin/bash
# Experiment (b) — learned 8-query readout trained under VICReg.
# Tests whether FAE's dimension cap (~10) is a mean-pool readout artifact:
# the encoder tokens carry ID~22 but it is not linearly accessible post-hoc.
# A readout trained UNDER the SSL pressure should expose it.
set -uo pipefail
cd "$(dirname "$0")/../.."
name=fae_vicreg_q8
python scripts/train_fae.py --method fae_vicreg --gpu 0 --readout_queries 8 \
    --out results/checkpoints/g1/${name}.pt > logs/${name}.log 2>&1
DIAG_DEVICE=cuda:0 python scripts/evaluate.py        > logs/evaluate_q8.log 2>&1
DIAG_DEVICE=cuda:0 python scripts/diag_dimension.py  > logs/diag_dimension_q8.log 2>&1
DIAG_DEVICE=cuda:0 python scripts/diag_offgrid.py    > logs/diag_offgrid_q8.log 2>&1
touch logs/.q8_done
echo "q8 queue done $(date)"
