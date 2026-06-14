#!/bin/bash
# FAE uncollapse sweep (mechanism-driven) + spatiotemporal. Memory-safe.
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs
run(){ tag=$1; shift; echo "=== [gpu1] FAE $tag $(date) ==="
  python scripts/train_fae_trl2d.py --tag "$tag" "$@" > $L/night_fae_$tag.log 2>&1 || echo "$tag FAILED"; }

# snapshot mechanism sweep (batch128, proj4096, capped N/query)
COM="--batch 128 --proj_dim 4096 --mcnt 256 512 --n_query 256"
run recononly  $COM --epochs 40 --sim 0 --std 0   --cov 0            # collapse w/o VICReg?
run lowsim     $COM --epochs 40 --sim 1 --std 50  --cov 1            # alignment-driven?
run strongvar  $COM --epochs 40 --sim 5 --std 100 --cov 10           # force variance+cov
run highrec    $COM --epochs 40 --sim 5 --std 50  --cov 1 --lam_rec 10  # force field preservation
run combo      $COM --epochs 60 --sim 5 --std 100 --cov 5 --n_freq 32   # combined best-guess

# spatiotemporal (coord_dim=3), memory-tighter settings
STC="--temporal --n_frames 8 --batch 64 --proj_dim 4096 --mcnt 512 1024 --n_query 512"
run st_strongvar $STC --epochs 40 --sim 5 --std 100 --cov 10
run st_combo     $STC --epochs 60 --sim 5 --std 100 --cov 5 --n_freq 32

touch $L/.night_fae_done
echo "=== [gpu1] FAE sweep done $(date) ==="
