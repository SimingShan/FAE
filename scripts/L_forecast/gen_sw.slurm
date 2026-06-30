#!/bin/bash
#SBATCH --job-name=sw_gen
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=logs/swgen_%j.out
#SBATCH --error=logs/swgen_%j.err
DEDPY=/gpfs/radev/home/ss5235/project/envs/dedalus/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
WORK=$HOME/scratch/sw_data; mkdir -p "$WORK"; cd "$WORK"          # snapshots dir written here, not the repo
$DEDPY "$SLURM_SUBMIT_DIR/scripts/gen_sw.py"
