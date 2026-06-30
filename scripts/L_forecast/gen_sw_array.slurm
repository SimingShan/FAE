#!/bin/bash
#SBATCH --job-name=sw_arr
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/swarr_%A_%a.out
#SBATCH --error=logs/swarr_%A_%a.err
# Array-chunked SW gen: each task does 11 samples to its OWN chunk npz + snapdir (no RAM leak, no clobber).
# Covers samples [29:150) — the existing shallow_water.npz already has 0..28.
DEDPY=/gpfs/radev/home/ss5235/project/envs/dedalus/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
WORK=$HOME/scratch/sw_data; mkdir -p "$WORK/chunks"; cd "$WORK"
CHUNK=11; BASE=29
START=$((BASE + SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK)); [ $END -gt 150 ] && END=150
export SW_START=$START SW_END=$END
export SW_OUT="$WORK/chunks/sw_chunk_${START}.npz"
export SW_SNAPDIR="snap_t${SLURM_ARRAY_TASK_ID}"
echo "=== array task $SLURM_ARRAY_TASK_ID -> samples [$START:$END) -> $SW_OUT ==="
$DEDPY "$SLURM_SUBMIT_DIR/scripts/gen_sw.py"
