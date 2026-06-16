#!/bin/bash
# Per-cluster config sourced by every SLURM script. EDIT THESE (or export them in
# ~/.bashrc) when moving to a new cluster — nothing else in the slurm scripts is
# cluster-specific. Defaults assume repo in $HOME/scratch, env in $HOME/project.
export THE_WELL_DATA_DIR="${THE_WELL_DATA_DIR:-$HOME/scratch/the_well_data}"   # holds shear_flow/data/*
export WFAE_ENV="${WFAE_ENV:-$HOME/project/envs/wfae}"                         # conda prefix OR venv dir
export WFAE_REPO="${WFAE_REPO:-$HOME/scratch/WFAE}"

# module + python env (adapt the module name to your cluster: `module avail`)
module load miniconda 2>/dev/null || module load anaconda 2>/dev/null || module load python 2>/dev/null || true
if eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null && conda activate "$WFAE_ENV" 2>/dev/null; then :
elif [ -f "$WFAE_ENV/bin/activate" ]; then source "$WFAE_ENV/bin/activate"; fi
cd "$WFAE_REPO" || exit 1
