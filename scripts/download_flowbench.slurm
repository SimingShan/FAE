#!/bin/bash
#SBATCH --job-name=dl_fb
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dl_fb_%j.out
#SBATCH --error=logs/dl_fb_%j.err
# FlowBench FPO subset (flow-past-object, time-dependent, Re-labeled) -> $FB. Resumable.
set -e
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
FB=${FLOWBENCH_DIR:-$HOME/scratch/flowbench}
python - <<PY
from huggingface_hub import snapshot_download
cases=range(1,21)
pats=[f"FPO_NS_2D_1024x256/harmonics/{c}/*" for c in cases]
snapshot_download("BGLab/FlowBench", repo_type="dataset", local_dir="$FB",
                  allow_patterns=pats, max_workers=8)
print("DONE")
PY
echo "[$(date)] footprint:"; du -sh $FB/FPO_NS_2D_1024x256 2>/dev/null
