#!/bin/bash
# Cluster bootstrap for the Misha (GPFS) setup.
#   repo + data -> scratch ;  conda env -> project
# Adapt the MODULE names / CUDA wheel / GPU partition to your cluster.
# Run the steps below by hand the first time (don't blind-run the whole file).
set -e

# ---- paths (yours) ----
export SCRATCH=/gpfs/radev/home/ss5235/scratch
export PROJECT=/gpfs/radev/home/ss5235/project
export REPO=$SCRATCH/WFAE
export THE_WELL_DATA_DIR=$SCRATCH/the_well_data          # 440 GB lives here
export ENV_PREFIX=$PROJECT/envs/wfae

# ===== 1. repo -> scratch =====
cd $SCRATCH
[ -d "$REPO" ] || git clone https://github.com/SimingShan/FAE.git WFAE
cd $REPO

# ===== 2. conda env -> project =====
# NOTE: plain `conda create` gets OOM-KILLED on login nodes (classic solver
# loads conda-forge+bioconda repodata into RAM). Use libmamba + conda-forge
# only; if still killed, run this inside an `srun --pty --mem=16G bash` job.
module load miniconda || module load anaconda || true     # adapt: `module avail`
conda config --set solver libmamba
conda create -y --prefix $ENV_PREFIX -c conda-forge --override-channels python=3.11
conda activate $ENV_PREFIX
# Alternative (no conda): module load python/3.11 && python -m venv $ENV_PREFIX
#                         && source $ENV_PREFIX/bin/activate

# ===== 3. PyTorch (match the cluster CUDA — check `nvidia-smi`) + deps =====
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install the_well hydra-core timm einops omegaconf h5py scikit-learn matplotlib numpy huggingface_hub

# ===== 4. data: FULL shear_flow (~440 GB) -> scratch =====
# Set an HF token first (faster, avoids rate limits): export HF_TOKEN=hf_xxx
mkdir -p $THE_WELL_DATA_DIR
# RUN THIS IN tmux/screen OR A SLURM JOB — it takes hours:
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('polymathic-ai/shear_flow', repo_type='dataset', \
local_dir='$THE_WELL_DATA_DIR/shear_flow', allow_patterns=['data/*'], max_workers=8)"

# ===== 5. external harness (JEPA + MAE originals) =====
mkdir -p $REPO/external && cd $REPO/external
[ -d physical-representation-learning ] || git clone https://github.com/helenqu/physical-representation-learning
[ -d mae ] || git clone https://github.com/facebookresearch/mae.git
cd physical-representation-learning
git apply $REPO/docs/benchmarks/helenqu_trl2d_integration.patch
cp $REPO/docs/benchmarks/*.yaml configs/
cp $REPO/docs/benchmarks/turbulent_radiative_layer_2D.yaml configs/dataset/
cd $REPO

# ===== 6. verify =====
python benchmarks/smoke_test.py        # MAE / AE / I-JEPA must all PASS

echo "Setup done. Remember to: export THE_WELL_DATA_DIR=$THE_WELL_DATA_DIR  (add to ~/.bashrc)"
