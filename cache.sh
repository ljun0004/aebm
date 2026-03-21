#!/bin/bash
# set -e

## -----------------------------
## Path Definitions
## -----------------------------
PROJECT_ROOT="/root/Junn"
IMAGENET_PATH="${PROJECT_ROOT}/datasets/imagenet/train"
CACHED_PATH="${PROJECT_ROOT}/datasets/imagenet/cached/vq-f8-n256"
VAE_PATH="${PROJECT_ROOT}/pretrained_models/vq-f8-n256/model.ckpt"
VAE_CFG="${PROJECT_ROOT}/aebm/first_stage_models/vq-f8-n256/config.yaml"
LOAD_PATH="${PROJECT_ROOT}/ckpts/vq-f8-n256/mar_large/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask32x32_seqlen16x16_zprojtied_wresmlp_L2norm_wu5_wd0.02_gc3_bsz1024"
SAVE_PATH="${PROJECT_ROOT}/ckpts/vq-f8-n256/mar_large/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask32x32_seqlen16x16_zprojtied_wresmlp_L2norm_wu5_wd0.02_gc3_bsz1024"
LOG_PATH="${PROJECT_ROOT}/logs"

## -----------------------------
## Automated Logging
## -----------------------------
mkdir -p "${LOG_PATH}"
LOG_FILE="${LOG_PATH}/train_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "========================================"
echo " Job Started: $(date)"
echo " Log file: ${LOG_FILE}"
echo "========================================"

## -----------------------------
## Environment Setup
## -----------------------------
# export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda/envs"
# export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda/pkgs"
# source /opt/conda/etc/profile.d/conda.sh
# conda config --prepend envs_dirs "${CONDA_ENVS_PATH}"
# conda config --prepend pkgs_dirs "${CONDA_PKGS_DIRS}"
# conda activate aebm || conda activate "${CONDA_ENVS_PATH}/aebm"
# conda info --envs
pip install tensorboard tqdm scipy einops timm torch-fidelity opencv-python pytorch-lightning omegaconf

echo "===== Environment Check ====="
which python
echo "CONDA_PREFIX=${CONDA_PREFIX}"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.version.cuda}')
print(f'GPUs:    {torch.cuda.device_count()}')
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f'Arch:    {major}.{minor}') 
"
echo "=========================================="

## -----------------------------
## Auto-Detect GPU Count
## -----------------------------
export NPROC_PER_NODE=$(nvidia-smi -L | grep -c "GPU")
echo " Node: $(hostname)" 
echo " Auto-detected GPUs: ${NPROC_PER_NODE}" 
nvidia-smi -L 
echo "========================================"

## -----------------------------
## Execution
## -----------------------------
cd "${PROJECT_ROOT}/aebm"
echo "Starting caching..."

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    main_cache.py \
    --img_size 256 \
    --vae_mode vq \
    --vae_path "${VAE_PATH}" \
    --vae_cfg "${VAE_CFG}" \
    --vae_embed_dim 4 \
    --batch_size 128 \
    --num_workers 16 \
    --data_path "${IMAGENET_PATH}" \
    --cached_path "${CACHED_PATH}"

echo "========================================"
echo " Job Finished: $(date)"
echo "========================================"
