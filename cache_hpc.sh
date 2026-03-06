#!/bin/bash
# set -e

## -----------------------------
## Path Definitions
## -----------------------------
IMAGENET_PATH="/root/highspeedstorage/Junn-US-West-4/datasets/imagenet/train"
CACHED_PATH="/root/highspeedstorage/Junn-US-West-4/datasets/imagenet/cached/vq-f8-n256"
VAE_PATH="/root/highspeedstorage/Junn-US-West-4/pretrained_models/vq-f8-n256/model.ckpt"
VAE_CFG="/root/highspeedstorage/Junn-US-West-4/aebm/first_stage_models/vq-f8-n256/config.yaml"
LOG_PATH="/root/highspeedstorage/Junn-US-West-4/logs"

## -----------------------------
## Automated Logging
## -----------------------------
mkdir -p "${LOG_PATH}"
LOG_FILE="${LOG_PATH}/cache_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo " Job Started: $(date)"
echo " Log file: ${LOG_FILE}"
echo "========================================"

## -----------------------------
## Environment Setup
## -----------------------------
source /opt/conda/etc/profile.d/conda.sh
source ~/.bashrc
conda activate aebm

echo "===== Environment Check ====="
which python
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "============================="

## -----------------------------
## Auto-Detect GPU Count
## -----------------------------
export NPROC_PER_NODE=$(nvidia-smi -L | grep -c "GPU")
echo " Node: $(hostname)" 
echo " Auto-detected GPUs: ${NPROC_PER_NODE}" 
echo "========================================"

nvidia-smi -L 

## -----------------------------
## Distributed Setup
## -----------------------------
# export MASTER_ADDR=localhost
# export MASTER_PORT=29500
# export NODE_RANK=0
# export NNODES=1

## -----------------------------
## Execution
## -----------------------------
cd /root/highspeedstorage/Junn-US-West-4/aebm

echo "Starting job..."

# Using torchrun to parallelize the encoding across all detected GPUs
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
