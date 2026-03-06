#!/bin/bash
# set -e

## -----------------------------
## Path Definitions
## -----------------------------
IMAGENET_PATH="/root/highspeedstorage/Junn-US-West-4/datasets/imagenet/train"
CACHED_PATH="/root/highspeedstorage/Junn-US-West-4/datasets/imagenet/cached/vq-f8-n256"
VAE_PATH="/root/highspeedstorage/Junn-US-West-4/pretrained_models/vq-f8-n256/model.ckpt"
VAE_CFG="/root/highspeedstorage/Junn-US-West-4/aebm/first_stage_models/vq-f8-n256/config.yaml"
LOAD_PATH="/root/highspeedstorage/Junn-US-West-4/ckpts/vq-f8-n256/mar_large/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask32x32_seqlen16x16_zprojtied_wresmlp_L2norm_wu5_wd0.02_gc3_bsz1024"
SAVE_PATH="/root/highspeedstorage/Junn-US-West-4/ckpts/vq-f8-n256/mar_large/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask32x32_seqlen16x16_zprojtied_wresmlp_L2norm_wu5_wd0.02_gc3_bsz1024"
LOG_PATH="/root/highspeedstorage/Junn-US-West-4/logs"

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

# Run training
torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    main_mar.py \
    --img_size 256 \
    --vae_mode vq \
    --vae_path ${VAE_PATH} \
    --vae_cfg ${VAE_CFG} \
    --vae_embed_dim 4 \
    --vae_stride 8 \
    --patch_size 2 \
    --model mar_large \
    --batch_size 112 \
    --num_workers 8 \
    --epochs 100 \
    --warmup_epochs 40 \
    --lr 4.0e-4 \
    --weight_decay 0.02 \
    --grad_clip 3.0 \
    --alpha 1.0 \
    --beta 1.0 \
    --ddpmloss_scale 1.0 \
    --celoss_scale 1.0 \
    --reloss_scale 0.0 \
    --diffusion_batch_mul 1 \
    --mask_ratio_min 0.70 \
    --mask_ratio_max 1.00 \
    --mask_ratio_mu 1.00 \
    --mask_ratio_std 0.25 \
    --data_path ${IMAGENET_PATH} \
    --resume ${LOAD_PATH} \
    --output_dir ${SAVE_PATH} \
    --save_freq 10 \
    --save_last_freq 1 \
    --encoder_adaln_mod \
    --decoder_adaln_mod \
    --final_layer_adaln_mod \
    --cached_path ${CACHED_PATH} \
    --use_cached

echo "========================================"
echo " Job Finished: $(date)"
echo "========================================"
