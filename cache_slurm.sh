#!/bin/bash
# ==========================================
# Training MAR on ImageNet (Single GPU)
# ==========================================

# Define paths
IMAGENET_PATH="/home/ljun0004/sa31_scratch2/Junn/datasets/imagenet/train"

# CACHED_PATH="/home/ljun0004/sa31_scratch2/Junn/datasets/imagenet/cache_kl16"
# VAE_CKPT="/home/ljun0004/sa31/Junn/mar/pretrained_models/vae/kl16.ckpt"
# VAE_CFG="/home/ljun0004/sa31/Junn/mar/pretrained_models/vae/kl16.ckpt"

# CACHED_PATH="/home/ljun0004/sa31_scratch2/Junn/datasets/imagenet/cache_vq-f8"
# VAE_CKPT="/home/ljun0004/sa31/Junn/bmar/pretrained_models/vae/vq-f8.ckpt"
# VAE_CFG="/home/ljun0004/sa31/Junn/bmar/first_stage_models/vq-f8/config.yaml"

CACHED_PATH="/home/ljun0004/sa31_scratch2/Junn/datasets/imagenet/cache_vq-f8-n256"
VAE_CKPT="/home/ljun0004/sa31/Junn/bmar/pretrained_models/vae/vq-f8-n256.ckpt"
VAE_CFG="/home/ljun0004/sa31/Junn/bmar/first_stage_models/vq-f8-n256/config.yaml"

# Run training
torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    main_cache.py \
    --img_size 256 \
    --vae_mode vq \
    --vae_ckpt ${VAE_CKPT} \
    --vae_cfg ${VAE_CFG} \
    --vae_embed_dim 4 \
    --batch_size 256 \
    --data_path ${IMAGENET_PATH} \
    --cached_path ${CACHED_PATH}