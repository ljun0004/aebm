#!/bin/bash
# ==========================================
# Evaluate MAR on ImageNet (Multi GPU)
# ==========================================

# Define paths
IMAGENET_PATH="/home/ljun0004/sa31_scratch2/Junn/datasets/imagenet/train"
CACHED_PATH="/home/ljun0004/sa31_scratch2/Junn/datasets/imagenet/cache_vq-f8-n256"

VAE_PATH="/home/ljun0004/sa31/Junn/bmar/pretrained_models/vae/vq-f8-n256.ckpt"
VAE_CFG="/home/ljun0004/sa31/Junn/bmar/first_stage_models/vq-f8-n256/config.yaml"

LOAD_PATH="/home/ljun0004/sa31/Junn/bmar/checkpoints/pretrain/vq-f8-n256/mar_base/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask32x32_seqlen16x16_zprojtied_encembed_wmlpratio4_rescon_L2norm_wu0_wd0.02_gc3"
SAVE_PATH="/home/ljun0004/sa31/Junn/bmar/checkpoints/pretrain/vq-f8-n256/mar_base/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask32x32_seqlen16x16_zprojtied_encembed_wmlpratio4_rescon_L2norm_wu0_wd0.02_gc3"

# LOAD_PATH="/home/ljun0004/sa31/Junn/bmar/checkpoints/pretrain/vq-f8-n256/mar_base/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask32x32_seqlen16x16_zprojtied_encembed_wmlpratio4_rescon_L2norm_wu0_wd0.0_gc10"
# SAVE_PATH="/home/ljun0004/sa31/Junn/bmar/checkpoints/pretrain/vq-f8-n256/mar_base/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask32x32_seqlen16x16_zprojtied_encembed_wmlpratio4_rescon_L2norm_wu0_wd0.0_gc10"

# Run testing
torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    main_mar.py \
    --img_size 256 \
    --vae_mode vq \
    --vae_path ${VAE_PATH} \
    --vae_cfg ${VAE_CFG} \
    --vae_embed_dim 4 \
    --vae_stride 8 \
    --patch_size 2 \
    --model mar_base \
    --batch_size 64 \
    --num_workers 10 \
    --epochs 400 \
    --warmup_epochs 0 \
    --blr 1.0e-4 \
    --weight_decay 0.0 \
    --grad_clip 10.0 \
    --alpha 1.0 \
    --beta 1.0 \
    --ddpmloss_scale 1.0 \
    --celoss_scale 1.0 \
    --reloss_scale 0.0 \
    --min_logit_scale 0.0 \
    --max_logit_scale 1.0 \
    --diffusion_batch_mul 1 \
    --mask_ratio_min 0.70 \
    --mask_ratio_max 1.00 \
    --mask_ratio_mu 1.00 \
    --mask_ratio_std 0.25 \
    --data_path ${IMAGENET_PATH} \
    --resume ${LOAD_PATH} \
    --output_dir ${SAVE_PATH} \
    --save_last_freq 1 \
    --evaluate \
    --sampling_mode diffusion \
    --gen_freq 1 \
    --eval_bsz 64 \
    --eval_num_images 5000 \
    --num_iter 1 \
    --num_sampling_steps 100 \
    --cfg 2.9 \
    --cfg_schedule linear \
    --temperature 1.0 \
    --encoder_adaln_mod \
    --decoder_adaln_mod \
    --final_layer_adaln_mod \
    --cached_path ${CACHED_PATH} \
    --use_cached

    # --decoder_adaln_mod \
    # --final_layer_adaln_mod \
    # --grad_checkpointing
