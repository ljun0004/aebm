#!/bin/bash
# ==========================================
# Evaluate MAR on ImageNet (Single GPU)
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

LOAD_PATH="/home/ljun0004/sa31/Junn/bmar/checkpoints/pretrain/vq-f8-n256/mar_large/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask32x32_seqlen16x16_zprojtied_wresmlp_L2norm_wu5_wd0.05_gc3_bsz32x4"
SAVE_PATH="/home/ljun0004/sa31/Junn/bmar/checkpoints/pretrain/vq-f8-n256/mar_large/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask32x32_seqlen16x16_zprojtied_wresmlp_L2norm_wu5_wd0.05_gc3_bsz32x4"

# Run testing
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    main_mar.py \
    --img_size 256 \
    --vae_mode vq \
    --vae_ckpt ${VAE_CKPT} \
    --vae_cfg ${VAE_CFG} \
    --vae_embed_dim 4 \
    --vae_stride 8 \
    --patch_size 2 \
    --model mar_large \
    --batch_size 32 \
    --num_workers 10 \
    --epochs 400 \
    --base_warmup_epochs 10 \
    --blr 1.0e-4 \
    --weight_decay 0.05 \
    --grad_clip 3.0 \
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
    --save_freq 20 \
    --save_last_freq 1 \
    --sampling_mode diffusion \
    --evaluate \
    --eval_freq 5 \
    --eval_bsz 32 \
    --eval_num_images 5000 \
    --online_gen \
    --gen_freq 1 \
    --gen_bsz 1 \
    --gen_num_images 1 \
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