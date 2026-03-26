#!/bin/bash
# set -e

## -----------------------------
## Path Definitions
## -----------------------------
PROJECT_ROOT="/root/autodl-tmp"
IMAGENET_PATH="${PROJECT_ROOT}/datasets/imagenet/train"
CACHED_PATH="${PROJECT_ROOT}/datasets/imagenet/cached/vq-f8-n256"
VAE_PATH="${PROJECT_ROOT}/pretrained_models/vq-f8-n256/model.ckpt"
VAE_CFG="${PROJECT_ROOT}/aebm/first_stage_models/vq-f8-n256/config.yaml"
LOAD_PATH="${PROJECT_ROOT}/ckpts/vq-f8-n256/mar_base/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask16x16_seqlen16x16_zprojtied_wresmlp_L2norm_blr2e-4_sqrt_wu50_wd0.05_gc3_bsz1024/checkpoint-last.pth"
SAVE_PATH="${PROJECT_ROOT}/ckpts/vq-f8-n256/mar_base/masked_alpha1.0_beta1.0_ddpm1.0_ce1.0_re0.0_mask16x16_seqlen16x16_zprojtied_wresmlp_L2norm_blr2e-4_sqrt_wu50_wd0.05_gc3_bsz1024"
LOG_PATH="${PROJECT_ROOT}/logs"

# mkdir -p $IMAGENET_PATH $CACHED_PATH $VAE_PATH
# tar -xvf archive.tar.gz -C /path/to/destination
# unzip archive.zip -d /path/to/destination
# chmod +x train.sh

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
# pip install tensorboard tqdm scipy einops timm torch-fidelity opencv-python pytorch-lightning omegaconf
# export OMP_NUM_THREADS=1

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
echo "Starting training..."
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
    --model mar_base \
    --batch_size 64 \
    --accum_iter 2 \
    --num_workers 16 \
    --epochs 100 \
    --warmup_epochs 50 \
    --blr 1.0e-4 \
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
    --cached_path ${CACHED_PATH} \
    --resume ${LOAD_PATH} \
    --output_dir ${SAVE_PATH} \
    --save_freq 10 \
    --save_last_freq 1 \
    --encoder_adaln_mod \
    --decoder_adaln_mod \
    --final_layer_adaln_mod \
    --use_cached
    # --grad_checkpointing

echo "========================================"
echo " Job Finished: $(date)"
echo "========================================"
