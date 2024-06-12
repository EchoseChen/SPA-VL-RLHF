#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8

# Set the log file path
LOGFILE="./results/ppo_lora/training.log"

# Ensure the log file directory exists
mkdir -p "$(dirname "$LOGFILE")"

# Run command and log output to the log file
{
    echo "Starting training at $(date)"
    accelerate launch --config_file ./scripts/config.yaml \
        ./train_ppo.py \
        --do_train \
        --seed 42 \
        --config_train_path "XXX/config_train.json" \
        --keywords keyword1 keyword2 \
        --finetune_mm_projector True \
        --batch_size 1 \
        --rollouts_gradient_accumulation_steps 1 \
        --lora_enable True \
        --base_model_name XXX/models/llava-v1.5-7b \
        --policy_model_name_or_path XXX/models/llava-v1.5-7b \
        --reward_model_name_or_path ./results/rm_lora/XXX \
        --learning_rate 1e-6 \
        --warmup_steps 10 \
        --output_dir ./results/dpo \
        --total_epochs 1 \
        --evaluation_strategy "no" \
        --weight_decay 0.0 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --ddp_backend "nccl" \
        --bf16 True \
        --ddp_find_unused_parameters False \
        --kl_coef 0.1 \
        --max_grad_norm 1.0 \
        --whitening_async_stats "full_batch" \
        --clean_tokens_after_eos True \
        --temperature 1.0 \
        --whiten_rewards False \
        --model_max_length 2048 \
        --image_folder XXX \
        --vision_tower XXX/clip-vit-large-patch14-336 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio 'pad'
    echo "Training completed at $(date)"
} | tee "$LOGFILE"
