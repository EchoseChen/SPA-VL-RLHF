#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=DETAIL



torchrun --master_port=50100 --standalone --nnodes=1 --nproc-per-node=$GPUS_PER_NODE \
    /opt/tiger/llava-rlhf/llava/RLHF/train_rm.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --config_train_path "XXX/config_train.json" \
    --config_test_path "XXX/config_test.json" \
    --keywords keywords1 keywords2 keywords3 \
    --finetune_mm_projector True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --lora_enable False \
    --gradient_accumulation_steps 1 \
    --model_name_or_path XXX/models/llava-v1.5-7b \
    --image_folder XXX \
    --vision_tower XXX/models/clip-vit-large-patch14-336 \
    --learning_rate 3e-5 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --model_max_length 2048 \
    --output_dir ./results/rm \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "epoch" \
    --num_train_epochs 1 \
    --save_total_limit 1000 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --bits 16 \
    --ddp_find_unused_parameters False \
    --image_aspect_ratio 'pad'
