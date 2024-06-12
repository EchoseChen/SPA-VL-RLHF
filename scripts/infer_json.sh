#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8
export TRANSFORMERS_OFFLINE=1

dataset_paths=(
    # "XXX/final_test.json"
)
output_dirs=(
    # "XXX/test"

)

output_files=(
    # "test.json"
)


for i in "${!dataset_paths[@]}"; do
    echo "Processing dataset: ${dataset_paths[i]}"
    torchrun \
        --master_port=50100 \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=$GPUS_PER_NODE \
        ./infer_json.py \
        --lora_enable False \
        --model_name_or_path XXX \
        --dataset_path "${dataset_paths[i]}" \
        --output_dir "${output_dirs[i]}" \
        --output_file "${output_files[i]}" \
        --ddp_backend "nccl" \
        --temperature 0.0 \
        --model_max_length 2048 \
        --image_folder XXX \
        --vision_tower XXX/clip-vit-large-patch14-336 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio 'pad'
done

