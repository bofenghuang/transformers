#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED="1"

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
export CUDA_VISIBLE_DEVICES="0,3"

    # --num_train_epochs 1 \
    # --multi_gpu \

accelerate launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    --main_process_port 12345 \
    run_clm_no_trainer.py \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config_name "20231101.fr" \
    --model_name_or_path "gpt2" \
    --output_dir ./output/test-clm \
    --num_warmup_steps 5 \
    --max_train_steps 20 \
    --block_size 512 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type "linear" \
    --report_to "wandb" \
    --with_tracking


# export WANDB_PROJECT=clm_no_trainer

# accelerate launch \
#     --multi_gpu \
#     --num_processes 2 \
#     --mixed_precision fp16 \
#     --main_process_port 12345 \
#     run_clm.py \
#     --dataset_name "wikimedia/wikipedia" \
#     --dataset_config_name "20231101.fr" \
#     --model_name_or_path "gpt2" \
#     --output_dir ./output/test-clm \
#     --overwrite_output_dir \
#     --warmup_steps 5 \
#     --max_steps 20 \
#     --block_size 512 \
#     --preprocessing_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --gradient_accumulation_steps 2 \
#     --lr_scheduler_type "linear" \
#     --logging_steps 1 \
#     --report_to "wandb" \
#     --do_train