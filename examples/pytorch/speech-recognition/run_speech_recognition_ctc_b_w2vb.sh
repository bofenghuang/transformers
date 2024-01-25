#!/usr/bin/env bash

# HF cache
export HF_HOME="/projects/bhuang/.cache/huggingface"

# WANDB related
# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT="hf-asr-general"

# export PYTHONPATH="$PYTHONPATH:/home/bhuang/my-scripts"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS="1"

# https://github.com/microsoft/DeepSpeed/issues/662
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="1,2,3,5"

# Debugging flags (optional)
# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# https://pytorch.org/docs/stable/elastic/run.html
# export HOST_NODE_ADDR="localhost:29001"

# models
# model_name_or_path="facebook/wav2vec2-large-xlsr-53"
# model_name_or_path="facebook/wav2vec2-xls-r-300m"
# model_name_or_path="facebook/wav2vec2-xls-r-1b"
# model_name_or_path="LeBenchmark/wav2vec2-FR-7K-large"
# model_name_or_path="LeBenchmark/wav2vec2-FR-14K-large"
# model_name_or_path="LeBenchmark/wav2vec2-FR-14K-xlarge"
model_name_or_path="facebook/w2v-bert-2.0"

# train_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21-normalized/train_asr_normalized_cleaned.json"
train_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21-normalized/train_asr_normalized_cleaned_goodchars.json"
validation_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21-normalized/test_asr_mcv13_manifest_normalized.json"

noisedir="/home/bhuang/corpus/speech/public/musan_wo_speech"

run_name="w2v-bert-2.0-ft-ep30-bs256-lr1e4"
output_dir="./outputs/general/$run_name"

# bh: pb related to DDP & dropout (not all params used) & gradient_checkpointing?
# https://github.com/pytorch/pytorch/issues/43259
# https://stackoverflow.com/questions/68000761/pytorch-ddp-finding-the-cause-of-expected-to-mark-a-variable-ready-only-once
# print grad after backward to locate unused params
# https://zhuanlan.zhihu.com/p/640226293
# --ddp_find_unused_parameters true \  # small overhead to identify all unused parameters
# --gradient_checkpointing \

    # --max_train_samples 10000 \
    # --max_eval_samples 1024 \

# python \
#     run_speech_recognition_ctc_b_w2vb.py \

# deepspeed \
#     --master_port 29001 \
#     --include localhost:1,2 \
#     run_speech_recognition_ctc_b_w2vb.py \
#     --deepspeed ds_config.json \
    # --deepspeed ds_config_zero2_no_offload.json \

# todo: add_adapter

torchrun \
    --master_port 29001 \
    --nproc_per_node 4 \
    run_speech_recognition_ctc_b_w2vb.py \
    --model_name_or_path $model_name_or_path \
    --use_auth_token \
    --train_file $train_file \
    --validation_file $validation_file \
    --max_train_samples 50000 \
    --max_eval_samples 1000 \
    --audio_column_name "audio_filepath" \
    --text_column_name "text" \
    --max_duration_in_seconds "30" \
    --min_duration_in_seconds "1" \
    --apply_audio_augmentation false \
    --background_noise_dir $noisedir \
    --audio_augmentation_prob "0.2" \
    --remove_unused_columns false \
    --preprocessing_num_workers "16" \
    --dataloader_num_workers "4" \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --run_name $run_name \
    --num_train_epochs "30" \
    --per_device_train_batch_size "16" \
    --per_device_eval_batch_size "16" \
    --gradient_accumulation_steps "4" \
    --learning_rate "1e-4" \
    --adam_beta2 "0.95" \
    --warmup_ratio "0.05" \
    --lr_scheduler_type "cosine" \
    --weight_decay "0.01" \
    --fp16 \
    --gradient_checkpointing \
    --ctc_zero_infinity \
    --layerdrop "0" \
    --feat_proj_dropout "0" \
    --attention_dropout "0.05" \
    --activation_dropout "0" \
    --hidden_dropout "0" \
    --conformer_conv_dropout "0.1" \
    --final_dropout "0.1" \
    --logging_steps "10" \
    --evaluation_strategy "steps" \
    --eval_steps "1000" \
    --save_strategy "steps" \
    --save_steps "1000" \
    --save_total_limit "3" \
    --metric_for_best_model "wer" \
    --greater_is_better false \
    --load_best_model_at_end \
    --do_train \
    --do_eval

    # --mask_time_prob "0.05" \
    # --mask_time_length "10" \
    # --mask_feature_prob "0.05" \
    # --mask_feature_length "10" \