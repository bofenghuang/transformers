#!/usr/bin/env bash

# HF cache
export TRANSFORMERS_CACHE="/projects/bhuang/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

# WANDB related
# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=hf-whisper-v3

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS="1"

# https://github.com/microsoft/DeepSpeed/issues/662
# export CUDA_VISIBLE_DEVICES="1,2,4,5"
export CUDA_VISIBLE_DEVICES="1,5"

# Debugging flags (optional)
# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONFAULTHANDLER=1

# https://pytorch.org/docs/stable/elastic/run.html
# export HOST_NODE_ADDR="localhost:29001"

# model_name_or_path="openai/whisper-large-v2"
# model_name_or_path="versae/whisper-large-v3"
model_name_or_path="openai/whisper-large-v3"

train_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-07/train_asr_processed_dedup256_shuffled.json"
validation_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-07/valid_asr_mcv13_manifest_normalized_pnc.json"

run_name="whisper-large-v3-ft-french-lr4e6-bs256-streaming"
output_dir="./outputs/hf_whisper/$run_name"

# --gradient_checkpointing \ vs use_cache
# --ddp_find_unused_parameters="True" \ and layerdrop, but can't do --gradient_checkpointing

    # --adam_beta2 "0.98" \
    # --weight_decay "0.01" \
    # --num_train_epochs "2" \
    # --dataloader_num_workers "4" \
    # --max_steps "10319" \
    # --dataloader_num_workers "1" \

    # --ddp_timeout 36000 \

# RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 18 but got size 19 for tensor number 1 in the list.
# https://github.com/huggingface/transformers/issues/24999
    # --dispatch_batches false \

# python \
#     run_speech_recognition_seq2seq_streaming_b.py \

# deepspeed \
#     --master_port 29001 \
#     --include localhost:4,5 \
#     run_speech_recognition_seq2seq_streaming_b.py \
#     --deepspeed ds_config_zero2_no_offload.json \

torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:6000 \
    --max_restarts 0 \
    run_speech_recognition_seq2seq_streaming_b.py \
    --deepspeed ds_config_zero2_no_offload.json \
    --model_name_or_path $model_name_or_path \
    --use_auth_token \
    --apply_spec_augment \
    --train_file $train_file \
    --validation_file $validation_file \
    --audio_column_name "audio_filepath" \
    --streaming \
    --num_shards "16" \
    --max_duration_in_seconds "30" \
    --do_lower_case false \
    --do_normalize_eval false \
    --language "french" \
    --task "transcribe" \
    --dataloader_num_workers "4" \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --run_name $run_name \
    --max_steps "10319" \
    --per_device_train_batch_size "32" \
    --per_device_eval_batch_size "16" \
    --gradient_accumulation_steps "4" \
    --optim "adamw_bnb_8bit" \
    --adam_beta2 "0.98" \
    --learning_rate "4.375e-6" \
    --warmup_ratio "0.05" \
    --logging_steps "10" \
    --evaluation_strategy "steps" \
    --eval_steps "500" \
    --save_strategy "steps" \
    --save_steps "500" \
    --save_total_limit "3" \
    --metric_for_best_model "wer" \
    --greater_is_better false \
    --load_best_model_at_end \
    --freeze_feature_encoder false \
    --fp16 \
    --use_cache false \
    --gradient_checkpointing \
    --predict_with_generate \
    --generation_num_beams "1" \
    --generation_max_length="225" \
    --do_train \
    --do_eval
