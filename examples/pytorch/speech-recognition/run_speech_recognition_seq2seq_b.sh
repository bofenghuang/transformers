#!/usr/bin/env bash

# HF cache
export HF_HOME="/projects/bhuang/.cache/huggingface"

# WANDB related
# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=hf-whisper-v3

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS="1"

# https://github.com/microsoft/DeepSpeed/issues/662
export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
# export CUDA_VISIBLE_DEVICES="1"

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

# train_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-07/train_asr_processed_dedup256.json"
# validation_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-07/valid_asr_mcv13_manifest_normalized_pnc.json"
# train_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-13/train_asr_processed_dedup256_processed_cleaned.json"
# validation_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-13/test_asr_mcv13_manifest_normalized_pnc.json"
train_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/train_asr_processed_cleaned.json"
validation_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_mcv13_manifest_normalized_pnc.json"

noisedir="/home/bhuang/corpus/speech/public/musan_wo_speech"

run_name="whisper-large-v3-ft-french-pnc-ep8-bs280-lr4e6-wd001-audioaug-specaug-betterdata"
output_dir="./outputs/hf_whisper/$run_name"

# --gradient_checkpointing \ vs use_cache
# --ddp_find_unused_parameters="True" \ and layerdrop, but can't do --gradient_checkpointing

# todo :  --timestamp_probability 0.2; --condition_on_prev_probability 0

    # --ddp_timeout 36000 \
    # --adam_beta2 "0.98" \
    # --adam_beta2 "0.95" \
    # --weight_decay "0.01" \

# accelerate but
    # --group_by_length \

# python \
#     run_speech_recognition_seq2seq_b.py \

# deepspeed \
#     --master_port 29001 \
#     --include localhost:4,5 \
#     run_speech_recognition_seq2seq_streaming_b.py \
#     --deepspeed ds_config_zero2_no_offload.json \

torchrun \
    --master_port 29001 \
    --nproc_per_node 5 \
    run_speech_recognition_seq2seq_c.py \
    --deepspeed ds_config_zero2_no_offload.json \
    --model_name_or_path $model_name_or_path \
    --use_auth_token \
    --train_file $train_file \
    --validation_file $validation_file \
    --audio_column_name "audio_filepath" \
    --max_duration_in_seconds "30" \
    --apply_audio_augmentation false \
    --background_noise_dir $noisedir \
    --audio_augmentation_prob "0.2" \
    --apply_spec_augment \
    --mask_time_prob "0.05" \
    --mask_time_length "10" \
    --mask_feature_prob "0.05" \
    --mask_feature_length "10" \
    --do_lower_case false \
    --remove_unused_columns false \
    --language "french" \
    --task "transcribe" \
    --preprocessing_num_workers "16" \
    --dataloader_num_workers "5" \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --run_name $run_name \
    --num_train_epochs "8" \
    --per_device_train_batch_size "56" \
    --per_device_eval_batch_size "28" \
    --gradient_accumulation_steps "1" \
    --optim "adamw_bnb_8bit" \
    --learning_rate "4.375e-6" \
    --warmup_ratio "0.05" \
    --lr_scheduler_type "cosine" \
    --weight_decay "0.01" \
    --fp16 \
    --gradient_checkpointing \
    --use_cache false \
    --freeze_feature_encoder false \
    --logging_steps "10" \
    --evaluation_strategy "steps" \
    --eval_steps "500" \
    --save_strategy "steps" \
    --save_steps "500" \
    --save_total_limit "3" \
    --metric_for_best_model "wer" \
    --greater_is_better false \
    --load_best_model_at_end \
    --predict_with_generate \
    --generation_num_beams "1" \
    --do_train \
    --do_eval
