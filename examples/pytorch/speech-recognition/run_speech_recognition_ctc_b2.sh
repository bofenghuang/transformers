#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

set -e

echo "START TIME: $(date)"

# Debugging flags (optional)
# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONFAULTHANDLER=1

# export PYTHONPATH="$PYTHONPATH:/home/bhuang/my-scripts"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

# wandb
# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT="asr-w2v2ctc"

# cuda
# https://github.com/microsoft/DeepSpeed/issues/662
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# Set your number of GPUs here
num_gpus=4

# https://pytorch.org/docs/stable/elastic/run.html
# export HOST_NODE_ADDR="localhost:29001"

# cmd
# CMD="python"
# CMD="torchrun --master_port=29001 --nproc_per_node=$num_gpus"
# CMD="deepspeed --master_port 29001 --include localhost:1,2"
CMD="accelerate launch --multi_gpu --num_processes=$num_gpus --main_process_port 29002"

# models
# model_name_or_path="facebook/wav2vec2-large-xlsr-53"
# model_name_or_path="facebook/wav2vec2-xls-r-300m"
# model_name_or_path="facebook/wav2vec2-xls-r-1b"
model_name_or_path="LeBenchmark/wav2vec2-FR-7K-large"
# model_name_or_path="LeBenchmark/wav2vec2-FR-14K-large"
# model_name_or_path="LeBenchmark/wav2vec2-FR-14K-xlarge"

train_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_norm.json"
validation_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/validation/validation_mozilla-foundation_common_voice_17_0_manifest_norm.json"

noisedir="/projects/bhuang/corpus/speech/musan_wo_speech"

tmp_model_id="$(echo "${model_name_or_path##*/}" | sed -e "s/[ |=/-]/_/g")"
run_name="${tmp_model_id}_ft_ep80_bs256_lr1e4_specaugxtime03x10x01x64"
output_dir="./outputs/w2v2_ctc_mcv/$run_name"

# multiple gpus - layerdropout vs gradient_checkpointing
# --ddp_find_unused_parameters true \
# --gradient_checkpointing \

# --layerdrop "0" \
# --feat_proj_dropout "0" \
# --attention_dropout "0.05" \
# --activation_dropout "0" \
# --hidden_dropout "0.05" \
# --final_dropout "0.05" \

    # --adam_beta2 "0.95" \
    # --weight_decay "0.01" \

    # --attn_implementation "sdpa" \
    # --torch_compile \

    # --max_train_samples "8192" \
    # --max_eval_samples "1024" \

$CMD run_speech_recognition_ctc_b.py \
    --model_name_or_path $model_name_or_path \
    --train_file $train_file \
    --validation_file $validation_file \
    --audio_column_name "audio_filepath" \
    --text_column_name "text_norm" \
    --max_duration_in_seconds "30" \
    --min_duration_in_seconds "1" \
    --remove_unused_columns false \
    --apply_audio_augmentation false \
    --background_noise_dir $noisedir \
    --audio_augmentation_prob "0.2" \
    --mask_time_prob "0.3" \
    --mask_time_length "10" \
    --mask_feature_prob "0.1" \
    --mask_feature_length "64" \
    --preprocessing_num_workers "16" \
    --dataloader_num_workers "8" \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --run_name $run_name \
    --num_train_epochs "80" \
    --per_device_train_batch_size "64" \
    --per_device_eval_batch_size "64" \
    --gradient_accumulation_steps "1" \
    --learning_rate "1e-4" \
    --warmup_ratio "0.05" \
    --lr_scheduler_type "cosine" \
    --fp16 \
    --gradient_checkpointing \
    --ctc_zero_infinity \
    --freeze_feature_encoder \
    --logging_steps "10" \
    --eval_strategy "steps" \
    --eval_steps "200" \
    --save_strategy "steps" \
    --save_steps "200" \
    --save_total_limit "3" \
    --eval_metrics "wer" "cer" \
    --metric_for_best_model "wer" \
    --greater_is_better false \
    --load_best_model_at_end \
    --do_train \
    --do_eval

echo "END TIME: $(date)"
