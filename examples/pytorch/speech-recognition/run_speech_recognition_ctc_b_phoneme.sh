#!/usr/bin/env bash

# HF cache
export HF_HOME="/projects/bhuang/.cache/huggingface"

# WANDB related
# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
# export WANDB_PROJECT="hf-asr-general"
export WANDB_PROJECT="hf-phoneme-general"

# export PYTHONPATH="$PYTHONPATH:/home/bhuang/my-scripts"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS="1"

# https://github.com/microsoft/DeepSpeed/issues/662
export CUDA_VISIBLE_DEVICES="0,1,2,3"

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

# models
# model_name_or_path="facebook/wav2vec2-large-xlsr-53"
# model_name_or_path="facebook/wav2vec2-xls-r-300m"
# model_name_or_path="facebook/wav2vec2-xls-r-1b"
model_name_or_path="LeBenchmark/wav2vec2-FR-7K-large"
# model_name_or_path="LeBenchmark/wav2vec2-FR-14K-large"
# model_name_or_path="LeBenchmark/wav2vec2-FR-14K-xlarge"

train_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/train/train_mozilla-foundation_common_voice_13_0_manifest_normalized_phoneme.json"
validation_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/validation/validation_mozilla-foundation_common_voice_13_0_manifest_normalized_phoneme.json"

noisedir="/home/bhuang/corpus/speech/public/musan_wo_speech"

run_name="wav2vec2-lebenchmark-fr-7k-large-ft-ep20-bs256-lr1e4"
output_dir="./outputs/phoneme/$run_name"

# multiple gpus - layerdropout vs gradient_checkpointing
# --ddp_find_unused_parameters true \
# --gradient_checkpointing \

    # --max_train_samples 10000 \
    # --max_eval_samples 1024 \

# python \
#     run_speech_recognition_ctc_b.py \

# deepspeed \
#     --master_port 29001 \
#     --include localhost:1,2 \
#     run_speech_recognition_ctc_b.py \
#     --deepspeed ds_config.json \

torchrun \
    --master_port 29001 \
    --nproc_per_node 4 \
    run_speech_recognition_ctc_b.py \
    --model_name_or_path $model_name_or_path \
    --use_auth_token \
    --train_file $train_file \
    --validation_file $validation_file \
    --audio_column_name "audio_filepath" \
    --text_column_name "phoneme" \
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
    --num_train_epochs "20" \
    --per_device_train_batch_size "32" \
    --per_device_eval_batch_size "32" \
    --gradient_accumulation_steps "2" \
    --learning_rate "1e-4" \
    --warmup_ratio "0.05" \
    --lr_scheduler_type "cosine" \
    --weight_decay "0.01" \
    --fp16 \
    --gradient_checkpointing \
    --ctc_zero_infinity \
    --freeze_feature_encoder \
    --layerdrop "0" \
    --feat_proj_dropout "0" \
    --attention_dropout "0.05" \
    --activation_dropout "0" \
    --hidden_dropout "0.05" \
    --final_dropout "0.05" \
    --mask_time_prob "0.05" \
    --mask_time_length "10" \
    --mask_feature_prob "0.05" \
    --mask_feature_length "10" \
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
