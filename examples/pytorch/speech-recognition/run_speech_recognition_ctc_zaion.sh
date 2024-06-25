#!/usr/bin/env bash

export TRANSFORMERS_CACHE="/projects/bhuang/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=hf-asr-hmhm

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=0,1,4,5

# trainfile="/home/bhuang/corpus/speech/internal/hm_hm/train_hmhm_190h.csv"
# trainfile="/home/bhuang/corpus/speech/internal/hm_hm/train_hmhm_190h_wo_space_after_apostrophe.csv"
# trainfile="/home/bhuang/corpus/speech/internal/hm_hm_merged/train_hmhm_merged_and_raw.csv"
trainfile="/home/bhuang/corpus/speech/internal/hm_hm_merged/train_hmhm_merged_and_raw_wo_space_after_apostrophe.csv"
# validationfile="/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h.csv"
validationfile="/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h_wo_space_after_apostrophe.csv"

# models
# --model_name_or_path="facebook/wav2vec2-large-960h" \
# --model_name_or_path="facebook/wav2vec2-conformer-rel-pos-large" \
# --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
# --model_name_or_path="facebook/wav2vec2-xls-r-300m" \
# --model_name_or_path="facebook/wav2vec2-xls-r-1b" \
# --model_name_or_path="facebook/wav2vec2-xls-r-2b" \
# --model_name_or_path="LeBenchmark/wav2vec2-FR-3K-large" \
# --model_name_or_path="LeBenchmark/wav2vec2-FR-7K-large" \
# --model_name_or_path="facebook/hubert-xlarge-ll60k" \
# --model_name_or_path="microsoft/unispeech-1350-en-353-fr-ft-1h" \
# --model_name_or_path="microsoft/wavlm-large" \

# specify trained tokenizer
# --model_name_or_path="jonatasgrosman/wav2vec2-xls-r-1b-french" \
# --tokenizer_name_or_path="jonatasgrosman/wav2vec2-xls-r-1b-french" \
# --model_name_or_path="bofenghuang/asr-wav2vec2-ctc-french" \
# --tokenizer_name_or_path="bofenghuang/asr-wav2vec2-ctc-french" \

# warmup
# --warmup_steps="500" \
# --warmup_ratio="0.1" \

# BS
# 32 for wav2vec2-FR-7K-large, 16 for wav2vec2-xls-r-1b
# global bs 128

# python run_speech_recognition_ctc_b.py \
torchrun --nproc_per_node 4 run_speech_recognition_ctc_b.py \
    --train_file=$trainfile \
    --validation_file=$validationfile \
    --dataset_config_name="fr" \
    --text_column_name="wrd" \
    --audio_column_name="wav" \
    --length_column_name="input_length" \
    --group_by_length \
    --max_duration_in_seconds="30" \
    --min_duration_in_seconds="1" \
    --preprocessing_num_workers="8" \
    --dataloader_num_workers="4" \
    --do_speech_augment \
    --model_name_or_path="LeBenchmark/wav2vec2-FR-7K-large" \
    --output_dir="./outputs/hmhm_merged_and_raw/lebenchmark-wav2vec2_fr_7k_large-ft-ep30-bs128-lr1e4-wd1e2-drp" \
    --run_name="lebenchmark-wav2vec2_fr_7k_large-ft-ep30-bs128-lr1e4-wd1e2-drp" \
    --overwrite_output_dir \
    --num_train_epochs="30" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --gradient_accumulation_steps="1" \
    --learning_rate="1e-4" \
	--warmup_steps="800" \
	--weight_decay="0.01" \
    --logging_steps="25" \
    --evaluation_strategy="steps" \
    --eval_steps="500" \
    --save_strategy="steps" \
    --save_steps="500" \
    --save_total_limit="3" \
    --metric_for_best_model="eval_wer" \
    --greater_is_better false \
    --load_best_model_at_end \
    --ctc_zero_infinity \
    --freeze_feature_encoder \
    --fp16 \
    --gradient_checkpointing \
    --layerdrop="0.1" \
    --feat_proj_dropout="0" \
    --attention_dropout="0" \
    --activation_dropout="0.1" \
    --hidden_dropout="0" \
    --final_dropout="0" \
    --mask_time_prob="0.05" \
    --mask_time_length="2" \
    --mask_feature_prob="0.05" \
    --mask_feature_length="10" \
    --do_train --do_eval

# best hmhm
# --layerdrop="0.1" \
# --feat_proj_dropout="0" \
# --attention_dropout="0" \
# --activation_dropout="0.1" \
# --hidden_dropout="0" \
# --final_dropout="0" \
