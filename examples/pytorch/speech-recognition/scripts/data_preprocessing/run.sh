#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

# prep data for mcv

set -e

echo "START TIME: $(date)"

myscriptspath="/home/bhuang/transformers_new/examples/pytorch/speech-recognition"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# cuda
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export CUDA_VISIBLE_DEVICES="4"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

input_file=/projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr100/train/train_espnet_yodas_manifest_sample.json

# python scripts/data_preprocessing/pre_normalize_text.py \
#     --input_file_path $input_file \
#     --output_file_path ${input_file%.*}_norm.json \
#     --text_column_name text \
#     --normalized_text_column_name normalized_text

output_file=${input_file/\/train\//\/train_concatenated\/}

# python scripts/data_preprocessing/concat_asr_examples.py \
#     --input_file_path $input_file \
#     --output_file_path $output_file \
#     --max_duration 15 \
#     --preprocessing_batch_size 1000 \
#     --preprocessing_num_workers 64

input_file=$output_file

# python scripts/infer_audio_lid.py \
#     --input_file_path $input_file \
#     --output_file_path ${input_file%.*}_lid.json \
#     --batch_size 64 \
#     --dataloader_num_workers 4

# python scripts/data_preprocessing/normalize_text.py \
#     --input_file_path $input_file \
#     --output_file_path ${input_file%.*}_norm.json \
#     --text_column_name text \
#     --normalized_text_column_name text_norm

model_name_or_path=bofenghuang/asr-wav2vec2-ctc-french
# model_name_or_path=/home/bhuang/transformers_new/examples/pytorch/speech-recognition/outputs/w2v2_ctc/wav2vec2_FR_7K_large_ft_ep80_bs256_lr1e4_specaugxtime03x10x01x64
# model_name_or_path=bofenghuang/phonemizer-wav2vec2-ctc-french

# python scripts/data_preprocessing/convert_grapheme_to_phoneme.py \
#     --input_file_path ${input_file%.*}_norm.json \
#     --output_file_path ${input_file%.*}_norm_phoneme.json \
#     --text_column_name text_norm \
#     --phoneme_column_name text_norm_phoneme \
#     --num_workers 32

python scripts/data_preprocessing/infer_wav2vec2_ctc_segmentation.py \
    --model_name_or_path $model_name_or_path \
    --dataset_file ${input_file%.*}_norm.json \
    --output_file_path ${input_file%.*}_norm_pred.json \
    --text_column_name text_norm \
    --sort_by_length True \
    --torch_dtype float32 \
    --attn_implementation sdpa \
    --greedy True \
    --compute_ctc_loss True \
    --batch_size 64 \
    --pad_to_multiple_of 8 \
    --dataloader_num_workers 8 \
    --score_min_mean_over_l 60 \
    # --max_samples 1024 \

# python scripts/data_preprocessing/infer_wav2vec2_ctc_segmentation.py \
#     --model_name_or_path $model_name_or_path \
#     --dataset_file ${input_file%.*}_norm_phoneme.json \
#     --output_file_path ${input_file%.*}_norm_phoneme_pred.json \
#     --text_column_name text_norm_phoneme \
#     --sort_by_length True \
#     --torch_dtype float32 \
#     --attn_implementation sdpa \
#     --greedy True \
#     --compute_ctc_loss True \
#     --batch_size 64 \
#     --pad_to_multiple_of 8 \
#     --dataloader_num_workers 8 \
#     --max_samples 1024 \

echo "END TIME: $(date)"
