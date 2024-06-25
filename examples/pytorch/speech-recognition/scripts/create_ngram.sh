#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

# export CUDA_VISIBLE_DEVICES=5,4,3,2
# export CUDA_VISIBLE_DEVICES=1

# todo: train+validation
# python create_ngram.py \
#     --dataset "mozilla-foundation/common_voice_9_0" \
#     --config "fr" \
#     --split "train" \
# 	--text_column_name="sentence" \
#     --outdir "outputs/common_voice_9_0_fr/language_model"

python scripts/create_ngram.py \
	--dataset "polinaeterna/voxpopuli" \
    --config "fr" \
    --split "train+validation" \
	--text_column_name="normalized_text" \
    --outdir "outputs/all/language_model"
