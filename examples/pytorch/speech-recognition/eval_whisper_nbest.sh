#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

export CUDA_VISIBLE_DEVICES=0

# model_name_or_path="openai/whisper-small"
# model_name_or_path="outputs/hf_whisper_sprint/whisper-small-ft-lr6e6-bs256-adamw_bnb_8bit"
model_name_or_path="outputs/hf_whisper_sprint/whisper-small-ft-lr125e5-bs256-dropout01-casepunc"
# model_name_or_path="outputs/hf_whisper_sprint/whisper-medium-ft-lr6e6-bs256-steps4k-adamw_bnb_8bit"
# model_name_or_path="outputs/hf_whisper_sprint/whisper-medium-ft-lr6e6-bs256-steps4k-adamw_bnb_8bit-dropout005"
# model_name_or_path="outputs/hf_whisper_sprint/whisper-medium-ft-lr625e6-bs256-dropout01"
# model_name_or_path="outputs/hf_whisper_sprint/whisper-medium-ft-lr625e6-bs256-dropout01-casepunc"

# outdir="./outputs/hf_whisper_sprint/whisper-small/results_cv11"
outdir=${model_name_or_path}/results_cv11

#     --greedy \
#     --chunk_length_s 30.0 \
#     --stride_length_s 5.0 \
    # --chunk_length_s 12.0 \
    # --stride_length_s 2.0 \

# todo: beam, lm, normalizer into tokenizer, suppress_tokens, audio normalization

# bh: --num_workers 1 got "Segmentation fault" error with DataLoader \
python eval_whisper_nbest.py \
    --model_id $model_name_or_path \
    --language "french" \
    --task "transcribe" \
    --dataset "mozilla-foundation/common_voice_11_0" \
	--config "fr" \
	--split "test" \
    --batch_size 16 \
    --fp16 \
    --log_outputs \
    --outdir ${outdir}_greedysampling_nbest10
