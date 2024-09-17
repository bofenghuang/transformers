#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Short-form inference and evaluation

set -x -e

echo "START TIME: $(date)"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# py
myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED=1

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
# export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

# cuda
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# Set your number of GPUs here
num_gpus=4

# load models
# model_name_or_path="openai/whisper-small"
model_name_or_path="openai/whisper-large-v2"
# model_name_or_path="openai/whisper-large-v3"
# model_name_or_path="distil-whisper/distil-large-v3"
# model_name_or_path="/projects/bhuang/models/asr/whisper/it/whisper-large-v3-distil-it"

output_root_dir="./outputs/whisper_distil/it"

# take args
# model_name_or_path=$1
# output_root_dir=$2

tmp_model_id="$(echo "${model_name_or_path##*/}" | sed -e "s/[ |=/-]/_/g")"
outdir="$output_root_dir/$tmp_model_id/results"

# CMD
# CMD="python"
# CMD="accelerate launch"
CMD="accelerate launch --multi_gpu --num_processes=$num_gpus --main_process_port 29002"

# decoding options
infer_opt=(
    "--model_name_or_path $model_name_or_path"
    "--torch_dtype float16"
    "--attn_implementation flash_attention_2"
    "--language italian"
    "--task transcribe"
    "--return_timestamps False"
    "--generation_num_beams 1"
    "--per_device_eval_batch_size 128"
    "--dataloader_num_workers 8"
    "--num_processing_workers 32"
)
    # "--max_samples 100"
    # "--torch_dtype float32"
    # "--sort_by_length True"

# Join array elements into a single string separated by spaces
infer_opt_string="${infer_opt[*]}"

decode_suffix=_greedy
# decode_suffix=_beam5

# grep WER
# grep "%WER" ~/transformers/examples/pytorch/speech-recognition/outputs/hf_whisper/openai-whisper_large*/results_*_greedy/normalized_wer_summary/wer_summary.txt

# mcv validation
# tmp_outdir="${outdir}_mcv17_validation${decode_suffix}"

# $CMD infer_whisper_b.py \
#     $infer_opt_string \
#     --dataset_name "mozilla-foundation/common_voice_17_0" \
#     --dataset_config_name "it" \
#     --dataset_split_name "validation" \
#     --audio_column_name "audio" \
#     --output_file_path ${tmp_outdir}/predictions.json

# python scripts/compute_wer.py \
#     --input_file_path ${tmp_outdir}/predictions.json \
#     --target_column_name "sentence" \
#     --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt

# mcv test
tmp_outdir="${outdir}_mcv17_test${decode_suffix}"

$CMD infer_whisper_b.py \
    $infer_opt_string \
    --dataset_name "mozilla-foundation/common_voice_17_0" \
    --dataset_config_name "it" \
    --dataset_split_name "test" \
    --audio_column_name "audio" \
    --output_file_path ${tmp_outdir}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${tmp_outdir}/predictions.json \
    --target_column_name "sentence" \
    --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt

# mls
tmp_outdir="${outdir}_mls_test${decode_suffix}"

$CMD infer_whisper_b.py \
    $infer_opt_string \
    --dataset_name "facebook/multilingual_librispeech" \
    --dataset_config_name "italian" \
    --dataset_split_name "test" \
    --audio_column_name "audio" \
    --output_file_path ${tmp_outdir}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${tmp_outdir}/predictions.json \
    --target_column_name "text" \
    --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt

# voxpopuli
tmp_outdir="${outdir}_voxpopuli_test${decode_suffix}"

$CMD infer_whisper_b.py \
    $infer_opt_string \
    --dataset_name "facebook/voxpopuli" \
    --dataset_config_name "it" \
    --dataset_split_name "test" \
    --audio_column_name "audio" \
    --output_file_path ${tmp_outdir}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${tmp_outdir}/predictions.json \
    --target_column_name "raw_text" \
    --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt

# fleurs
tmp_outdir="${outdir}_fleurs_test${decode_suffix}"

$CMD infer_whisper_b.py \
    $infer_opt_string \
    --dataset_name "google/fleurs" \
    --dataset_config_name "it_it" \
    --dataset_split_name "test" \
    --audio_column_name "audio" \
    --output_file_path ${tmp_outdir}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${tmp_outdir}/predictions.json \
    --target_column_name "raw_transcription" \
    --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt

echo "END TIME: $(date)"
