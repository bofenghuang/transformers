#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Long-form (chunked/sequential) inference and evaluation

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
export CUDA_VISIBLE_DEVICES="3"

# load models
# multilingual
# model_name_or_path="openai/whisper-small"
# model_name_or_path="openai/whisper-large-v2"
# model_name_or_path="openai/whisper-large-v3"

# en
# model_name_or_path="distil-whisper/distil-large-v3"

# it
# model_name_or_path="/projects/bhuang/models/asr/whisper/it/whisper-large-v3-distil-it"

# fr
# model_name_or_path="eustlb/distil-large-v3-fr"
# model_name_or_path="/projects/bhuang/models/asr/whisper/fr/whisper-large-v3-distil-fr"

# take args
model_name_or_path=$1

# assistant_model_name_or_path="bofenghuang/whisper-large-v3-french-distil-dec2"

# language
# lang="$lang"
lang="french"

output_root_dir="./outputs/whisper_distil/$lang"
# output_root_dir=$2

tmp_model_id="$(echo "${model_name_or_path##*/}" | sed -e "s/[ |=/-]/_/g")"
outdir="$output_root_dir/$tmp_model_id/results"

# decoding options

    # "--torch_dtype float32"
    # "--sort_by_length True"

# chunk
eval_chunk_function() {
    test_data_file=$1
    # echo "test_data_file: $1"

    infer_opt=(
        "--model_name_or_path $model_name_or_path"
        "--torch_dtype float16"
        "--attn_implementation flash_attention_2"
        "--language $lang"
        "--task transcribe"
        "--return_timestamps False"
        "--generation_num_beams 1"
        "--per_device_eval_batch_size 128"
        "--dataloader_num_workers 8"
        "--num_processing_workers 32"
        "--chunk_length_s 30"
    )
    decode_suffix=_greedy_chunk30

    # Join array elements into a single string separated by spaces
    infer_opt_string="${infer_opt[*]}"

    test_name="${test_data_file##*/}"
    test_name="${test_name%.*}"
    test_name="${test_name%_manifest}"
    tmp_outdir="${outdir}_${test_name}${decode_suffix}"

    python infer_whisper_a.py \
        $infer_opt_string \
        --dataset_file $test_data_file \
        --audio_column_name "audio_filepath" \
        --output_file_path ${tmp_outdir}/predictions.json

    python scripts/compute_wer.py \
        --input_file_path ${tmp_outdir}/predictions.json \
        --language $lang \
        --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt
}

# sequential
eval_sequential_function() {
    test_data_file=$1
    # echo "test_data_file: $1"

    infer_opt=(
        "--model_name_or_path $model_name_or_path"
        "--torch_dtype float16"
        "--attn_implementation flash_attention_2"
        "--language $lang"
        "--task transcribe"
        "--return_timestamps True"
        "--generation_num_beams 1"
        "--per_device_eval_batch_size 1"
        "--dataloader_num_workers 1"
        "--num_processing_workers 32"
    )
    decode_suffix=_greedy_sequential

    # Join array elements into a single string separated by spaces
    infer_opt_string="${infer_opt[*]}"

    test_name="${test_data_file##*/}"
    test_name="${test_name%.*}"
    test_name="${test_name%_manifest}"
    tmp_outdir="${outdir}_${test_name}${decode_suffix}"

    python infer_whisper_a.py \
        $infer_opt_string \
        --dataset_file $test_data_file \
        --audio_column_name "audio_filepath" \
        --output_file_path ${tmp_outdir}/predictions.json

    python scripts/compute_wer.py \
        --input_file_path ${tmp_outdir}/predictions.json \
        --language $lang \
        --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt
}

# chunk, speculative_decoding
# eval_chunk_speculative_function() {
#     test_data_file=$1
#     # echo "test_data_file: $1"

#     infer_opt=(
#         "--model_name_or_path $model_name_or_path"
#         "--torch_dtype float16"
#         "--attn_implementation flash_attention_2"
#         "--language $lang"
#         "--task transcribe"
#         "--return_timestamps False"
#         "--generation_num_beams 1"
#         "--per_device_eval_batch_size 1"
#         "--chunk_length_s 30"
#         "--num_processing_workers 4"
#         "--assistant_model_name_or_path $assistant_model_name_or_path"
#     )
#     decode_suffix=_greedy_chunk30_speculative_decoding

#     # Join array elements into a single string separated by spaces
#     infer_opt_string="${infer_opt[*]}"

#     test_name="${test_data_file##*/}"
#     test_name="${test_name%.*}"
#     test_name="${test_name%_manifest}"
#     tmp_outdir="${outdir}_${test_name}${decode_suffix}"

#     python infer_whisper_a.py \
#         $infer_opt_string \
#         --dataset_file $test_data_file \
#         --audio_column_name "audio_filepath" \
#         --output_file_path ${tmp_outdir}/predictions.json

#     python scripts/compute_wer.py \
#         --input_file_path ${tmp_outdir}/predictions.json \
#         --language $lang \
#         --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt
# }

# grep WER
# grep "%WER" outputs/whisper_distil/french/whisper_large_v3_distil_fr/results_*/wer_summary_normalized/wer_summary.txt

# it
# test_data_files=(
# "/projects/bhuang/corpus/speech/nemo_manifests/speech-recognition-community-v2/dev_data/it/validation/validation_speech-recognition-community-v2_dev_data_manifest.json"
# )

# fr
test_data_files=(
# "/projects/bhuang/corpus/speech/nemo_manifests/speech-recognition-community-v2/dev_data/fr/validation/validation_speech-recognition-community-v2_dev_data_manifest.json"
"/projects/bhuang/corpus/speech/zaion/dekuple_5h_merged/test_zaion_dekuple_5h_merged_by_channel_manifest.json"
"/projects/bhuang/corpus/speech/zaion/dekuple_5h_merged/test_zaion_dekuple_5h_merged_by_conversation_manifest.json"
)

# Iterate over the array and apply the function to each element
for test_data_file in "${test_data_files[@]}"; do
    eval_chunk_function "$test_data_file"
    eval_sequential_function "$test_data_file"
done

# hf dev data
# tmp_outdir="${outdir}_hf_dev_data${decode_suffix}"

# python infer_whisper_a.py \
#     $infer_opt_string \
#     --dataset_name "speech-recognition-community-v2/dev_data" \
#     --dataset_config_name "it" \
#     --dataset_split_name "validation" \
#     --audio_column_name "audio" \
#     --output_file_path ${tmp_outdir}/predictions.json

# python scripts/compute_wer.py \
#     --input_file_path ${tmp_outdir}/predictions.json \
#     --target_column_name "sentence" \
#     --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt

echo "END TIME: $(date)"
