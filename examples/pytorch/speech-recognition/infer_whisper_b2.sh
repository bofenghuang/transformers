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

# language
# lang="italian"
lang="french"

output_root_dir="./outputs/whisper_distil/$lang"
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
    "--language $lang"
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
# grep "%WER" outputs/whisper_distil/french/whisper_large_v3_distil_fr/results_*/wer_summary_normalized/wer_summary.txt

eval_function() {
    test_data_file=$1
    # echo "test_data_file: $1"

    test_name="${test_data_file##*/}"
    test_name="${test_name%.*}"
    test_name="${test_name%_manifest}"
    tmp_outdir="${outdir}_${test_name}${decode_suffix}"

    $CMD infer_whisper_b.py \
        $infer_opt_string \
        --dataset_file $test_data_file \
        --audio_column_name "audio_filepath" \
        --output_file_path ${tmp_outdir}/predictions.json

    python scripts/compute_wer.py \
        --input_file_path ${tmp_outdir}/predictions.json \
        --language $lang \
        --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt
}

# it
# test_data_files=(
# "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/test/test_mozilla-foundation_common_voice_17_0_manifest.json"
# "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/italian/test/test_facebook_multilingual_librispeech_manifest.json"
# "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/it/test/test_facebook_voxpopuli_manifest.json"
# "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/it_it/test/test_google_fleurs_manifest.json"
# )

# fr
test_data_files=(
"/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/test/test_mozilla-foundation_common_voice_17_0_manifest.json"
"/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/test/test_facebook_multilingual_librispeech_manifest.json"
"/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/test/test_facebook_voxpopuli_manifest.json"
"/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/fr-fr/test/test_mtedx_asr_manifest.json"
"/projects/bhuang/corpus/speech/nemo_manifests/gigant/african_accented_french/test/test_gigant_african_accented_french_manifest.json"
"/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/test/test_google_fleurs_manifest.json"
# "/projects/bhuang/corpus/speech/nemo_manifests/BrunoHays/Accueil_UBS/test/test_BrunoHays_Accueil_UBS_manifest.json"
"/projects/bhuang/corpus/speech/zaion/hmhm_10h/test_zaion_hmhm_10h_manifest.json"
"/projects/bhuang/corpus/speech/zaion/carglass_5h/test_zaion_carglass_5h_manifest.json"
"/projects/bhuang/corpus/speech/zaion/dekuple_5h/test_zaion_dekuple_5h_manifest.json"
"/projects/bhuang/corpus/speech/zaion/lbpa_2.35h/test_zaion_lbpa_2h_manifest.json"
)

# Iterate over the array and apply the function to each element
for test_data_file in "${test_data_files[@]}"; do
    eval_function "$test_data_file"
done

echo "END TIME: $(date)"

