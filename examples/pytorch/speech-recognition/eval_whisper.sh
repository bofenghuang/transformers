#!/usr/bin/env bash

export TRANSFORMERS_CACHE="/projects/bhuang/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES="1"

# model_name_or_path="openai/whisper-small"
model_name_or_path="openai/whisper-large-v3"
# model_name_or_path="bofenghuang/whisper-large-v2-french"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/hf_whisper/$tmp_model_id/results"

# model_name_or_path="outputs/hf_whisper_sprint/whisper-medium-ft-lr6e6-bs256-dropout01-punct"
model_name_or_path="outputs/hf_whisper/whisper-large-v3-ft-french-pnc-ep5-bs280-lr4e6-wd001-audioaug-specaug"

outdir=${model_name_or_path}/results


CMD="python"
# CMD="accelerate launch --num_processes=4"

decode_suffix=_greedy
# decode_suffix=_beam5

infer_opt=(
    "--model_name_or_path $model_name_or_path"
    "--language french"
    "--task transcribe"
    "--per_device_eval_batch_size 32"
    "--generation_num_beams 1"
)
# Join array elements into a single string separated by spaces
infer_opt_string="${infer_opt[*]}"

# todo: lm, suppress_tokens
# $CMD infer_whisper.py \
#     $infer_opt_string \
#     --dataset_name "mozilla-foundation/common_voice_13_0" \
#     --dataset_config_name "fr" \
#     --dataset_split_name "test" \
#     --id_column_name "id" \
#     --audio_column_name "audio" \
#     --text_column_name "sentence" \
#     --output_file_path ${outdir}_mcv13${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_mcv13${decode_suffix}/predictions.json \
    --id_column_name "id" \
    --target_column_name "sentence" \
    --output_dir ${outdir}_mcv13${decode_suffix}

# $CMD infer_whisper.py \
#     $infer_opt_string \
#     --dataset_file "/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-13/test_asr_mcv13_manifest_normalized_pnc.json" \
#     --id_column_name "id" \
#     --audio_column_name "audio_filepath" \
#     --text_column_name "text" \
#     --output_file_path ${outdir}_mcv13${decode_suffix}_norm/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_mcv13${decode_suffix}_norm/predictions.json \
    --id_column_name "id" \
    --target_column_name "text" \
    --output_dir ${outdir}_mcv13${decode_suffix}_norm
exit
$CMD infer_whisper.py \
    $infer_opt_string \
    --dataset_name "facebook/multilingual_librispeech" \
    --dataset_config_name "french" \
    --dataset_split_name "test" \
    --output_file_path ${outdir}_mls${decode_suffix}/predictions.json

$CMD infer_whisper.py \
    $infer_opt_string \
    --dataset_name "facebook/voxpopuli" \
    --dataset_config_name "fr" \
    --dataset_split_name "test" \
    --output_file_path ${outdir}_voxpopuli${decode_suffix}/predictions.json

    # --text_column_name="normalized_text" \

$CMD infer_whisper.py \
    $infer_opt_string \
    --dataset_name "google/fleurs" \
    --dataset_config_name "fr_fr" \
    --dataset_split_name "test" \
    --output_file_path ${outdir}_fleurs${decode_suffix}/predictions.json

    # --text_column_name="transcription" \

$CMD infer_whisper.py \
    $infer_opt_string \
    --dataset_name "gigant/african_accented_french" \
    --dataset_config_name "fr" \
    --dataset_split_name "test" \
    --output_file_path ${outdir}_african_accented_french${decode_suffix}/predictions.json
