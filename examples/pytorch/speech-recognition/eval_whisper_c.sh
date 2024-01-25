#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Short-form inference and evaluation

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
export CUDA_VISIBLE_DEVICES="1,2"

# load open sourced models
# model_name_or_path="openai/whisper-small"
# model_name_or_path="openai/whisper-large-v2"
# model_name_or_path="bofenghuang/whisper-large-v2-french"
# model_name_or_path="openai/whisper-large-v3"
# model_name_or_path="bofenghuang/whisper-large-v3-french"
# model_name_or_path="bofenghuang/whisper-large-v3-french-distil-dec16"
# model_name_or_path="bofenghuang/whisper-large-v3-french-distil-dec4"
# model_name_or_path="bofenghuang/whisper-large-v3-french-distil-dec2"

model_name_or_path=$1
output_root_dir=$2

tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/hf_whisper_new/$tmp_model_id/results"
outdir="$output_root_dir/$tmp_model_id/results"

# load local models
# model_name_or_path="/projects/bhuang/models/asr/bofenghuang-whisper_large_v3_french_dec8_init_ft_ep12_bs256_lr1e4_no_specaugment"

# outdir=${model_name_or_path}/results

# CMD
# CMD="python"
# CMD="accelerate launch --multi_gpu --num_processes=6"
# CMD="accelerate launch --mixed_precision=fp16 --num_processes=1"
CMD="accelerate launch --multi_gpu --num_processes=2 --main_process_port 29002"
# CMD="torchrun --master_port 29001 --nproc_per_node 2"

# decoding options
infer_opt=(
    "--model_name_or_path $model_name_or_path"
    "--language french"
    "--task transcribe"
    "--torch_dtype float16"
    "--per_device_eval_batch_size 32"
    "--generation_num_beams 1"
    "--dataloader_num_workers 8"
    "--num_processing_workers 32"
)
    # "--torch_dtype float32"
    # "--sort_by_length true"
    # "--chunk_length_s 30"

# Join array elements into a single string separated by spaces
infer_opt_string="${infer_opt[*]}"

decode_suffix=_greedy
# decode_suffix=_beam5

# todo: lm, suppress_tokens
# grep "%WER" ~/transformers/examples/pytorch/speech-recognition/outputs/hf_whisper/openai-whisper_large*/results_*_greedy/normalized_wer_summary/wer_summary.txt

$CMD infer_whisper_c.py \
    $infer_opt_string \
    --dataset_name "mozilla-foundation/common_voice_13_0" \
    --dataset_config_name "fr" \
    --dataset_split_name "test" \
    --audio_column_name "audio" \
    --output_file_path ${outdir}_mcv13${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_mcv13${decode_suffix}/predictions.json \
    --target_column_name "sentence" \
    --output_dir ${outdir}_mcv13${decode_suffix} 2>&1 | tee ${outdir}_mcv13${decode_suffix}/log.txt

# $CMD infer_whisper_c.py \
#     $infer_opt_string \
#     --dataset_file "/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-13/test_asr_mcv13_manifest_normalized_pnc.json" \
#     --id_column_name "id" \
#     --audio_column_name "audio_filepath" \
#     --text_column_name "text" \
#     --output_file_path ${outdir}_mcv13${decode_suffix}_norm/predictions.json

# python scripts/compute_wer.py \
#     --input_file_path ${outdir}_mcv13${decode_suffix}_norm/predictions.json \
#     --id_column_name "id" \
#     --target_column_name "text" \
#     --output_dir ${outdir}_mcv13${decode_suffix}_norm
# exit 0;

# mls
$CMD infer_whisper_c.py \
    $infer_opt_string \
    --dataset_name "facebook/multilingual_librispeech" \
    --dataset_config_name "french" \
    --dataset_split_name "test" \
    --output_file_path ${outdir}_mls${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_mls${decode_suffix}/predictions.json \
    --target_column_name "text" \
    --output_dir ${outdir}_mls${decode_suffix} 2>&1 | tee ${outdir}_mls${decode_suffix}/log.txt

# voxpopuli
$CMD infer_whisper_c.py \
    $infer_opt_string \
    --dataset_name "facebook/voxpopuli" \
    --dataset_config_name "fr" \
    --dataset_split_name "test" \
    --output_file_path ${outdir}_voxpopuli${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_voxpopuli${decode_suffix}/predictions.json \
    --target_column_name "raw_text" \
    --output_dir ${outdir}_voxpopuli${decode_suffix} 2>&1 | tee ${outdir}_voxpopuli${decode_suffix}/log.txt

# fleurs
$CMD infer_whisper_c.py \
    $infer_opt_string \
    --dataset_name "google/fleurs" \
    --dataset_config_name "fr_fr" \
    --dataset_split_name "test" \
    --id_column_name "ID" \
    --output_file_path ${outdir}_fleurs${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_fleurs${decode_suffix}/predictions.json \
    --id_column_name "ID" \
    --target_column_name "raw_transcription" \
    --output_dir ${outdir}_fleurs${decode_suffix} 2>&1 | tee ${outdir}_fleurs${decode_suffix}/log.txt

# african_accented_french
$CMD infer_whisper_c.py \
    $infer_opt_string \
    --dataset_name "gigant/african_accented_french" \
    --dataset_config_name "fr" \
    --dataset_split_name "test" \
    --output_file_path ${outdir}_african_accented_french${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_african_accented_french${decode_suffix}/predictions.json \
    --target_column_name "sentence" \
    --output_dir ${outdir}_african_accented_french${decode_suffix} 2>&1 | tee ${outdir}_african_accented_french${decode_suffix}/log.txt

# zaion hmhm
$CMD infer_whisper_c.py \
    $infer_opt_string \
    --dataset_file "/home/ywang/NeMo/examples/asr_zaion/190h_manifest/test_manifest_segment_16k_190h.json" \
    --audio_column_name "audio_filepath" \
    --duration_column_name "duration" \
    --output_file_path ${outdir}_zaion_test_hmhm${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_zaion_test_hmhm${decode_suffix}/predictions.json \
    --target_column_name "text" \
    --output_dir ${outdir}_zaion_test_hmhm${decode_suffix} 2>&1 | tee ${outdir}_zaion_test_hmhm${decode_suffix}/log.txt

# zaion carglass
$CMD infer_whisper_c.py \
    $infer_opt_string \
    --dataset_file "/home/ywang/NeMo/examples/asr_zaion/benchmark_robustess/carglass_5h/16k_segment.json" \
    --audio_column_name "audio_filepath" \
    --duration_column_name "duration" \
    --output_file_path ${outdir}_zaion_test_carglass${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_zaion_test_carglass${decode_suffix}/predictions.json \
    --target_column_name "text" \
    --output_dir ${outdir}_zaion_test_carglass${decode_suffix} 2>&1 | tee ${outdir}_zaion_test_carglass${decode_suffix}/log.txt

# zaion dekuple
$CMD infer_whisper_c.py \
    $infer_opt_string \
    --dataset_file "/home/ywang/NeMo/examples/asr_zaion/benchmark_robustess/dekuple_5h/16k_segment.json" \
    --audio_column_name "audio_filepath" \
    --duration_column_name "duration" \
    --output_file_path ${outdir}_zaion_test_dekuple${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_zaion_test_dekuple${decode_suffix}/predictions.json \
    --target_column_name "text" \
    --output_dir ${outdir}_zaion_test_dekuple${decode_suffix} 2>&1 | tee ${outdir}_zaion_test_dekuple${decode_suffix}/log.txt

# zaion lbpa
$CMD infer_whisper_c.py \
    $infer_opt_string \
    --dataset_file "/home/ywang/NeMo/examples/asr_zaion/benchmark_robustess/lbpa_2.35h/16k_segment.json" \
    --audio_column_name "audio_filepath" \
    --duration_column_name "duration" \
    --output_file_path ${outdir}_zaion_test_lbpa${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_zaion_test_lbpa${decode_suffix}/predictions.json \
    --target_column_name "text" \
    --output_dir ${outdir}_zaion_test_lbpa${decode_suffix} 2>&1 | tee ${outdir}_zaion_test_lbpa${decode_suffix}/log.txt
