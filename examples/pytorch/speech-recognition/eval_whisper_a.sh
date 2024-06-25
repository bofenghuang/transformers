#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Long-form (chunked) inference and evaluation

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
export CUDA_VISIBLE_DEVICES="1"

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

# assistant_model_name_or_path="bofenghuang/whisper-large-v3-french-distil-dec2"

tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/hf_whisper_new/$tmp_model_id/results"
outdir="$output_root_dir/$tmp_model_id/results"

# load local models
# model_name_or_path="/projects/bhuang/models/asr/bofenghuang-whisper_large_v3_french_dec8_init_ft_ep12_bs256_lr1e4_no_specaugment"

# outdir=${model_name_or_path}/results

# decoding options
infer_opt=(
    "--model_name_or_path $model_name_or_path"
    "--language french"
    "--task transcribe"
    "--torch_dtype float16"
    "--per_device_eval_batch_size 16"
    "--generation_num_beams 1"
    "--num_processing_workers 4"
    "--chunk_length_s 30"
)
    # "--torch_dtype float32"
    # "--sort_by_length true"

# infer_opt=(
#     "--model_name_or_path $model_name_or_path"
#     "--language french"
#     "--task transcribe"
#     "--torch_dtype float16"
#     "--per_device_eval_batch_size 1"
#     "--generation_num_beams 1"
#     "--num_processing_workers 4"
#     "--chunk_length_s 30"
#     "--assistant_model_name_or_path $assistant_model_name_or_path"
#     "--attn_type None"
# )
    # "--chunk_length_s 15"


# Join array elements into a single string separated by spaces
infer_opt_string="${infer_opt[*]}"

# decode_suffix=_greedy
# decode_suffix=_beam5
# decode_suffix=_greedy_chunk15
decode_suffix=_greedy_chunk30
# decode_suffix=_greedy_chunk30_speculative_decoding

# todo: lm, suppress_tokens
# grep "%WER" ~/transformers/examples/pytorch/speech-recognition/outputs/hf_whisper/openai-whisper_large*/results_*_greedy/normalized_wer_summary/wer_summary.txt

# hf dev data
python infer_whisper_a.py \
    $infer_opt_string \
    --dataset_name "speech-recognition-community-v2/dev_data" \
    --dataset_config_name "fr" \
    --dataset_split_name "validation" \
    --audio_column_name "audio" \
    --output_file_path ${outdir}_hf_dev_data${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_hf_dev_data${decode_suffix}/predictions.json \
    --target_column_name "sentence" \
    --num_processing_workers 4 \
    --output_dir ${outdir}_hf_dev_data${decode_suffix} 2>&1 | tee ${outdir}_hf_dev_data${decode_suffix}/log.txt

# zaion dkpl
python infer_whisper_a.py \
    $infer_opt_string \
    --dataset_file "/projects/corpus/voice/zaion/dekuple/2022_10_17/output/data/data_clean_without_words_with_dash_by_conversation_merged.jsonl" \
    --audio_column_name "audio_filepath" \
    --duration_column_name "duration" \
    --output_file_path ${outdir}_zaion_dekuple${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_zaion_dekuple${decode_suffix}/predictions.json \
    --target_column_name "text" \
    --num_processing_workers 4 \
    --output_dir ${outdir}_zaion_dekuple${decode_suffix} 2>&1 | tee ${outdir}_zaion_dekuple${decode_suffix}/log.txt

# zaion dkpl by channel
python infer_whisper_a.py \
    $infer_opt_string \
    --dataset_file "/projects/corpus/voice/zaion/dekuple/2022_10_17/output/data/data_clean_without_words_with_dash_by_channel_merged.jsonl" \
    --audio_column_name "audio_filepath" \
    --duration_column_name "duration" \
    --output_file_path ${outdir}_zaion_dekuple_by_channel${decode_suffix}/predictions.json

python scripts/compute_wer.py \
    --input_file_path ${outdir}_zaion_dekuple_by_channel${decode_suffix}/predictions.json \
    --target_column_name "text" \
    --num_processing_workers 4 \
    --output_dir ${outdir}_zaion_dekuple_by_channel${decode_suffix} 2>&1 | tee ${outdir}_zaion_dekuple_by_channel${decode_suffix}/log.txt

# /projects/corpus/voice/zaion/carglass/data/20220120/data_wo_incomplet_words_by_conversation_merged.jsonl