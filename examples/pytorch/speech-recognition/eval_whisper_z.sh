#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
# export CUDA_VISIBLE_DEVICES="0"

# load open sourced models
# model_name_or_path="openai/whisper-small"
# model_name_or_path="openai/whisper-large-v3"
# model_name_or_path="bofenghuang/whisper-large-v2-french"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/hf_whisper/$tmp_model_id/results"

# load local models
# model_name_or_path="outputs/hf_whisper_sprint/whisper-medium-ft-lr6e6-bs256-dropout01-punct"
model_name_or_path="outputs/hmhm_merged_and_raw/bofenghuang-whisper_large_v2_french-ft-ep2-bs256-lr4e6-wd1e2-aug-specaug"
# model_name_or_path="outputs/hf_whisper/whisper-large-v3-ft-french-pnc-ep5-bs280-lr4e6-wd001-audioaug-specaug"
# model_name_or_path="outputs/hf_whisper/tmp_model"

outdir=${model_name_or_path}/results

# CMD
# CMD="python"
CMD="accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=5"
# CMD="accelerate launch --mixed_precision=fp16 --num_processes=1"
# CMD="accelerate launch --multi_gpu --num_processes=2"
# CMD="torchrun --master_port 29001 --nproc_per_node 2"

# decoding options
infer_opt=(
    "--model_name_or_path $model_name_or_path"
    "--sort_by_length true"
    "--language french"
    "--task transcribe"
    "--generation_num_beams 1"
    "--per_device_eval_batch_size 32"
    "--dataloader_num_workers 5"
    "--num_processing_workers 64"
)
    # "--chunk_length_s 30"

# Join array elements into a single string separated by spaces
infer_opt_string="${infer_opt[*]}"

decode_suffix=_greedy
# decode_suffix=_beam5

$CMD infer_whisper_c.py \
    $infer_opt_string \
    --dataset_file "/projects/corpus/voice/zaion/renault/2023-11-24/BO_max30s.json" \
    --id_column_name "id" \
    --audio_column_name "audio_filepath" \
    --start_column_name "offset" \
    --duration_column_name "duration" \
    --output_file_path ${outdir}_renault_bo${decode_suffix}/predictions.json


# python eval_whisper.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --text_column_name "wrd" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "duration" \
#     ${decode_opts[@]} \
#     --log_outputs \
#     --outdir ${outdir}_hmhm${decode_suffix}

python eval_whisper.py \
    --model_id $model_name_or_path \
    --test_csv_file "/home/bhuang/transformers/examples/pytorch/speech-recognition/data/hmhm_max30s_max448tokens/test.csv" \
    --id_column_name "ID" \
    --audio_column_name "wav" \
    --text_column_name "wrd" \
    --start_column_name "start" \
    --end_column_name "end" \
    --length_column_name "duration" \
    ${decode_opts[@]} \
    --log_outputs \
    --outdir ${outdir}_hmhm_max30s_max448tokens${decode_suffix}

# python eval_whisper.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/carglass/data/20220111/data_wo_incomplet_words.csv" \
#     --id_column_name "utt" \
#     --audio_column_name "path" \
#     --text_column_name "text" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "dur" \
#     ${decode_opts[@]} \
#     --log_outputs \
#     --outdir ${outdir}_carglass${decode_suffix} # --test_csv_file "/projects/corpus/voice/zaion/carglass/data/20220120/data_wo_incomplet_words.csv" \

# python eval_whisper.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/dekuple/2022_10_17/output/data/data_clean_without_words_with_dash.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --text_column_name "wrd" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "duration" \
#     ${decode_opts[@]} \
#     --log_outputs \
#     --outdir ${outdir}_dekuple${decode_suffix}
