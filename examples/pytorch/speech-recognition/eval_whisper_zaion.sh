#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

export CUDA_VISIBLE_DEVICES=4

# bh: load open sourced models
# model_name_or_path="openai/whisper-small"
model_name_or_path="openai/whisper-large-v2"
# model_name_or_path="bofenghuang/whisper-large-v2-french"
# model_name_or_path="bofenghuang/whisper-large-v2-cv11-french"

tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
outdir="./outputs/general/$tmp_model_id/results"

# bh: load local models
# model_name_or_path="outputs/general/whisper-medium-ft-french-lr6e6-bs256-augment"

# outdir=${model_name_or_path}/results

# decoding options
decode_opts=(--preprocessing_num_workers=4 --num_workers=1 --batch_size=8 --fp16)
# decode_opts=(--preprocessing_num_workers=4 --num_workers=1 --batch_size=8 --fp16 --gen_num_beams=5)

decode_suffix=_greedy
# decode_suffix=_beam5

#     --greedy \
#     --chunk_length_s 30.0 \
#     --stride_length_s 5.0 \
# --chunk_length_s 12.0 \
# --stride_length_s 2.0 \

# todo: lm, suppress_tokens
# bh: --num_workers 1 got "Segmentation fault" error with DataLoader \

python eval_whisper.py \
    --model_id $model_name_or_path \
    --test_csv_file "/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h.csv" \
    --id_column_name "ID" \
    --audio_column_name "wav" \
    --text_column_name "wrd" \
    --start_column_name "start" \
    --end_column_name "end" \
    --length_column_name "duration" \
    ${decode_opts[@]} \
    --log_outputs \
    --outdir ${outdir}_hmhm${decode_suffix}

python eval_whisper.py \
    --model_id $model_name_or_path \
    --test_csv_file "/projects/corpus/voice/zaion/carglass/data/20220111/data_wo_incomplet_words.csv" \
    --id_column_name "utt" \
    --audio_column_name "path" \
    --text_column_name "text" \
    --start_column_name "start" \
    --end_column_name "end" \
    --length_column_name "dur" \
    ${decode_opts[@]} \
    --log_outputs \
    --outdir ${outdir}_carglass${decode_suffix} # --test_csv_file "/projects/corpus/voice/zaion/carglass/data/20220120/data_wo_incomplet_words.csv" \

python eval_whisper.py \
    --model_id $model_name_or_path \
    --test_csv_file "/projects/corpus/voice/zaion/dekuple/2022_10_17/output/data/data_clean_without_words_with_dash.csv" \
    --id_column_name "ID" \
    --audio_column_name "wav" \
    --text_column_name "wrd" \
    --start_column_name "start" \
    --end_column_name "end" \
    --length_column_name "duration" \
    ${decode_opts[@]} \
    --log_outputs \
    --outdir ${outdir}_dekuple${decode_suffix}
