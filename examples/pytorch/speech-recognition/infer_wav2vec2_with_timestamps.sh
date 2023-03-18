#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export CUDA_VISIBLE_DEVICES=4

# bh: load open sourced models
# model_name_or_path="openai/whisper-small"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/general/$tmp_model_id/results"

# bh: load local models
# model_name_or_path="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep30_with_hmhm_lm"
model_name_or_path="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep30_with_lm"
# model_name_or_path="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep30_with_carglass_lm"
# model_name_or_path="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep30_with_dekuple_lm"

# outdir=${model_name_or_path}/transcriptions

# zaion pipeline
./tmp_run_transcribe_ctc_with_timestamps_pipeline.py \
    --model_name_or_path $model_name_or_path \
    --input_dir "/projects/pipeline/tmp" \
    --input_filename "output.json" \
    --suffix "_transcribed" \
    --audio_column_name "wav" \
    --start_column_name "start" \
    --end_column_name "end" \
    --preprocessing_num_workers 4 \
    --num_workers 1 \
    --batch_size 16 \
    --chunk_length_s 30 \
    --stride_length_s 5 \
    --return_timestamps "word" \
    --return_nbest


# ./infer_wav2vec2_with_timestamps.py \
#     --model_name_or_path $model_name_or_path \
#     --input_file_path "/projects/corpus/voice/zaion/lfm/weak_supervised_momo/raw/oct_dec_head5.tsv" \
#     --prediction_file "/projects/corpus/voice/zaion/lfm/weak_supervised_momo/oct_dec_head5_text.tsv" \
#     --sampling_rate 8000 \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --preprocessing_num_workers 4 \
#     --num_workers 1 \
#     --batch_size 4 \
#     --chunk_length_s 30 \
#     --stride_length_s 5
