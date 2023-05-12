#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export TRANSFORMERS_CACHE="/projects/bhuang/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

export CUDA_VISIBLE_DEVICES=3

myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED=1

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
# ./tmp_run_transcribe_ctc_with_timestamps_pipeline.py \
#     --model_name_or_path $model_name_or_path \
#     --input_dir "/projects/pipeline/tmp" \
#     --input_filename "output.json" \
#     --suffix "_transcribed" \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --preprocessing_num_workers 4 \
#     --num_workers 1 \
#     --batch_size 16 \
#     --chunk_length_s 30 \
#     --stride_length_s 5 \
#     --return_timestamps "word" \
#     --return_nbest

# ./infer_wav2vec2.py \
#     --model_name_or_path $model_name_or_path \
#     --input_file_path "/projects/corpus/voice/zaion/lfm/weak_supervised_momo/raw/oct_dec_head5.tsv" \
#     --prediction_file "/projects/corpus/voice/zaion/lfm/weak_supervised_momo/oct_dec_head5_text.tsv" \
#     --sampling_rate 8000 \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --preprocessing_num_workers 4 \
#     --num_workers 1 \
#     --batch_size 16 \
#     --chunk_length_s 30 \
#     --stride_length_s 5

# ./infer_wav2vec2.py \
#     --model_name_or_path $model_name_or_path \
#     --input_file_path "/projects/ilaaridh/weakAnnotation/data/dec_14_25/data.tsv" \
#     --prediction_file "/projects/ilaaridh/weakAnnotation/data/dec_14_25/data_wav2vec.tsv" \
#     --sampling_rate 8000 \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --preprocessing_num_workers 4 \
#     --num_workers 1 \
#     --batch_size 16

# new

# ./infer_wav2vec2.py \
#     --model_name_or_path $model_name_or_path \
#     --input_file_path "/projects/ilaaridh/weakAnnotation/data/dec_14_25/data.tsv" \
#     --output_file_path "/projects/ilaaridh/weakAnnotation/data/dec_14_25/data_wav2vec.tsv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --text_column_name "text" \
#     --greedy true \
#     --fp16 true \
#     --dataloader_num_workers 4 \
#     --batch_size 16

# greedy
# ./infer_wav2vec2.py \
#     --model_name_or_path $model_name_or_path \
#     --input_file_path "/projects/ilaaridh/weakAnnotation/data/dec_9_14/data.tsv" \
#     --output_file_path "/projects/ilaaridh/weakAnnotation/data/dec_9_14/data_wav2vec.tsv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --text_column_name "text" \
#     --greedy true \
#     --fp16 true \
#     --dataloader_num_workers 4 \
#     --batch_size 16

# beam search
./infer_wav2vec2.py \
    --model_name_or_path $model_name_or_path \
    --input_file_path "/projects/corpus/voice/zaion/edenred/2023-05-10/data.tsv" \
    --output_file_path "/projects/corpus/voice/zaion/edenred/2023-05-10/data_wav2vec.tsv" \
    --id_column_name "ID" \
    --audio_column_name "wav" \
    --start_column_name "start" \
    --end_column_name "end" \
    --text_column_name "text" \
    --greedy false \
    --fp16 true \
    --dataloader_num_workers 4 \
    --batch_size 16
