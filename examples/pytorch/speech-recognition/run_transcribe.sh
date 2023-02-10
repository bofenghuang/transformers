#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
# export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

# export CUDA_VISIBLE_DEVICES=5,4,3,2
export CUDA_VISIBLE_DEVICES=1

# model_name_or_path="outputs/common_voice_9_0_fr/wav2vec2-xls-r-1b-ft-ep10_with_lm"
# model_name_or_path="outputs/hmhm/wav2vec2-xls-r-1b-ft-medLR-ep30_with_lm"
# model_name_or_path="outputs/hmhm/wav2vec2-FR-7K-large-ft-medLR-ep30_with_lm"
# model_name_or_path="outputs/hmhm_merged/wav2vec2-FR-7K-large_ft_ep30_with_lm"
# model_name_or_path="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_with_lm"
model_name_or_path="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep30_with_lm"

# testfile="/home/bhuang/corpus/speech/internal/hm_hm_16k/sb/test_hm_hm.csv"
# testfile="/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h.csv"
# testfile="/projects/corpus/voice/zaion/dekuple/2022_10_17/output/data/data_clean_without_words_with_dash.csv"
testfile="/projects/corpus/voice/zaion/dekuple/2022_10_28/output/segments/all.csv"

# outdir=${model_name_or_path}/results_hmhm
outdir=${model_name_or_path}/results_dekuple_lot3
# outdir=${model_name_or_path}/results_dekuple_chunk12_stride2
# outdir=${model_name_or_path}/results_dekuple_chunk30_stride5

#     --greedy \
#     --chunk_length_s 30.0 \
#     --stride_length_s 5.0 \
# --chunk_length_s 12.0 \
# --stride_length_s 2.0 \

python run_transcribe.py \
    --model_id $model_name_or_path \
    --test_csv_file $testfile \
    --audio_column_name "wav" \
    --start_column_name "start" \
    --end_column_name "end" \
    --preprocessing_num_workers 4 \
    --num_workers 0 \
    --batch_size 32 \
    --greedy \
    --outdir ${outdir}

# bh: --num_workers 1 got "Segmentation fault" error with DataLoader \
# python run_transcribe.py \
#     --model_id $model_name_or_path \
#     --test_csv_file $testfile \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 2 \
#     --log_outputs \
#     --outdir ${outdir}_with_lm

# python run_transcribe.py \
#     --model_id $model_name_or_path \
#     --test_csv_file $testfile \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 32 \
#     --chunk_length_s 30.0 \
#     --stride_length_s 5.0 \
#     --log_outputs \
#     --outdir ${outdir}_chunk30_stride5_with_lm

# python run_transcribe.py \
#     --model_id $model_name_or_path \
#     --test_csv_file $testfile \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 32 \
#     --chunk_length_s 12.0 \
#     --stride_length_s 2.0 \
#     --log_outputs \
#     --outdir ${outdir}_chunk12_stride2_with_lm
