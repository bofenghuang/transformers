#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

export CUDA_VISIBLE_DEVICES=1

# bh: load open sourced models
# model_name_or_path="openai/whisper-small"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/general/$tmp_model_id/results"

# bh: load local models
# model_name_or_path="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep30_with_hmhm_lm"
# model_name_or_path="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep30_with_lm"
# model_name_or_path="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep30_with_carglass_lm"
# model_name_or_path="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep30_with_dekuple_lm"
# model_name_or_path="outputs/hmhm/wav2vec2-FR-7K-large-ft-medLR-ep30_with_lm"

model_name_or_path="outputs/hmhm_merged_and_raw/lebenchmark-wav2vec2_fr_7k_large-ft-ep15-bs128-lr1e4-wd1e2-aug-drp_with_lm"

outdir=${model_name_or_path}/results

# --greedy \
# --chunk_length_s 30.0 \
# --stride_length_s 5.0 \
# --chunk_length_s 12.0 \
# --stride_length_s 2.0 \

# bh: --num_workers 1 got "Segmentation fault" error with DataLoader \

python eval_wav2vec2_zaion.py \
    --model_id $model_name_or_path \
    --test_csv_file "/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h.csv" \
    --id_column_name "ID" \
    --audio_column_name "wav" \
    --text_column_name "wrd" \
    --start_column_name "start" \
    --end_column_name "end" \
    --length_column_name "duration" \
    --preprocessing_num_workers 4 \
    --num_workers 0 \
    --batch_size 4 \
    --log_outputs \
    --greedy \
    --outdir ${outdir}_hmhm

python eval_wav2vec2_zaion.py \
    --model_id $model_name_or_path \
    --test_csv_file "/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h.csv" \
    --id_column_name "ID" \
    --audio_column_name "wav" \
    --text_column_name "wrd" \
    --start_column_name "start" \
    --end_column_name "end" \
    --length_column_name "duration" \
    --preprocessing_num_workers 4 \
    --num_workers 0 \
    --batch_size 4 \
    --log_outputs \
    --outdir ${outdir}_hmhm_with_lm

# python eval_wav2vec2_zaion.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/carglass/data/20220111/data_wo_incomplet_words.csv" \
#     --id_column_name "utt" \
#     --audio_column_name "path" \
#     --text_column_name "text" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "dur" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 1 \
#     --log_outputs \
#     --greedy \
#     --outdir ${outdir}_carglass

# python eval_wav2vec2_zaion.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/carglass/data/20220111/data_wo_incomplet_words.csv" \
#     --id_column_name "utt" \
#     --audio_column_name "path" \
#     --text_column_name "text" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "dur" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 1 \
#     --log_outputs \
#     --outdir ${outdir}_carglass_with_lm

# python eval_wav2vec2_zaion.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/dekuple/2022_10_17/output/data/data_clean_without_words_with_dash.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --text_column_name "wrd" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "duration" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 4 \
#     --log_outputs \
#     --greedy \
#     --outdir ${outdir}_dekuple

# python eval_wav2vec2_zaion.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/dekuple/2022_10_17/output/data/data_clean_without_words_with_dash.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --text_column_name "wrd" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "duration" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 4 \
#     --log_outputs \
#     --outdir ${outdir}_dekuple_with_lm

# python eval_wav2vec2_zaion.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/lbpa/2023-02-21/data/data_without_partial_words.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --text_column_name "wrd" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "duration" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 4 \
#     --log_outputs \
#     --greedy \
#     --outdir ${outdir}_lbpa

# python eval_wav2vec2_zaion.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/lbpa/2023-02-21/data/data_without_partial_words.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --text_column_name "wrd" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "duration" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 4 \
#     --log_outputs \
#     --outdir ${outdir}_lbpa_with_lm

# python eval_wav2vec2_zaion.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/lbpa/2023-02-22/data/data_without_partial_words.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --text_column_name "wrd" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "duration" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 4 \
#     --log_outputs \
#     --greedy \
#     --outdir ${outdir}_lbpa_lot2

# python eval_wav2vec2_zaion.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/lbpa/2023-02-22/data/data_without_partial_words.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --text_column_name "wrd" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "duration" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 4 \
#     --log_outputs \
#     --outdir ${outdir}_lbpa_lot2_with_lm

# python eval_wav2vec2_zaion.py \
#     --model_id $model_name_or_path \
#     --test_csv_file "/projects/corpus/voice/zaion/lbpa/2023-02-22/data/data.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --text_column_name "wrd" \
#     --start_column_name "start" \
#     --end_column_name "end" \
#     --length_column_name "duration" \
#     --preprocessing_num_workers 4 \
#     --num_workers 0 \
#     --batch_size 4 \
#     --log_outputs \
#     --outdir ${outdir}_lbpa_lot2.1_with_lm
