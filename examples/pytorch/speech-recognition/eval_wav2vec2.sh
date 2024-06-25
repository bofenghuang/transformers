#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

export CUDA_VISIBLE_DEVICES=2

# model_name_or_path="outputs/common_voice_9_0_fr/wav2vec2-xls-r-1b-ft-ep10_with_lm"
# model_name_or_path="outputs/all/wav2vec2-xls-r-1b-ft_with_lm"
model_name_or_path="outputs/big/wav2vec2-FR-7K-large-ft_with_lm"


python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "speech-recognition-community-v2/dev_data" \
    --config "fr" \
    --split "validation" \
    --log_outputs \
    --chunk_length_s 30.0 \
    --stride_length_s 5.0 \
    --outdir "$model_name_or_path/results_speech-recognition-community-v2_dev_data_with_lm"

python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "speech-recognition-community-v2/dev_data" \
    --config "fr" \
    --split "validation" \
    --log_outputs \
    --greedy \
    --chunk_length_s 30.0 \
    --stride_length_s 5.0 \
    --outdir "$model_name_or_path/results_speech-recognition-community-v2_dev_data"


python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "mozilla-foundation/common_voice_11_0" \
    --config "fr" \
    --split "test" \
    --log_outputs \
    --outdir "$model_name_or_path/results_mozilla-foundatio_common_voice_11_0_with_lm"

python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "mozilla-foundation/common_voice_11_0" \
    --config "fr" \
    --split "test" \
    --log_outputs \
    --greedy \
    --outdir "$model_name_or_path/results_mozilla-foundatio_common_voice_11_0"

python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "facebook/voxpopuli" \
    --config "fr" \
    --split "test" \
    --text_column_name="normalized_text" \
    --log_outputs \
    --outdir "$model_name_or_path/results_facebook_voxpopuli_with_lm"

python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "facebook/voxpopuli" \
    --config "fr" \
    --split "test" \
    --text_column_name="normalized_text" \
    --log_outputs \
    --greedy \
    --outdir "$model_name_or_path/results_facebook_voxpopuli"

python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "facebook/multilingual_librispeech" \
    --config "french" \
    --split "test" \
    --text_column_name="text" \
    --log_outputs \
    --outdir "$model_name_or_path/results_multilingual_librispeech_with_lm"

python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "facebook/multilingual_librispeech" \
    --config "french" \
    --split "test" \
    --text_column_name="text" \
    --log_outputs \
    --greedy \
    --outdir "$model_name_or_path/results_multilingual_librispeech"

python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "gigant/african_accented_french" \
    --config "fr" \
    --split "test" \
    --log_outputs \
    --outdir "$model_name_or_path/results_african_accented_french_with_lm"

python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "gigant/african_accented_french" \
    --config "fr" \
    --split "test" \
    --log_outputs \
    --greedy \
    --outdir "$model_name_or_path/results_african_accented_french"

python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "google/fleurs" \
    --config "fr_fr" \
    --split "test" \
    --text_column_name "transcription" \
    --log_outputs \
    --outdir "$model_name_or_path/results_google_fleurs_with_lm"

python eval_wav2vec2.py \
    --model_id "$model_name_or_path" \
    --dataset "google/fleurs" \
    --config "fr_fr" \
    --split "test" \
    --text_column_name "transcription" \
    --log_outputs \
    --greedy \
    --outdir "$model_name_or_path/results_google_fleurs"
