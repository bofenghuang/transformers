#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

export CUDA_VISIBLE_DEVICES=5

# model_name_or_path="openai/whisper-small"
# model_name_or_path="jwkritchie/whisper-small-common-voice-fr"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/hf_whisper_sprint/$tmp_model_id/results"

# model_name_or_path="outputs/hf_whisper_sprint/whisper-medium-ft-lr6e6-bs256-dropout01-punct"
# model_name_or_path="outputs/hf_whisper_sprint/whisper-medium-ft-french-lr6e6-bs256-dropout005"
# model_name_or_path="outputs/hf_whisper_sprint/whisper-medium-ft-lr6e6-bs256-punct-augment"
# model_name_or_path="outputs/hf_whisper_sprint/whisper-small-ft-lr1e5-bs256-punct-augment"
model_name_or_path="outputs/hf_whisper_sprint/whisper-medium-ft-french-lr6e6-bs256-augment"

outdir=${model_name_or_path}/results

decode_suffix=_greedy
# decode_suffix=_beam5

# todo: lm, suppress_tokens

python eval_whisper.py \
    --model_id $model_name_or_path \
    --language "french" \
    --task "transcribe" \
    --dataset "mozilla-foundation/common_voice_11_0" \
    --config "fr" \
    --split "test" \
    --batch_size 32 \
    --fp16 \
    --log_outputs \
    --outdir ${outdir}_cv11${decode_suffix}

python eval_whisper.py \
    --model_id $model_name_or_path \
    --language "french" \
    --task "transcribe" \
    --dataset "facebook/multilingual_librispeech" \
    --config "french" \
    --split "test" \
    --text_column_name="text" \
    --batch_size 32 \
    --fp16 \
    --log_outputs \
    --outdir ${outdir}_mls${decode_suffix}

python eval_whisper.py \
    --model_id $model_name_or_path \
    --language "french" \
    --task "transcribe" \
    --dataset "facebook/voxpopuli" \
    --config "fr" \
    --split "test" \
    --text_column_name="normalized_text" \
    --batch_size 32 \
    --fp16 \
    --log_outputs \
    --outdir ${outdir}_voxpopuli${decode_suffix}

python eval_whisper.py \
    --model_id $model_name_or_path \
    --language "french" \
    --task "transcribe" \
    --dataset "google/fleurs" \
    --config "fr_fr" \
    --split "test" \
    --text_column_name="transcription" \
    --batch_size 32 \
    --fp16 \
    --log_outputs \
    --outdir ${outdir}_fleurs${decode_suffix}

python eval_whisper.py \
    --model_id $model_name_or_path \
    --language "french" \
    --task "transcribe" \
    --dataset "gigant/african_accented_french" \
    --config "fr" \
    --split "test" \
    --batch_size 32 \
    --fp16 \
    --log_outputs \
    --outdir ${outdir}_african_accented_french${decode_suffix}
