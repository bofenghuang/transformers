#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang


model_names_or_paths=(
# "openai/whisper-large-v2"
# "openai/whisper-large-v3"
# "bofenghuang/whisper-large-v3-french"
# "bofenghuang/whisper-large-v3-french-distil-dec16"
# "bofenghuang/whisper-large-v3-french-distil-dec2"
# "eustlb/distil-large-v3-fr"
"/projects/bhuang/models/asr/whisper/fr/whisper-large-v3-distil-fr"
)

for model_name_or_path in "${model_names_or_paths[@]}"; do
    ./infer_whisper_b2.sh $model_name_or_path
    # ./infer_whisper_a.sh $model_name_or_path
done

