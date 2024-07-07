#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
export CUDA_VISIBLE_DEVICES="2"

input_file="/projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr000/train/train_espnet_yodas_manifest.json"
output_file="${input_file%.*}_lang.json"

python scripts/infer_audio_lid.py \
    --input_file_path $input_file \
    --output_file_path $output_file \
    --batch_size 64 \
    --dataloader_num_workers 16

    # --only_first_seconds 5.0 \