#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# convert whisper hf checkpoint to other formats

set -e

export HF_HOME="/projects/bhuang/.cache/huggingface"

transformers_whisper_root=/home/bhuang/transformers_new
openai_whisper_root=/home/bhuang/speech/whisper
whisper_cpp_root="/home/bhuang/speech/whisper.cpp"
mlx_eg_root="/home/bhuang/speech/mlx-examples"

# install ctranslate2
# pip install faster-whisper
# pip install ctranslate2

# model_dir="/projects/bhuang/models/asr/public/whisper-large-v3-french"
# model_dir="/projects/bhuang/models/asr/public/whisper-large-v3-french-distil-dec16"
# model_dir="/projects/bhuang/models/asr/public/whisper-large-v3-french-distil-dec8"
# model_dir="/projects/bhuang/models/asr/public/whisper-large-v3-french-distil-dec4"
# model_dir="/projects/bhuang/models/asr/public/whisper-large-v3-french-distil-dec2"
# model_dir="/projects/bhuang/models/asr/whisper/fr/whisper-large-v3-distil-fr-v0.2"
model_dir="/projects/bhuang/models/asr/whisper/it/whisper-large-v3-distil-it-v0.2"


# openai
###################################################################################################
echo "Converting to OpenAI..."
python $transformers_whisper_root/examples/pytorch/speech-recognition/scripts/convert_whisper_to_openai.py \
    --hf_model_name_or_path $model_dir \
    --whisper_state_path $model_dir/original_model.pt \
    --torch_dtype float16


# faster whisper
###################################################################################################
# also need to fix alignment_heads
# see https://huggingface.co/bofenghuang/whisper-large-v3-french-distil-dec2/discussions/1
echo "Converting to ct2..."
[[ -d $model_dir/ctranslate2 ]] || mkdir -p $model_dir/ctranslate2
ct2-transformers-converter \
    --model $model_dir \
    --output_dir $model_dir/ctranslate2 \
    --copy_files tokenizer.json preprocessor_config.json \
    --quantization float16 \
    --force

# whisper.cpp
###################################################################################################
cd $whisper_cpp_root

echo "Converting to ggml..."
python models/convert-pt-to-ggml.py \
    $model_dir/original_model.pt \
    $openai_whisper_root \
    $model_dir

echo "Quantizing ggml..."
./quantize \
    $model_dir/ggml-model.bin \
    $model_dir/ggml-model-q5_0.bin \
    q5_0

cd -


# mlx
###################################################################################################
# cd $mlx_eg_root

# echo "Converting to mlx..."
# python whisper/convert.py \
#     --torch-name-or-path $model_dir/original_model.pt \
#     --mlx-path $model_dir/mlx

# python whisper/convert.py \
#     --torch-name-or-path $model_dir/original_model.pt \
#     --mlx-path $model_dir/mlx_4_bit \
#     -q \
#     --q_bits 4

# cd -
