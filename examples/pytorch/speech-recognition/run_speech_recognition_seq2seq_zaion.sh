#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=hf-asr-hmhm

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS=1

# https://github.com/microsoft/DeepSpeed/issues/662
export CUDA_VISIBLE_DEVICES=5

# trainfile="/home/bhuang/corpus/speech/internal/hm_hm/train_hmhm_190h.csv"
# trainfile="/home/bhuang/corpus/speech/internal/hm_hm_merged/train_hmhm_merged_and_raw.csv"
# validationfile="/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h.csv"
trainfile="/home/bhuang/transformers/examples/pytorch/speech-recognition/data/hmhm_max30s_max448tokens/train.csv"
validationfile="/home/bhuang/transformers/examples/pytorch/speech-recognition/data/hmhm_max30s_max448tokens/test.csv"

# --model_name_or_path="openai/whisper-large" \

# python run_speech_recognition_seq2seq_b.py \

# python -m torch.distributed.launch \
    # --rdzv_endpoint=$HOST_NODE_ADDR \
# torchrun \
#     --rdzv_endpoint=$HOST_NODE_ADDR \
#     --nproc_per_node 2 run_speech_recognition_seq2seq_b.py \

# deepspeed --include localhost:0,1,2,3 --master_port 29001 run_speech_recognition_seq2seq_b.py \
python run_speech_recognition_seq2seq_b.py \
    --train_file=$trainfile \
    --validation_file=$validationfile \
    --text_column_name="wrd" \
    --audio_column_name="wav" \
    --max_duration_in_seconds="30" \
    --preprocessing_num_workers="8" \
    --dataloader_num_workers="4" \
    --do_speech_augment \
    --language="french" \
    --task="transcribe" \
    --model_name_or_path="openai/whisper-large-v2" \
    --output_dir="./outputs/hmhm/whisper-large-v2-ft-bs256-lr4e6-augment" \
    --overwrite_output_dir \
	--preprocessing_only \
    --num_train_epochs="5" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="32" \
    --gradient_accumulation_steps="1" \
    --learning_rate="4.375e-6" \
    --warmup_steps="500" \
    --weight_decay "0.01" \
    --logging_steps="25" \
    --evaluation_strategy="steps" \
    --eval_steps="500" \
    --save_strategy="steps" \
    --save_steps="500" \
    --save_total_limit="5" \
    --metric_for_best_model="wer" \
    --greater_is_better="False" \
    --load_best_model_at_end \
    --freeze_feature_encoder="False" \
    --dropout="0.05" \
    --use_cache="False" \
    --fp16 \
    --gradient_checkpointing \
    --predict_with_generate \
    --generation_max_length="40" \
    --generation_num_beams="1" \
    --do_train \
    --do_eval
