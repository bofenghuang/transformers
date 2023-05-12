#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=hf-whisper-sprint-v2

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS=1

# https://github.com/microsoft/DeepSpeed/issues/662
# export CUDA_VISIBLE_DEVICES=5,4,3,2

# https://pytorch.org/docs/stable/elastic/run.html
# export HOST_NODE_ADDR="localhost:29001"

# --gradient_checkpointing \ vs use_cache
# --ddp_find_unused_parameters="True" \ and layerdrop, but can't do --gradient_checkpointing

# mcv
# --dataset_name="mozilla-foundation/common_voice_11_0" \
# --dataset_config_name="fr" \
# --train_split_name="train+validation" \
# --length_column_name="input_length" \

# --model_name_or_path="openai/whisper-large" \

# python run_speech_recognition_seq2seq_b.py \

# python -m torch.distributed.launch \
    # --rdzv_endpoint=$HOST_NODE_ADDR \
# torchrun \
#     --rdzv_endpoint=$HOST_NODE_ADDR \
#     --nproc_per_node 2 run_speech_recognition_seq2seq_b.py \

deepspeed --include localhost:0,1,2,3 --master_port 29001 run_speech_recognition_seq2seq_b.py \
    --dataset_name="CUSTOMIZED" \
    --deepspeed="ds_config.json" \
    --text_column_name="sentence" \
    --use_auth_token \
    --max_duration_in_seconds="30" \
    --preprocessing_num_workers="8" \
    --dataloader_num_workers="4" \
    --do_speech_augment \
    --language="french" \
    --task="transcribe" \
    --model_name_or_path="openai/whisper-large-v2" \
    --output_dir="./outputs/hf_whisper_sprint/whisper-large-v2-ft-french-lr4e6-bs256-augment" \
    --overwrite_output_dir \
    --num_train_epochs="2" \
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
    --save_total_limit="3" \
    --metric_for_best_model="wer" \
    --greater_is_better="False" \
    --load_best_model_at_end \
    --freeze_feature_encoder="False" \
    --fp16 \
    --use_cache="False" \
    --gradient_checkpointing \
    --predict_with_generate \
    --generation_max_length="40" \
    --generation_num_beams="1" \
    --do_train \
    --do_eval



# --encoder_layerdrop="0.1" \
# --decoder_layerdrop="0.1" \
# --dropout="0.1" \
# --attention_dropout="0.05" \
# --activation_dropout="0.05" \

# --encoder_layerdrop="0.05" \
# --decoder_layerdrop="0.05" \
# --dropout="0.1" \


# todo specaugment
# --mask_feature_length="10" \
# --mask_feature_prob="0" \
# --mask_time_length="10" \
# --mask_time_prob="0.05" \