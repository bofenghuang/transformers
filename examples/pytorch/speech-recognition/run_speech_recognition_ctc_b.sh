#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

# export WANDB_MODE=disabled
# export WANDB_PROJECT=hf-asr-cv-fr
export WANDB_PROJECT=hf-asr-fr

# export PYTHONPATH="$PYTHONPATH:/home/bhuang/my-scripts"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS=1

# https://github.com/microsoft/DeepSpeed/issues/662
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=1,2

# Corpus stat training_duration / test_duration
# mcv: 723.62h / 26.21h
# mtedx: 171.55h / 1.55h
# mediaspeech: 10.00h / 
# mls: 1086.65h / 10.07h
# voxpopuli: 210.66h / 4.89h
# african_accented_fr: 11.68h / 1.69h

# models
# --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
# --model_name_or_path="facebook/wav2vec2-xls-r-300m" \
# --model_name_or_path="facebook/wav2vec2-xls-r-1b" \
# --model_name_or_path="LeBenchmark/wav2vec2-FR-7K-large" \

# multiple gpu
# --ddp_find_unused_parameters true \
# --gradient_checkpointing \

# https://pytorch.org/docs/stable/elastic/run.html
# export HOST_NODE_ADDR="localhost:29001"

# python run_speech_recognition_ctc_b.py \

# python -m torch.distributed.launch \
    # --rdzv_endpoint=$HOST_NODE_ADDR \
# torchrun \
#     --rdzv_endpoint=$HOST_NODE_ADDR \
#     --nproc_per_node 2 run_speech_recognition_ctc_b.py \

# deepspeed --include localhost:1,2 --master_port 29001 run_speech_recognition_ctc_b.py \
#     --deepspeed="ds_config.json" \

# torchrun \
#     --nproc_per_node 4 run_speech_recognition_ctc_b.py \
python run_speech_recognition_ctc_b.py \
    --use_auth_token \
    --text_column_name="sentence" \
    --length_column_name="input_length" \
    --group_by_length \
    --max_duration_in_seconds="30" \
    --min_duration_in_seconds="1" \
    --do_speech_augment \
    --preprocessing_num_workers="16" \
    --dataloader_num_workers="4" \
    --model_name_or_path="LeBenchmark/wav2vec2-FR-7K-large" \
    --output_dir="./outputs/big/wav2vec2-FR-7K-large-ft-augment-bs256-lr1e4-tmp" \
    --overwrite_output_dir \
    --num_train_epochs="15" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --gradient_accumulation_steps="4" \
    --learning_rate="1e-4" \
    --warmup_steps="1000" \
    --weight_decay="0.01" \
    --logging_steps="25" \
    --evaluation_strategy="steps" \
    --eval_steps="1000" \
    --save_steps="1000" \
    --save_total_limit="3" \
    --metric_for_best_model="eval_wer" \
    --greater_is_better false \
    --load_best_model_at_end \
    --ctc_zero_infinity \
    --freeze_feature_encoder \
    --fp16 \
    --gradient_checkpointing \
    --layerdrop="0" \
    --feat_proj_dropout="0" \
    --attention_dropout="0" \
    --activation_dropout="0" \
    --hidden_dropout="0.05" \
    --final_dropout="0.05" \
    --mask_time_prob="0.05" \
    --mask_time_length="10" \
    --mask_feature_prob="0.05" \
    --mask_feature_length="10" \
    --do_train --do_eval
