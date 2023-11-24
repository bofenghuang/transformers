#!/usr/bin/env bash

export TRANSFORMERS_CACHE="/projects/bhuang/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED=1

export WANDB_PROJECT=hf-asr-pretrain

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS=1

# https://github.com/microsoft/DeepSpeed/issues/662
# export CUDA_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES="0,1,2,5"


    # --overwrite_cache \
    # --preprocessing_only \

# accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16

# accelerate launch --num_processes=1 --mixed_precision=fp16 run_wav2vec2_pretraining_no_trainer_b.py \
accelerate launch --num_machines=1 --multi_gpu --num_processes=4 --mixed_precision=fp16 run_wav2vec2_pretraining_no_trainer_b.py \
    --model_name_or_path "LeBenchmark/wav2vec2-FR-7K-large" \
	--output_dir "outputs/wav2vec2-FR-7K-large-ft" \
    --validation_split_percentage "1" \
    --audio_column_name "wav" \
    --max_duration_in_seconds "30" \
    --min_duration_in_seconds "1" \
    --preprocessing_num_workers "16" \
    --dataloader_num_workers "4" \
    --pad_to_multiple_of "8" \
    --num_train_epochs "1" \
	--per_device_train_batch_size "8" \
	--per_device_eval_batch_size "8" \
	--gradient_accumulation_steps "32" \
	--learning_rate "1e-4" \
	--adam_beta1 "0.9" \
	--adam_beta2 "0.98" \
	--adam_epsilon "1e-06" \
	--num_warmup_steps "2000" \
	--weight_decay "0.01" \
	--mask_time_prob "0.65" \
	--mask_time_length "10" \
	--gradient_checkpointing \
	--logging_steps "1" \
    --saving_steps "500" \
