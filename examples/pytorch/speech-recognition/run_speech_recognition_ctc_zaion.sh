#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
# export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=hf-asr-hmhm
# export WANDB_NAME=wav2vec2-xls-r-1b-ft-medLR-ep30

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS=1

# export CUDA_VISIBLE_DEVICES=5,4,3,2
export CUDA_VISIBLE_DEVICES=5


# --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \

# 16k data
# --train_file="/home/bhuang/corpus/speech/internal/hm_hm_16k/sb/train_hm_hm.csv" \
# --validation_file="/home/bhuang/corpus/speech/internal/hm_hm_16k/sb/test_hm_hm.csv" \

# models
# --model_name_or_path="facebook/wav2vec2-large-960h" \
# --model_name_or_path="facebook/wav2vec2-conformer-rel-pos-large" \
# --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
# --model_name_or_path="facebook/wav2vec2-xls-r-300m" \
# --model_name_or_path="facebook/wav2vec2-xls-r-1b" \
# --model_name_or_path="facebook/wav2vec2-xls-r-2b" \
# --model_name_or_path="LeBenchmark/wav2vec2-FR-3K-large" \
# --model_name_or_path="LeBenchmark/wav2vec2-FR-7K-large" \
# --model_name_or_path="facebook/hubert-xlarge-ll60k" \
# --model_name_or_path="microsoft/unispeech-1350-en-353-fr-ft-1h" \
# --model_name_or_path="microsoft/wavlm-large" \

# specify trained tokenizer
# --model_name_or_path="jonatasgrosman/wav2vec2-xls-r-1b-french" \
# --tokenizer_name_or_path="jonatasgrosman/wav2vec2-xls-r-1b-french" \

# wer validation metric
# --metric_for_best_model "eval_wer" \
# --greater_is_better false \

# warmup
# --warmup_steps="500" \
# --warmup_ratio="0.1" \

# early stopping
# --early_stopping_patience="3" \

# LR
# --learning_rate="7.5e-5" \

# BS
# 32 for wav2vec2-FR-7K-large, 16 for wav2vec2-xls-r-1b
# global bs 128


# python -m torch.distributed.launch \
# 	--nproc_per_node 2 run_speech_recognition_ctc_b.py \
python run_speech_recognition_ctc_b.py \
	--train_file="/home/bhuang/corpus/speech/internal/hm_hm_merged/train_hmhm_merged_and_raw.csv" \
	--validation_file="/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h.csv" \
	--dataset_config_name="fr" \
	--text_column_name="wrd" \
	--audio_column_name="wav" \
	--length_column_name="input_length" \
	--group_by_length \
	--max_duration_in_seconds="30" \
	--min_duration_in_seconds="1" \
	--preprocessing_num_workers="8" \
	--dataloader_num_workers="4" \
	--model_name_or_path="LeBenchmark/wav2vec2-FR-7K-large" \
	--output_dir="./outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep50" \
	--overwrite_output_dir \
	--num_train_epochs="50" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--gradient_accumulation_steps="4" \
	--gradient_checkpointing \
	--learning_rate="1e-4" \
	--warmup_steps="800" \
	--weight_decay "0.01" \
	--logging_steps="100" \
	--evaluation_strategy="steps" \
	--eval_steps="500" \
	--save_steps="500" \
	--save_total_limit="5" \
	--metric_for_best_model="eval_wer" \
	--greater_is_better false \
	--load_best_model_at_end \
	--layerdrop="0.1" \
	--attention_dropout="0" \
	--activation_dropout="0.1" \
	--hidden_dropout="0" \
	--feat_proj_dropout="0" \
	--final_dropout="0" \
    --mask_feature_length="10" \
    --mask_feature_prob="0" \
    --mask_time_length="10" \
    --mask_time_prob="0.05" \
	--freeze_feature_encoder \
	--ctc_zero_infinity \
	--fp16 \
	--do_train --do_eval

	# --layerdrop="0.05" \
	# --attention_dropout="0.05" \
	# --activation_dropout="0.05" \
	# --hidden_dropout="0.05" \
	# --feat_proj_dropout="0.05" \
	# --final_dropout="0.05" \
    # --mask_feature_length="10" \
    # --mask_feature_prob="0" \
    # --mask_time_length="10" \
    # --mask_time_prob="0.05" \
