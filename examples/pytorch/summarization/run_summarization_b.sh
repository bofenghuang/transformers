#!/usr/bin/env bash

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=hf-dialogsum-fr-summ

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=4

# path_to_csv_or_jsonlines_train_file=/home/bhuang/transformers/examples/pytorch/summarization/data/knkarthick_dialogsum_fr_nbbl/train.tsv
# path_to_csv_or_jsonlines_validation_file=/home/bhuang/transformers/examples/pytorch/summarization/data/knkarthick_dialogsum_fr_nbbl/validation.tsv
# path_to_csv_or_jsonlines_test_file=/home/bhuang/transformers/examples/pytorch/summarization/data/knkarthick_dialogsum_fr_nbbl/test.tsv

# path_to_csv_or_jsonlines_train_file=/home/bhuang/transformers/examples/pytorch/summarization/data/mixed-knkarthick_dialogsum-samsum-fr-nllb_200_distilled_600M/train.tsv
# path_to_csv_or_jsonlines_validation_file=/home/bhuang/transformers/examples/pytorch/summarization/data/mixed-knkarthick_dialogsum-samsum-fr-nllb_200_distilled_600M/validation.tsv
# path_to_csv_or_jsonlines_test_file=/home/bhuang/transformers/examples/pytorch/summarization/data/mixed-knkarthick_dialogsum-samsum-fr-nllb_200_distilled_600M/test.tsv

path_to_csv_or_jsonlines_train_file=/home/bhuang/transformers/examples/pytorch/summarization/data/knkarthick_dialogsum-fr-nllb_200_distilled_600M//train.tsv
path_to_csv_or_jsonlines_validation_file=/home/bhuang/transformers/examples/pytorch/summarization/data/knkarthick_dialogsum-fr-nllb_200_distilled_600M//validation.tsv
path_to_csv_or_jsonlines_test_file=/home/bhuang/transformers/examples/pytorch/summarization/data/knkarthick_dialogsum-fr-nllb_200_distilled_600M//test.tsv

outdir=outputs/knkarthick_dialogsum-fr-nllb_200_distilled_600M

# python run_summarization_b.py \
#     --train_file $path_to_csv_or_jsonlines_train_file \
#     --validation_file $path_to_csv_or_jsonlines_validation_file \
#     --test_file $path_to_csv_or_jsonlines_test_file \
#     --text_column "dialogue_fr" \
#     --summary_column "summary_fr" \
#     --preprocessing_num_workers="8" \
#     --dataloader_num_workers="4" \
#     --model_name_or_path "moussaKam/barthez" \
#     --output_dir "outputs/knkarthick_dialogsum_fr_nbbl/moussakam_barthez" \
#     --overwrite_output_dir \
#     --num_train_epochs="10" \
#     --per_device_train_batch_size="32" \
#     --per_device_eval_batch_size="16" \
# 	--gradient_accumulation_steps="2" \
#     --learning_rate="1e-5" \
#     --warmup_ratio="0.05" \
#     --weight_decay "0.01" \
#     --logging_steps="25" \
#     --evaluation_strategy="steps" \
#     --eval_steps="100" \
#     --save_strategy="steps" \
#     --save_steps="100" \
#     --save_total_limit="3" \
#     --predict_with_generate \
#     --load_best_model_at_end \
#     --do_train \
#     --do_eval \
#     --do_predict

python run_summarization_b.py \
    --train_file $path_to_csv_or_jsonlines_train_file \
    --validation_file $path_to_csv_or_jsonlines_validation_file \
    --test_file $path_to_csv_or_jsonlines_test_file \
    --text_column "dialogue_fr" \
    --summary_column "summary_fr" \
    --preprocessing_num_workers="8" \
    --dataloader_num_workers="4" \
    --model_name_or_path "moussaKam/mbarthez" \
    --output_dir "${outdir}/moussakam_mbarthez-ft" \
    --overwrite_output_dir \
    --num_train_epochs="10" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="4" \
	--gradient_accumulation_steps="16" \
    --learning_rate="2e-5" \
    --warmup_ratio="0.05" \
    --weight_decay "0.01" \
    --logging_steps="25" \
    --evaluation_strategy="steps" \
    --eval_steps="100" \
    --save_strategy="steps" \
    --save_steps="100" \
    --save_total_limit="3" \
    --load_best_model_at_end \
    --predict_with_generate \
    --do_train \
    --do_eval \
    --do_predict


# python run_summarization_b.py \
#     --train_file $path_to_csv_or_jsonlines_train_file \
#     --validation_file $path_to_csv_or_jsonlines_validation_file \
#     --test_file $path_to_csv_or_jsonlines_test_file \
#     --text_column "dialogue_fr" \
#     --summary_column "summary_fr" \
#     --preprocessing_num_workers="8" \
#     --dataloader_num_workers="4" \
#     --model_name_or_path "t5-small" \
#     --source_prefix "summarize: " \
#     --output_dir "outputs/knkarthick_dialogsum_fr_nbbl/t5_small" \
#     --overwrite_output_dir \
#     --num_train_epochs="20" \
#     --per_device_train_batch_size="8" \
#     --per_device_eval_batch_size="4" \
# 	--gradient_accumulation_steps="8" \
#     --learning_rate="5e-5" \
#     --warmup_ratio="0.05" \
#     --weight_decay "0.01" \
#     --logging_steps="25" \
#     --evaluation_strategy="steps" \
#     --eval_steps="100" \
#     --save_strategy="steps" \
#     --save_steps="100" \
#     --save_total_limit="3" \
#     --predict_with_generate \
#     --load_best_model_at_end \
#     --do_train \
#     --do_eval \
#     --do_predict

# python run_summarization_b.py \
#     --train_file $path_to_csv_or_jsonlines_train_file \
#     --validation_file $path_to_csv_or_jsonlines_validation_file \
#     --test_file $path_to_csv_or_jsonlines_test_file \
#     --text_column "dialogue_fr" \
#     --summary_column "summary_fr" \
#     --preprocessing_num_workers="8" \
#     --dataloader_num_workers="4" \
#     --model_name_or_path "t5-base" \
#     --source_prefix "summarize: " \
#     --output_dir "outputs/knkarthick_dialogsum_fr_nbbl/t5_base" \
#     --overwrite_output_dir \
#     --num_train_epochs="20" \
#     --per_device_train_batch_size="8" \
#     --per_device_eval_batch_size="4" \
# 	--gradient_accumulation_steps="8" \
#     --learning_rate="5e-5" \
#     --warmup_ratio="0.05" \
#     --weight_decay "0.01" \
#     --logging_steps="25" \
#     --evaluation_strategy="steps" \
#     --eval_steps="100" \
#     --save_strategy="steps" \
#     --save_steps="100" \
#     --save_total_limit="3" \
#     --predict_with_generate \
#     --load_best_model_at_end \
#     --do_train \
#     --do_eval \
#     --do_predict

python run_summarization_b.py \
    --train_file $path_to_csv_or_jsonlines_train_file \
    --validation_file $path_to_csv_or_jsonlines_validation_file \
    --test_file $path_to_csv_or_jsonlines_test_file \
    --text_column "dialogue_fr" \
    --summary_column "summary_fr" \
    --preprocessing_num_workers="8" \
    --dataloader_num_workers="4" \
    --model_name_or_path "t5-large" \
    --source_prefix "summarize: " \
    --output_dir "${outdir}/t5_large-ft" \
    --overwrite_output_dir \
    --num_train_epochs="10" \
    --per_device_train_batch_size="4" \
    --per_device_eval_batch_size="2" \
	--gradient_accumulation_steps="32" \
    --learning_rate="5e-5" \
    --warmup_ratio="0.05" \
    --weight_decay "0.01" \
    --logging_steps="25" \
    --evaluation_strategy="steps" \
    --eval_steps="100" \
    --save_strategy="steps" \
    --save_steps="100" \
    --save_total_limit="3" \
    --load_best_model_at_end \
    --predict_with_generate \
    --do_train \
    --do_eval \
    --do_predict



# --use_cache="False" \
# --gradient_checkpointing \