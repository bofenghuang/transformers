# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export TRANSFORMERS_CACHE="/projects/bhuang/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT="case_punctuation"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS="1"

export TOKENIZERS_PARALLELISM="false"

# export CUDA_VISIBLE_DEVICES="0,1,2,5"
export CUDA_VISIBLE_DEVICES=""


	# --model_name_or_path "camembert-base" \
	# --output_dir "outputs/punc/camembert-base_ft" \

	# --model_name_or_path "xlm-roberta-base" \
	# --output_dir "outputs/punc/xlm-roberta-base_ft" \

	# --model_name_or_path "camembert/camembert-large" \
	# --output_dir "outputs/punc/camembert-large_ft" \

	# 	--model_name_or_path "xlm-roberta-large" \
	# 	--output_dir "outputs/punc/xlm-roberta-large_ft" \

# --overwrite_cache \
# --run_name=$WANDB_NAME \


# python3 run_ner_case_punc.py \
# 	--task_config '["case", "punc"]' \
# 	--train_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/train" \
# 	--validation_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/validation" \
# 	--test_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/test" \
# 	--text_column_name "word" \
# 	--label_column_name "label" \
# 	--preprocess_stride "100" \
# 	--preprocess_label_strategy '{"case": "first", "punc": "last"}' \
# 	--preprocess_label_replacers '{"case": {"OTHER": "LOWER"}, "punc": {"EXCLAMATION": "PERIOD"}}' \
# 	--group_by_length \
# 	--preprocessing_num_workers "16" \
# 	--dataloader_num_workers "1" \
# 	--model_name_or_path "camembert-base" \
# 	--output_dir "outputs/recasepunc/camembert-base_ft" \
# 	--overwrite_output_dir \
# 	--num_train_epochs "5" \
# 	--per_device_train_batch_size "64" \
# 	--per_device_eval_batch_size "64" \
# 	--gradient_accumulation_steps "1" \
# 	--gradient_checkpointing \
# 	--learning_rate "5e-5" \
# 	--warmup_ratio "0.1" \
# 	--weight_decay "0.01" \
# 	--logging_steps "100" \
# 	--evaluation_strategy "steps" \
# 	--eval_steps "500" \
# 	--save_steps "500" \
# 	--save_total_limit "3" \
# 	--load_best_model_at_end \
# 	--fp16 \
# 	--do_train \
# 	--do_eval

# todo
# train strategy "all"

	# --preprocess_label_strategy '{"case": "first", "punc": "last"}' \
	# --preprocess_label_replacers '{"case": {"OTHER": "LOWER"}, "punc": {"EXCLAMATION": "PERIOD", "COLON": "COMMA"}}' \
	# --group_by_length \
	# --fp16 \

python3 run_ner_case_punc.py \
	--task_config '["case", "punc"]' \
	--train_data_dir "/projects/bhuang/corpus/text/case_punctuation/final/train/fr_tedtalks" \
	--validation_data_dir "/projects/bhuang/corpus/text/case_punctuation/final/validation/fr_tedtalks" \
	--test_data_dir "/projects/bhuang/corpus/text/case_punctuation/final/test/fr_tedtalks" \
	--text_column_name "word" \
	--label_column_name "label" \
	--preprocess_stride "100" \
	--preprocess_label_strategy '{"case": "first", "punc": "last"}' \
	--preprocess_label_replacers '{"case": {"OTHER": "CAPITALIZE"}}' \
	--preprocessing_num_workers "16" \
	--dataloader_num_workers "1" \
	--model_name_or_path "xlm-roberta-base" \
	--output_dir "outputs_2023/casepunc_fr/xlm_roberta_base_ft" \
	--run_name "xlm_roberta_base_ft" \
	--overwrite_output_dir \
	--per_device_train_batch_size "1" \
	--per_device_eval_batch_size "1" \
	--gradient_accumulation_steps "1" \
	--num_train_epochs "3" \
	--learning_rate "5e-5" \
	--warmup_ratio "0.1" \
	--lr_scheduler_type "cosine" \
	--weight_decay "0.01" \
	--gradient_checkpointing \
	--logging_steps "10" \
	--logging_first_step \
	--save_strategy "steps" \
	--save_steps "500" \
	--save_total_limit "3" \
	--evaluation_strategy "steps" \
	--eval_steps "500" \
	--load_best_model_at_end \
	--report_to "tensorboard" "wandb" \
	--do_train \
	--do_eval \
	--do_predict


# python3 run_ner_case_punc.py \
# 	--task_config '["case", "punc"]' \
# 	--train_data_dir "/projects/bhuang/corpus/text/flaubert/raw/multilingual_plus/data/train" \
# 	--validation_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/validation" \
# 	--test_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/test" \
# 	--text_column_name "word" \
# 	--label_column_name "label" \
# 	--preprocess_stride "100" \
# 	--preprocess_label_strategy '{"case": "first", "punc": "last"}' \
# 	--preprocess_label_replacers '{"case": {"OTHER": "LOWER"}, "punc": {"EXCLAMATION": "PERIOD", "COLON": "COMMA"}}' \
# 	--group_by_length \
# 	--preprocessing_num_workers "16" \
# 	--dataloader_num_workers "1" \
# 	--model_name_or_path "xlm-roberta-large" \
# 	--output_dir "outputs/recasepunc-multilingual_plus/xlm-roberta-large_ft" \
# 	--overwrite_output_dir \
# 	--num_train_epochs "5" \
# 	--per_device_train_batch_size "64" \
# 	--per_device_eval_batch_size "64" \
# 	--gradient_accumulation_steps "1" \
# 	--gradient_checkpointing \
# 	--learning_rate "5e-5" \
# 	--warmup_ratio "0.1" \
# 	--weight_decay "0.01" \
# 	--logging_steps "100" \
# 	--evaluation_strategy "steps" \
# 	--eval_steps "500" \
# 	--save_steps "500" \
# 	--save_total_limit "3" \
# 	--load_best_model_at_end \
# 	--fp16 \
# 	--do_train \
# 	--do_eval \
# 	--do_predict


# python3 run_ner_case_punc.py \
# 	--task_config '["case", "punc"]' \
# 	--train_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/train" \
# 	--test_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/tmp" \
# 	--text_column_name "word" \
# 	--label_column_name "label" \
# 	--preprocess_stride "100" \
# 	--preprocess_label_strategy '{"case": "first", "punc": "last"}' \
# 	--preprocess_label_replacers '{"case": {"OTHER": "LOWER"}, "punc": {"EXCLAMATION": "PERIOD"}}' \
# 	--group_by_length \
# 	--preprocessing_num_workers "16" \
# 	--model_name_or_path "outputs/recasepunc/camembert-base_ft" \
# 	--output_dir "outputs/recasepunc/tmp_n" \
# 	--overwrite_output_dir \
# 	--per_device_eval_batch_size "64" \
# 	--fp16 \
# 	--do_predict
