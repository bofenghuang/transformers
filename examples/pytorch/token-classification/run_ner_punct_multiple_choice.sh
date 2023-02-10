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

# export WANDB_MODE=disabled
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=hf-punc
# export WANDB_NAME=wav2vec2-xls-r-1b-ft-medLR-ep30

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS=1

# export CUDA_VISIBLE_DEVICES=5,4,3,2
export CUDA_VISIBLE_DEVICES=3


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


# python3 run_ner_punc_multiple_choice.py \
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

python3 run_ner_punc_multiple_choice.py \
	--task_config '["case", "punc"]' \
	--train_data_dir "/projects/bhuang/corpus/text/flaubert/raw/multilingual_plus/data/train" \
	--validation_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/validation" \
	--test_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/test" \
	--text_column_name "word" \
	--label_column_name "label" \
	--preprocess_stride "100" \
	--preprocess_label_strategy '{"case": "first", "punc": "last"}' \
	--preprocess_label_replacers '{"case": {"OTHER": "LOWER"}, "punc": {"EXCLAMATION": "PERIOD", "COLON": "COMMA"}}' \
	--group_by_length \
	--preprocessing_num_workers "16" \
	--dataloader_num_workers "1" \
	--model_name_or_path "xlm-roberta-base" \
	--output_dir "outputs/recasepunc-multilingual_plus/xlm-roberta-base_ft" \
	--overwrite_output_dir \
	--num_train_epochs "5" \
	--per_device_train_batch_size "64" \
	--per_device_eval_batch_size "64" \
	--gradient_accumulation_steps "1" \
	--gradient_checkpointing \
	--learning_rate "5e-5" \
	--warmup_ratio "0.1" \
	--weight_decay "0.01" \
	--logging_steps "100" \
	--evaluation_strategy "steps" \
	--eval_steps "500" \
	--save_steps "500" \
	--save_total_limit "3" \
	--load_best_model_at_end \
	--fp16 \
	--do_train \
	--do_eval \
	--do_predict


python3 run_ner_punc_multiple_choice.py \
	--task_config '["case", "punc"]' \
	--train_data_dir "/projects/bhuang/corpus/text/flaubert/raw/multilingual_plus/data/train" \
	--validation_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/validation" \
	--test_data_dir "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/test" \
	--text_column_name "word" \
	--label_column_name "label" \
	--preprocess_stride "100" \
	--preprocess_label_strategy '{"case": "first", "punc": "last"}' \
	--preprocess_label_replacers '{"case": {"OTHER": "LOWER"}, "punc": {"EXCLAMATION": "PERIOD", "COLON": "COMMA"}}' \
	--group_by_length \
	--preprocessing_num_workers "16" \
	--dataloader_num_workers "1" \
	--model_name_or_path "xlm-roberta-large" \
	--output_dir "outputs/recasepunc-multilingual_plus/xlm-roberta-large_ft" \
	--overwrite_output_dir \
	--num_train_epochs "5" \
	--per_device_train_batch_size "64" \
	--per_device_eval_batch_size "64" \
	--gradient_accumulation_steps "1" \
	--gradient_checkpointing \
	--learning_rate "5e-5" \
	--warmup_ratio "0.1" \
	--weight_decay "0.01" \
	--logging_steps "100" \
	--evaluation_strategy "steps" \
	--eval_steps "500" \
	--save_steps "500" \
	--save_total_limit "3" \
	--load_best_model_at_end \
	--fp16 \
	--do_train \
	--do_eval \
	--do_predict


# python3 run_ner_punc_multiple_choice.py \
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
