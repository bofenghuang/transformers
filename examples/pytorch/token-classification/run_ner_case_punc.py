#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

from functools import partial
import logging
import os
import sys
from dataclasses import dataclass, field
import json
from typing import Optional

import datasets
import numpy as np
from datasets import ClassLabel, load_dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from utils.model_ner_case_punc import RobertaForCasePunc
from utils.collator_ner_case_punc import DataCollatorForCasePunc
# from utils.callbacks_ner_case_punc import WandbProgressResultsCallback

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    task_config: Optional[str] = field(default=None, metadata={"help": ""})
    train_data_dir: Optional[str] = field(default=None, metadata={"help": ""})
    validation_data_dir: Optional[str] = field(default=None, metadata={"help": ""})
    test_data_dir: Optional[str] = field(default=None, metadata={"help": ""})
    preprocess_label_replacers: Optional[str] = field(default=None, metadata={"help": ""})
    preprocess_stride: Optional[int] = field(default=None, metadata={"help": ""})
    preprocess_label_strategy: Optional[str] = field(default=None, metadata={"help": ""})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.train_data_dir is None and self.validation_data_dir is None and self.test_data_dir is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()
        if self.task_config is not None:
            self.task_config = json.loads(self.task_config)
        if self.preprocess_label_strategy is not None:
            self.preprocess_label_strategy = json.loads(self.preprocess_label_strategy)
        if self.preprocess_label_replacers is not None:
            self.preprocess_label_replacers = json.loads(self.preprocess_label_replacers)


def main():
# def main(args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # debug
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_ner", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters {data_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.train_data_dir is not None or data_args.validation_data_dir is not None or data_args.test_data_dir is not None:
        from utils.dataset_ner_case_punc import load_data_files, tokenize_and_align_examples
        load_data_files_ = partial(
            load_data_files,
            task_config=data_args.task_config,
            replacers=data_args.preprocess_label_replacers,
            preprocessing_num_workers=data_args.preprocessing_num_workers,
            overwrite_cache=data_args.overwrite_cache,
        )
        raw_datasets = DatasetDict()
        if data_args.train_data_dir is not None:
            raw_datasets["train"] = load_data_files_(data_args.train_data_dir)
        if data_args.validation_data_dir is not None:
            raw_datasets["validation"] = load_data_files_(data_args.validation_data_dir)
        if data_args.test_data_dir is not None:
            raw_datasets["test"] = load_data_files_(data_args.test_data_dir)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
        features = raw_datasets["test"].features
    else:
        raise ValueError("wtf do you want?")

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    # labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    # if labels_are_int:
    #     label_list = features[label_column_name].feature.names
    #     label_to_id = {i: i for i in range(len(label_list))}
    # else:
    #     label_list = get_label_list(raw_datasets["train"][label_column_name])
    #     label_to_id = {l: i for i, l in enumerate(label_list)}

    # num_labels = len(label_list)

    label_names = [column_name_[:-6] for column_name_ in column_names if column_name_[-6:] == "_label"]
    label_lists_by_choice = {}
    label_to_id_by_choice = {}
    for label_name_ in label_names:
        label_lists_by_choice[label_name_] = get_label_list(raw_datasets["train"][f"{label_name_}_label"])
        label_to_id_by_choice[label_name_] = {l: i for i, l in enumerate(label_lists_by_choice[label_name_])}
    # not used
    num_labels = 2
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"bloom", "gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # this model has a wrong maxlength value, so we need to set it manually
        if tokenizer_name_or_path == "camembert/camembert-large":
            tokenizer.model_max_length = 512

    model = RobertaForCasePunc.from_pretrained(
        model_args.model_name_or_path,
        num_case_labels=len(label_lists_by_choice["case"]),
        num_punc_labels=len(label_lists_by_choice["punc"]),
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Model has labels -> use them.
    # if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
    #     if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
    #         # Reorganize `label_list` to match the ordering of the model.
    #         if labels_are_int:
    #             label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
    #             label_list = [model.config.id2label[i] for i in range(num_labels)]
    #         else:
    #             label_list = [model.config.id2label[i] for i in range(num_labels)]
    #             label_to_id = {l: i for i, l in enumerate(label_list)}
    #     else:
    #         logger.warning(
    #             "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #             f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels:"
    #             f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
    #         )

    # Set the correspondences label/ID inside the model config
    # model.config.label2id = {l: i for i, l in enumerate(label_list)}
    # model.config.id2label = {i: l for i, l in enumerate(label_list)}
    model.config.case_label2id = {l: i for i, l in enumerate(label_lists_by_choice["case"])}
    model.config.case_id2label = {i: l for i, l in enumerate(label_lists_by_choice["case"])}
    model.config.punc_label2id = {l: i for i, l in enumerate(label_lists_by_choice["punc"])}
    model.config.punc_id2label = {i: l for i, l in enumerate(label_lists_by_choice["punc"])}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    # b_to_i_label = []
    # for idx, label in enumerate(label_list):
    #     if label.startswith("B-") and label.replace("B-", "I-") in label_list:
    #         b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
    #     else:
    #         b_to_i_label.append(idx)

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    # def tokenize_and_align_labels(examples):
    #     tokenized_inputs = tokenizer(
    #         examples[text_column_name],
    #         padding=padding,
    #         truncation=True,
    #         max_length=data_args.max_seq_length,
    #         # We use this argument because the texts in our dataset are lists of words (with a label for each word).
    #         is_split_into_words=True,
    #     )
    #     labels = []
    #     for i, label in enumerate(examples[label_column_name]):
    #         word_ids = tokenized_inputs.word_ids(batch_index=i)
    #         previous_word_idx = None
    #         label_ids = []
    #         for word_idx in word_ids:
    #             # Special tokens have a word id that is None. We set the label to -100 so they are automatically
    #             # ignored in the loss function.
    #             if word_idx is None:
    #                 label_ids.append(-100)
    #             # We set the label for the first token of each word.
    #             elif word_idx != previous_word_idx:
    #                 label_ids.append(label_to_id[label[word_idx]])
    #             # For the other tokens in a word, we set the label to either the current label or -100, depending on
    #             # the label_all_tokens flag.
    #             else:
    #                 if data_args.label_all_tokens:
    #                     label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
    #                 else:
    #                     label_ids.append(-100)
    #             previous_word_idx = word_idx

    #         labels.append(label_ids)
    #     tokenized_inputs["labels"] = labels
    #     return tokenized_inputs

    tokenize_and_align_examples_ = partial(
        tokenize_and_align_examples,
        tokenizer=tokenizer,
        label_names=label_names,
        label_to_id_mappings=label_to_id_by_choice,
        label_strategies=data_args.preprocess_label_strategy,
        text_column_name=text_column_name,
        stride=data_args.preprocess_stride,
    )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            # train_dataset = train_dataset.map(
            #     tokenize_and_align_labels,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     desc="Running tokenizer on train dataset",
            # )

            train_dataset = train_dataset.map(
                tokenize_and_align_examples_,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=train_dataset.column_names,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            # eval_dataset = eval_dataset.map(
            #     tokenize_and_align_labels,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     desc="Running tokenizer on validation dataset",
            # )

            eval_dataset = eval_dataset.map(
                tokenize_and_align_examples_,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=eval_dataset.column_names,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            # predict_dataset = predict_dataset.map(
            #     tokenize_and_align_labels,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     desc="Running tokenizer on prediction dataset",
            # )

            predict_dataset = predict_dataset.map(
                tokenize_and_align_examples_,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=predict_dataset.column_names,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    # data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    data_collator = DataCollatorForCasePunc(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metrics
    # metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        # predictions = np.argmax(predictions, axis=2)
        predictions = [np.argmax(predictions_, axis=2) for predictions_ in predictions]

        output = {}
        for idx, (label_name, label_list) in enumerate(label_lists_by_choice.items()):
            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions[idx], labels[idx])
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions[idx], labels[idx])
            ]

            # results = metric.compute(predictions=true_predictions, references=true_labels)
            # if data_args.return_entity_level_metrics:
            #     # Unpack nested dictionaries
            #     final_results = {}
            #     for key, value in results.items():
            #         if isinstance(value, dict):
            #             for n, v in value.items():
            #                 final_results[f"{key}_{n}"] = v
            #         else:
            #             final_results[key] = value
            #     return final_results
            # else:
            #     return {
            #         "precision": results["overall_precision"],
            #         "recall": results["overall_recall"],
            #         "f1": results["overall_f1"],
            #         "accuracy": results["overall_accuracy"],
            #     }

            true_labels = [true_l_ for true_labels_ in true_labels for true_l_ in true_labels_]
            true_predictions = [true_p_ for true_preds in true_predictions for true_p_ in true_preds]
            precision_, recall_, fscore_, _ = precision_recall_fscore_support(true_labels, true_predictions, average='weighted')
            accuracy_ = accuracy_score(true_labels, true_predictions)

            output[f"{label_name}_precision"] = precision_
            output[f"{label_name}_recall"] = recall_
            output[f"{label_name}_f1"] = fscore_
            output[f"{label_name}_accuracy"] = accuracy_

        return output

    # Logging prediction wandb callback
    # sample_dataset = load_dataset("text", data_files="/home/bhuang/transformers/examples/pytorch/token-classification/tmp_test.txt")["train"]
    # sample_dataset = sample_dataset.map(
    #     lambda x: tokenizer(
    #         x["text"],
    #         padding=True,
    #         truncation=True,
    #         return_offsets_mapping=True,
    #         return_tensors="pt",
    #     ),
    #     batched=True
    # )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # create the callback
    # progress_callback = WandbProgressResultsCallback(trainer, sample_dataset)
    # add the callback to the trainer
    # trainer.add_callback(progress_callback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        # predictions = np.argmax(predictions, axis=2)
        predictions = [np.argmax(predictions_, axis=2) for predictions_ in predictions]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        for idx, (label_name, label_list) in enumerate(label_lists_by_choice.items()):
            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions[idx], labels[idx])
            ]

            # Save predictions
            output_predictions_file = os.path.join(training_args.output_dir, f"{label_name}_predictions.txt")
            if trainer.is_world_process_zero():
                with open(output_predictions_file, "w") as writer:
                    for prediction in true_predictions:
                        writer.write(" ".join(prediction) + "\n")

            # Remove ignored index (special tokens)
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions[idx], labels[idx])
            ]
            # Save labels
            output_labels_file = os.path.join(training_args.output_dir, f"{label_name}_references.txt")
            if trainer.is_world_process_zero():
                with open(output_labels_file, "w") as writer:
                    for l in true_labels:
                        writer.write(" ".join(l) + "\n")

        # todo: add final metrics

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
