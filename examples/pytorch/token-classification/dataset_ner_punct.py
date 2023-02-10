#! /usr/bin/env python3
# coding=utf-8
# Copyright 2022 Bofeng Huang

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from tqdm import tqdm


def align_sequence_label(word_ids, sequence_label, label_to_id, label_strategy="all"):
    previous_word_idx = None
    new_sequence_label = []
    for word_idx in word_ids:
        # Special tokens have a word id that is None. We set the label to -100 so they are automatically
        # ignored in the loss function.
        if word_idx is None:
            new_sequence_label.append(-100)
        else:
            if label_strategy == "first":
                if word_idx != previous_word_idx:
                    new_sequence_label.append(label_to_id[sequence_label[word_idx]])
                else:
                    new_sequence_label.append(-100)
            elif label_strategy == "last":
                if word_idx == previous_word_idx:
                    new_sequence_label[-1] = -100
                new_sequence_label.append(label_to_id[sequence_label[word_idx]])
            elif label_strategy == "all":
                new_sequence_label.append(label_to_id[sequence_label[word_idx]])
                # todo: differenci beginning label and others
                # if word_idx != previous_word_idx:
                #     new_sequence_label.append(label_to_id[sequence_label[word_idx]])
                # else:
                #     new_sequence_label.append(b_to_i_label[label_to_id[sequence_label[word_idx]]])
            else:
                raise ValueError("Invalid labeling strategy")

        previous_word_idx = word_idx

    return new_sequence_label


def tokenize_and_align_examples(
    examples, tokenizer, label_to_id, text_column_name="word", label_column_name="label", stride=100, label_strategy="all"
):

    tokenized_inputs = tokenizer(
        examples[text_column_name], is_split_into_words=True, truncation=True, return_overflowing_tokens=True, stride=stride
    )

    new_labels = []
    for example_id in range(len(tokenized_inputs["input_ids"])):
        sequence_id = tokenized_inputs["overflow_to_sample_mapping"][example_id]
        sequence_label = examples[label_column_name][sequence_id]

        word_ids = tokenized_inputs.word_ids(batch_index=example_id)

        new_labels.append(align_sequence_label(word_ids, sequence_label, label_to_id, label_strategy=label_strategy))

        # todo: align case label

        # debug
        # for t, w_i, l in zip(tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][example_id]), word_ids, new_labels[-1]):
        #     print(f"{t:10}  {w_i}  {l}")
        # quit()

    tokenized_inputs["labels"] = new_labels
    del tokenized_inputs["overflow_to_sample_mapping"]

    return tokenized_inputs


def load_data_files(data_dir, task_config, replacers=None, preprocessing_num_workers=None, overwrite_cache=False):
    task_configs = ["case", "eos", "punc"]

    if task_config not in task_configs:
        raise ValueError("Invalid task name")
    task_id = task_configs.index(task_config)

    def my_gen():
        paths = Path(data_dir).rglob("*.tsv")
        paths_list = list(paths)

        # data = []
        # word_lst, label_lst = [], []
        for p in tqdm(paths_list):
            df_ = pd.read_csv(
                p.as_posix(),
                sep="\t",
                header=None,
                usecols=[0, task_id + 1],
                names=["word", "label"],
                dtype={"word": str, "label": str},
            )

            # if replacers is not None:
            #     # df_["label"] = df_["label"].map(replacers).fillna(df_["label"])
            #     # Non-Exhaustive mappings where you intend to map specific values to NaN
            #     keep_nan = [k for k, v in replacers.items() if pd.isnull(v)]
            #     df_["label"] = df_["label"].map(replacers).fillna(df_["label"].mask(df_["label"].isin(keep_nan)))

            # print(df_["word"].tolist())
            # print(df_["label"].tolist())
            # quit()

            # data.append({"word": df_["word"].tolist(), "label": df_["label"].tolist()})
            # word_lst.append(df_["word"].tolist())
            # label_lst.append(df_["label"].tolist())

            # bh: without .astype(str) can cause errors
            yield {"word": df_["word"].astype(str).tolist(), "label": df_["label"].astype(str).tolist()}

    # ds = Dataset.from_list(data)
    # ds = Dataset.from_dict({"word": word_lst, "label": label_lst})
    ds = Dataset.from_generator(my_gen)

    # replacers = {"0": "250"}
    if replacers is not None:

        def proc_(example):
            # example["label"] = list(map(replacers.get, example["label"]))
            example["label"] = [replacers.get(l, l) for l in example["label"]]
            return example

        ds = ds.map(
            proc_,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc="Replacing labels in dataset",
        )

    # print(ds)
    # print(ds[0])
    # quit()

    return ds


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    data_dir = "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data"
    ds = load_data_files(data_dir, "punc")
    print(ds)
    print(ds[0])

    label_to_id = {
        "0": 0,
        "COMMA": 1,
        "PERIOD": 2,
        "QUESTION": 3,
        "EXCLAMATION": 4,
    }

    def tokenize_and_align_examples_(examples):
        return tokenize_and_align_examples(
            examples, label_to_id, text_column_name="word", label_column_name="label", stride=100, label_strategy="last"
        )

    processed_ds = ds.map(
        tokenize_and_align_examples_, batched=True, num_proc=1, load_from_cache_file=False, remove_columns=ds.column_names
    )
    print(processed_ds)
    print(processed_ds[0])

    for t, t_i, l in zip(
        tokenizer.convert_ids_to_tokens(processed_ds[0]["input_ids"]), processed_ds[0]["input_ids"], processed_ds[0]["labels"]
    ):
        # print(f"{t:10}  {t_i}  {l}")
        print(f"{t:10}  {l}")
