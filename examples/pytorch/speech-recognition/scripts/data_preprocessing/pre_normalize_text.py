#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang



import fire
from datasets import load_dataset
from normalizers.french import FrenchTextNormalizer


from data_utils import write_dataset_to_json, print_dataset_info


def main(
    input_file_path: str,
    output_file_path: str,
    text_column_name: str = "text",
    normalized_text_column_name: str = "normalized_text",
    num_workers: int = 64,
):
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print_dataset_info(dataset)

    normalizer = FrenchTextNormalizer()

    def _normalize(s):
        s = normalizer(
            s,
            do_lowercase=False,
            do_ignore_words=False,
            symbols_to_keep="'-,.?!:;",
            do_num2text=False,
            do_text2num=False,
            do_remove_bracketed_words=True,
        )
        return s

    dataset = dataset.map(
        lambda x: {normalized_text_column_name: _normalize(x[text_column_name])},
        num_proc=num_workers,
        desc="normalizing text...",
    )

    dataset = dataset.filter(lambda x: x[normalized_text_column_name], num_proc=num_workers, desc="filtering...")
    print_dataset_info(dataset)

    # export
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
