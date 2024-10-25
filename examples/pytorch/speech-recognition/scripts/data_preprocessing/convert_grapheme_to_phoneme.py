#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Convert text into phonemes."""

import json

import fire
from datasets import load_dataset
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from tqdm import tqdm

from data_utils import write_dataset_to_json, print_dataset_info


def main(
    input_file_path: str,
    output_file_path: str,
    text_column_name: str = "text",
    phoneme_column_name: str = "phoneme",
    num_workers: int = 1,
):
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print_dataset_info(dataset)

    # debug
    # dataset = dataset.select(range(10))

    # initialize the espeak backend for French
    backend = EspeakBackend("fr-fr", language_switch="remove-flags")
    # separate phones by a space and ignoring words boundaries
    separator = Separator(phone=None, word=" ", syllable="")

    def phonemize_text(s):
        return backend.phonemize([s], separator=separator, strip=True, njobs=1)[0]

    dataset = dataset.map(
        lambda x: {phoneme_column_name: phonemize_text(x[text_column_name])},
        num_proc=num_workers,
    )
    print_dataset_info(dataset)

    # export
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
