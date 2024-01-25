#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import json

import fire
from datasets import load_dataset
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from tqdm import tqdm


def main(
    input_file_path: str,
    output_file_path: str,
    text_column_name: str = "text",
    phoneme_column_name: str = "phoneme",
    num_workers: int = 1,
):
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print(dataset)
    # print(max(dataset["duration"]))
    # quit()

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
    print(dataset)

    with open(output_file_path, "w") as fo:
        for sample in tqdm(dataset, desc="Writing to json", total=dataset.num_rows, unit=" samples"):
            fo.write(f"{json.dumps(sample, ensure_ascii=False)}\n")


if __name__ == "__main__":
    fire.Fire(main)
