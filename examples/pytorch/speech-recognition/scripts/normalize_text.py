#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

# HF_HOME="/projects/bhuang/.cache/huggingface" PYTHONPATH="${PYTHONPATH:-}:/home/bhuang/myscripts" python scripts/normalize_text.py /projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/train_asr_processed_cleaned.json /projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/train_asr_processed_cleaned_normalized.json

import json
import re
from collections import Counter
import fire
from datasets import load_dataset
from tqdm import tqdm

from text_normalization.normalize_french import FrenchTextNormalizer


def main(
    input_file_path: str,
    output_file_path: str,
    text_column_name: str = "text",
    normalized_text_column_name: str = "normalized_text",
    num_workers: int = 16,
):
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print(dataset[0])
    print(dataset.num_rows)

    # normalizer = FrenchTextNormalizer()

    # def normalize(s):
    #     s = normalizer(s, do_lowercase=True, do_ignore_words=False, symbols_to_keep="'", do_num2text=True)  # w/o "-"
    #     return s

    # dataset = dataset.map(lambda x: {normalized_text_column_name: normalize(x[text_column_name])}, num_proc=num_workers)

    # char_counts = Counter(" ".join(dataset[text_column_name]))
    # print(char_counts)
    # quit()

    # allowed_chars = "a-zàâäçéèêëîïñôöùûüÿ" + "'- "
    allowed_chars = "a-zàâäçéèêëîïôöùûü" + "'\- "

    # dataset = dataset.filter(lambda x: not bool(re.search(rf"[^{allowed_chars}]", x)), input_columns=normalized_text_column_name, num_proc=num_workers)
    dataset = dataset.filter(lambda x: not bool(re.search(rf"[^{allowed_chars}]", x)), input_columns=text_column_name, num_proc=num_workers)
    print(dataset.num_rows)

    with open(output_file_path, "w") as manifest_f:
        for sample in tqdm(dataset, desc="Saving", total=dataset.num_rows, unit=" samples"):
            # only save serializable types
            sample = {k: v for k, v in sample.items() if isinstance(v, (str, int, float))}
            manifest_f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")


if __name__ == "__main__":
    fire.Fire(main)
