#!/usr/bin/env python
# Copyright 2022  Bofeng Huang

import os

import pandas as pd
from datasets import load_dataset
from nltk import tokenize
from tqdm import tqdm

# import nltk
# nltk.download('punkt')

# or use simple regexes (.!?)
def split_sentences(s):

    # norm
    s = " ".join(s.split())

    return tokenize.sent_tokenize(s)


dataset = load_dataset("allenai/soda")

dataset = dataset.remove_columns(set(dataset["train"].features.keys()) - set(["dialogue", "narrative", "speakers"]))

print(dataset)
print(dataset["train"][0])
# print(dataset["train"][70467])
# quit()

# for split in ["train", "validation", "test"]:
for split in ["train"]:
# for split in ["validation"]:

    data = []

    # debug
    # for example_idx, example in tqdm(enumerate(dataset[split].select([0]))):
    for example_idx, example in tqdm(enumerate(dataset[split])):
        for s in split_sentences(example["narrative"]):
            data.append(
                {
                    "idx": example_idx,
                    "type": "summary",
                    "spk_turn_idx": -1,
                    "speaker": "NA",
                    "sentence": s,
                }
            )

        for spk_turn_idx, (sentence, speaker) in enumerate(zip(example["dialogue"], example["speakers"])):
            for s in split_sentences(sentence):
                data.append(
                    {
                        "idx": example_idx,
                        "type": "dialogue",
                        "spk_turn_idx": spk_turn_idx,
                        "speaker": speaker,
                        "sentence": s,
                    }
                )

    df = pd.DataFrame(data)

    output_dir = "data/allenai_soda-text"
    # output_dir = "data/tmp"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(f"{output_dir}/{split}.tsv", index=False, sep="\t")
