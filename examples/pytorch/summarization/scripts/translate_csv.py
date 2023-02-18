#!/usr/bin/env python
# Copyright 2022  Bofeng Huang

import os

import torch
import pandas as pd
from datasets import load_dataset
from datasets import Dataset

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


# pipe = pipeline("translation_en_to_fr", "t5-large", device=0)
# pipe = pipeline("translation", "t5-large", device=0)
# pipe = pipeline("translation", "Helsinki-NLP/opus-mt-en-fr", device=0)

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype=torch.float16)
model.eval()
model = model.cuda()

# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", torch_dtype=torch.float16)
# model.eval()
# model = model.cuda()

# def translate_sentence(s):
# gen_args = {
#     # "num_beams": 5,
#     # contrastive search
#     "top_k": 6,
#     "penalty_alpha": 0.6,
# }
#     return pipe(s, clean_up_tokenization_spaces=True, **gen_args)[0]["translation_text"]


def translate_sentence(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    gen_args = {
        "num_beams": 5,
        # "top_k": 6, "penalty_alpha": 0.6,
    }

    translated_tokens = model.generate(input_ids, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], **gen_args)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


# for split in ["validation", "train", "test"]:
for split in ["train", "test"]:

    csv_path = f"data/allenai_soda-text/{split}.tsv"

    # ds = load_dataset("csv", csv_path, sep="\t")["train"]
    df = pd.read_csv(csv_path, sep="\t")
    # NB
    df["sentence"] = df["sentence"].fillna("")
    print(df.shape)
    print(df.head())
    ds = Dataset.from_pandas(df)
    print(ds)

    # debug
    # ds = ds.select(10)

    # debug
    # def temp_funct(example):
    #     example["len"] = len(example["sentence"])
    #     return example
    # ds = ds.map(temp_funct).sort("len", reverse=True)
    # print(ds[0])

    def translate_func(examples):
        try:
            examples["sentence"] = translate_sentence(examples["sentence"])
        except Exception as e:
            print(examples)
            print(str(e))
            # NB: later to be removed
            # examples["sentence"] = "_WRONG_EXAMPLE_"
        return examples

    output_dir = "data/allenai_soda-text-fr"
    # output_dir = "data/tmp"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ds.map(translate_func, batched=True, batch_size=64).to_csv(f"{output_dir}/{split}.tsv", index=False, sep="\t")
