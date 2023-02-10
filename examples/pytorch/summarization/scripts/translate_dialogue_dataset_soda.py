#!/usr/bin/env python
# Copyright 2022  Bofeng Huang

import os

import torch
from datasets import load_dataset
from nltk import tokenize

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


# pipe = pipeline("translation_en_to_fr", "t5-large", device=0)
# pipe = pipeline("translation", "t5-large", device=0)
# pipe = pipeline("translation", "Helsinki-NLP/opus-mt-en-fr", device=0)

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
model.eval()
model = model.cuda()

# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", torch_dtype=torch.float16)
# model.eval()
# model = model.cuda()


# or use simple regexes (.!?)
def split_sentences(s):
    return tokenize.sent_tokenize(s)


# def translate_sentence(s):
# gen_args = {
#     # "num_beams": 5,
#     # contrastive search
#     "top_k": 6,
#     "penalty_alpha": 0.6,
# }
#     return pipe(s, clean_up_tokenization_spaces=True, **gen_args)[0]["translation_text"]


def translate_sentence(s):
    inputs = tokenizer(s, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    gen_args = {
        "num_beams": 5,
        # "top_k": 6, "penalty_alpha": 0.6,
    }

    translated_tokens = model.generate(
        input_ids, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], **gen_args
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


def translate_sentences(s):
    sentences = split_sentences(s)
    return " ".join([translate_sentence(s_) for s_ in sentences])


def translate_dialogue(dialogue, speakers):
    try:
        new_dialogue_turns_list = []
        for sentence, speaker in zip(dialogue, speakers):
            translated_sentence = translate_sentences(sentence)
            new_dialogue_turns_list.append(": ".join([speaker, translated_sentence]))

        new_dialogue_turns = "\n".join(new_dialogue_turns_list)
    except Exception as e:
        print(dialogue)
        print(speakers)
        print(str(e))
        new_dialogue_turns = ""

    return new_dialogue_turns


def translate_example(example):
    example["dialogue_fr"] = translate_dialogue(example["dialogue"], example["speakers"])
    example["summary_fr"] = translate_sentences(example["narrative"])
    # example["topic_fr"] = translate_sentence(example["topic"])
    return example


# dataset = load_dataset("samsum")
# dataset = load_dataset("knkarthick/dialogsum")
# dataset = load_dataset("knkarthick/highlightsum")
dataset = load_dataset("allenai/soda")

dataset = dataset.remove_columns(set(dataset["train"].features.keys()) - set(["dialogue", "narrative", "speakers"]))

print(dataset)

# debug
# dataset["train"] = dataset["train"].select([4])
# print(dataset["train"])
# print(dataset["train"][0])
# quit()

# debug
# dataset["train"] = dataset["train"].select(range(10))
# dataset["validation"] = dataset["validation"].select(range(10))
# dataset["test"] = dataset["test"].select(range(10))

# processed_dataset = dataset.map(translate_example)

# processed_dataset["train"].to_csv("data/knkarthick_dialogsum_fr/train.tsv", index=False, sep="\t")
# processed_dataset["validation"].to_csv("data/knkarthick_dialogsum_fr/validation.tsv", index=False, sep="\t")
# processed_dataset["test"].to_csv("data/knkarthick_dialogsum_fr/test.tsv", index=False, sep="\t")
# processed_dataset.save_to_disk("data/knkarthick_dialogsum_fr/raw")


# output_dir = "data/knkarthick_dialogsum-fr-nllb_200_distilled_600M"
# output_dir = "data/samsum-fr-nllb_200_distilled_600M"
output_dir = "data/allenai_soda-fr-nllb_200_distilled_600M"
# output_dir = "data/tmp"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dataset["validation"].map(translate_example).to_csv(f"{output_dir}/validation.tsv", index=False, sep="\t")
dataset["train"].map(translate_example).to_csv(f"{output_dir}/train.tsv", index=False, sep="\t")
dataset["test"].map(translate_example).to_csv(f"{output_dir}/test.tsv", index=False, sep="\t")
