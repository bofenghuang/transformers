#!/usr/bin/env python
# coding=utf-8
# Copyright 2021  Bofeng Huang

import argparse
import os
import re

from datasets import load_dataset

kenlm_root = os.getenv("KENLM_ROOT", "/home/bhuang/kenlm")
kenlm_bin = os.path.join(kenlm_root, "build/bin")

pattern_not_allowed_chars = re.compile(r"[^a-zàâäçéèêëîïôöûùüÿñœ\'\- ]")
pattern_special_quote = re.compile(r"’")


def main(args):
    """
    dataset = load_dataset(args.dataset, args.config, split=args.split, use_auth_token=True)

    # b/ remove irrelevant chars
    def remove_and_replace_special_characters(batch):
        text_ = batch[args.text_column_name].lower()
        text_ = pattern_special_quote.sub("'", text_)
        text_ = pattern_not_allowed_chars.sub("", text_)
        batch["target_text"] = text_
        return batch

    # debug
    # dataset = dataset.select(range(10))

    processed_dataset = dataset.map(
        remove_and_replace_special_characters,
        remove_columns=dataset.column_names,
        desc="remove special characters from datasets",
    )

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    tmp_text_path = os.path.join(args.outdir, "text.txt")
    with open(tmp_text_path, "w") as f:
        # f.write(" ".join(processed_dataset["target_text"]))
        f.write("\n".join(processed_dataset["target_text"]))
    """

    tmp_text_path = "/home/bhuang/transformers/examples/pytorch/speech-recognition/data/big/training_data_wo_trailing_space.txt"
    args.outdir = "/home/bhuang/transformers/examples/pytorch/speech-recognition/outputs/lm/big_n"

    out_ngram_arpa_path = os.path.join(args.outdir, "5gram.arpa")
    os.system(f"{kenlm_bin}/lmplz -o 5 <{tmp_text_path} > {out_ngram_arpa_path}")

    # ?? if didn't add space to end
    # add </s>
    # new_out_ngram_arpa_path = os.path.join(args.outdir, "5gram_correct.arpa")
    # with open(out_ngram_arpa_path, "r") as read_file, open(new_out_ngram_arpa_path, "w") as write_file:
    #     has_added_eos = False
    #     for line in read_file:
    #         if not has_added_eos and "ngram 1=" in line:
    #             count = line.strip().split("=")[-1]
    #             write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
    #         elif not has_added_eos and "<s>" in line:
    #             write_file.write(line)
    #             write_file.write(line.replace("<s>", "</s>"))
    #             has_added_eos = True
    #         else:
    #             write_file.write(line)
    # out_ngram_arpa_path = new_out_ngram_arpa_path

    out_ngram_bin_path = os.path.join(args.outdir, "5gram.bin")
    os.system(f"{kenlm_bin}/build_binary {out_ngram_arpa_path} {out_ngram_bin_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument("--config", type=str, required=True, help="Config of the dataset. *E.g.* `'en'`  for Common Voice")
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset. *E.g.* `'train'`")
    parser.add_argument("--text_column_name", type=str, default="text", help="Text field")
    parser.add_argument("--outdir", type=str, required=True, help="Output dir")

    args = parser.parse_args()

    main(args)


## VERY IMPORTANT!!!:
# After the language model is created, one should open the file. one should add a `</s>`
# The file should have a structure which looks more or less as follows:

# \data\
# ngram 1=86586
# ngram 2=546387
# ngram 3=796581
# ngram 4=843999
# ngram 5=850874

# \1-grams:
# -5.7532206      <unk>   0
# 0       <s>     -0.06677356
# -3.4645514      drugi   -0.2088903
# ...

# Now it is very important also add a </s> token to the n-gram
# so that it can be correctly loaded. You can simple copy the line:

# 0       <s>     -0.06677356

# and change <s> to </s>. When doing this you should also inclease `ngram` by 1.
# The new ngram should look as follows:

# \data\
# ngram 1=86587
# ngram 2=546387
# ngram 3=796581
# ngram 4=843999
# ngram 5=850874

# \1-grams:
# -5.7532206      <unk>   0
# 0       <s>     -0.06677356
# 0       </s>     -0.06677356
# -3.4645514      drugi   -0.2088903
# ...

# Now the ngram can be correctly used with `pyctcdecode`
