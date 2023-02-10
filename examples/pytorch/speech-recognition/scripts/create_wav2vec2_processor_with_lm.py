#!/usr/bin/env python
# coding=utf-8
# Copyright 2021  Bofeng Huang

import os
import argparse

from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ProcessorWithLM


def main(args):
    # arpa_path, arpa_extension = args.arpa_path.rsplit(".", 1)
    # new_arpa_path = f"{arpa_path}_correct.{arpa_extension}"

    # with open(args.arpa_path, "r") as read_file, open(new_arpa_path, "w") as write_file:
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

    # args.arpa_path = new_arpa_path

    processor = AutoProcessor.from_pretrained(args.model_id)

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    decoder = build_ctcdecoder(labels=list(sorted_vocab_dict.keys()), kenlm_model_path=args.arpa_path)

    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor, tokenizer=processor.tokenizer, decoder=decoder
    )

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    processor_with_lm.save_pretrained(args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers")
    parser.add_argument("--arpa_path", type=str, required=True, help="Arpa path.")
    parser.add_argument("--outdir", type=str, required=True, help="Save path.")

    args = parser.parse_args()

    main(args)
