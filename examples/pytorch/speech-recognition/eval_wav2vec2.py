#!/usr/bin/env python

import argparse
import os
import re
from typing import Dict

import torch
from datasets import Audio, Dataset, load_dataset, load_metric

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    pipeline,
)


def log_results(result: Dataset, args: Dict[str, str]):
    """ DO NOT CHANGE. This function computes and logs the result metrics. """

    log_outputs = args.log_outputs
    dataset_id = "_".join(args.dataset.split("/") + [args.config, args.split])

    # load metric
    wer = load_metric("wer")
    cer = load_metric("cer")

    # compute metrics
    wer_result = wer.compute(references=result["target"], predictions=result["prediction"])
    cer_result = cer.compute(references=result["target"], predictions=result["prediction"])

    # print & log results
    result_str = f"WER: {wer_result}\n" f"CER: {cer_result}"
    print(result_str)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    with open(f"{args.outdir}/{dataset_id}_eval_results.txt", "w") as f:
        f.write(result_str)

    # log all results in text file. Possibly interesting for analysis
    if log_outputs is not None:
        pred_file = f"{args.outdir}/log_{dataset_id}_predictions.txt"
        target_file = f"{args.outdir}/log_{dataset_id}_targets.txt"

        with open(pred_file, "w") as p, open(target_file, "w") as t:

            # mapping function to write output
            def write_to_file(batch, i):
                p.write(f"{i}" + "\n")
                p.write(batch["prediction"] + "\n")
                t.write(f"{i}" + "\n")
                t.write(batch["target"] + "\n")

            result.map(write_to_file, with_indices=True)


def normalize_text(text: str, invalid_chars_regex: str) -> str:
    """ DO ADAPT FOR YOUR USE CASE. this function normalizes the target text. """

    text = text.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(invalid_chars_regex, " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main(args):
    # load dataset
    dataset = load_dataset(args.dataset, args.config, split=args.split, use_auth_token=True)

    # data_files = [
    #     "/projects/bhuang/corpus/speech/multilingual-tedx/data_test.csv",
    # ]
    # dataset = load_dataset("csv", data_files=data_files)["train"]
    # dataset = dataset.remove_columns(["duration", "text", "id"])
    # dataset = dataset.rename_column("normalized_text", "sentence")
    # # dataset = dataset.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))
    # dataset = dataset.filter(lambda example: example["sentence"] is not None)

    # for testing: only process the first two examples as a test
    # dataset = dataset.select(range(10))

    # load processor
    if args.greedy:
        processor = Wav2Vec2Processor.from_pretrained(args.model_id)
        decoder = None
    else:
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(args.model_id)
        decoder = processor.decoder

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    sampling_rate = feature_extractor.sampling_rate

    # resample audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # load eval pipeline
    if args.device is None:
        args.device = 0 if torch.cuda.is_available() else -1

    config = AutoConfig.from_pretrained(args.model_id)
    model = AutoModelForCTC.from_pretrained(args.model_id)

    # asr = pipeline("automatic-speech-recognition", model=args.model_id, device=args.device)
    asr = pipeline(
        "automatic-speech-recognition",
        config=config,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        decoder=decoder,
        device=args.device,
    )

    # build normalizer config
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokens = [x for x in tokenizer.convert_ids_to_tokens(range(0, tokenizer.vocab_size))]
    special_tokens = [
        tokenizer.pad_token,
        tokenizer.word_delimiter_token,
        tokenizer.unk_token,
        tokenizer.bos_token,
        tokenizer.eos_token,
    ]
    non_special_tokens = [x for x in tokens if x not in special_tokens]
    invalid_chars_regex = f"[^\s{re.escape(''.join(set(non_special_tokens)))}]"

    # normalize_to_lower = False
    # for token in non_special_tokens:
    #     if token.isalpha() and token.islower():
    #         normalize_to_lower = True
    #         break

    # map function to decode audio
    def map_to_pred(batch):
        prediction = asr(batch["audio"]["array"], chunk_length_s=args.chunk_length_s, stride_length_s=args.stride_length_s)

        batch["prediction"] = prediction["text"]
        # todo
        # batch["prediction"] = re.sub(r"'\s+", "'", prediction["text"])
        # batch["target"] = normalize_text(batch["sentence"], invalid_chars_regex)
        batch["target"] = normalize_text(batch[args.text_column_name], invalid_chars_regex)
        return batch

    # run inference on all examples
    result = dataset.map(map_to_pred, remove_columns=dataset.column_names)

    # filtering out empty targets
    result = result.filter(lambda example: example["target"] != "")

    # compute and log_results
    # do not change function below
    log_results(result, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument("--config", type=str, required=True, help="Config of the dataset. *E.g.* `'en'`  for Common Voice")
    parser.add_argument("--split", type=str, required=True, help="Split of the dataset. *E.g.* `'test'`")
    parser.add_argument("--text_column_name", type=str, default="sentence", help="")
    parser.add_argument(
        "--chunk_length_s",
        type=float,
        default=None,
        help="Chunk length in seconds. Defaults to None. For long audio files a good value would be 5.0 seconds.",
    )
    parser.add_argument(
        "--stride_length_s",
        type=float,
        default=None,
        help="Stride of the audio chunks. Defaults to None. For long audio files a good value would be 1.0 seconds.",
    )
    parser.add_argument("--log_outputs", action="store_true", help="If defined, write outputs to log file for analysis.")
    parser.add_argument("--greedy", action="store_true", help="If defined, the LM will be ignored during inference.")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument("--outdir", type=str, required=True, help="Save path.")

    args = parser.parse_args()

    main(args)
