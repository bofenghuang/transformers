#!/usr/bin/env python
# Copyright 2022  Bofeng Huang

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import datasets
import torch
import torchaudio
import torchaudio.sox_effects as ta_sox
from datasets import Audio, Dataset, load_dataset
from tqdm import tqdm

import evaluate
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

# NB
# from utils.normalize_french import FrenchTextNormalizer
from utils.normalize_french_zaion import FrenchTextNormalizer


text_normalizer = FrenchTextNormalizer()

sys.path.append("/home/bhuang/my-scripts")
from myscripts.data.text.wer.compute_wer import compute_wer


def log_results(result: Dataset, args: Dict[str, str]):
    """DO NOT CHANGE. This function computes and logs the result metrics."""

    log_outputs = args.log_outputs

    def _log_results(result: Dataset, target_file, pred_file):
        # log all results in text file. Possibly interesting for analysis
        with open(pred_file, "w") as p, open(target_file, "w") as t:

            # mapping function to write output
            # def write_to_file(batch, i):
            #     p.write(f"{i}" + "\n")
            #     p.write(batch["prediction"] + "\n")
            #     t.write(f"{i}" + "\n")
            #     t.write(batch["target"] + "\n")

            # ! to adjust for your cases
            def write_to_file(batch, i):
                # p.write(batch[args.id_column_name] + "\t" + str(batch[args.start_column_name]) + "\t" + str(batch[args.end_column_name]) + "\t" + batch["prediction"] + "\n")
                # t.write(batch[args.id_column_name] + "\t" + str(batch[args.start_column_name]) + "\t" + str(batch[args.end_column_name]) + "\t" + batch["target"] + "\n")
                p.write(batch[args.id_column_name] + "\t" + batch["prediction"] + "\n")
                t.write(batch[args.id_column_name] + "\t" + batch["target"] + "\n")

            result.map(write_to_file, with_indices=True)

    if log_outputs is not None:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        _log_results(result, target_file=f"{args.outdir}/log_targets_raw.txt", pred_file=f"{args.outdir}/log_predictions_raw.txt")

    def map_norm(example):

        def norm_func(s):
            # NB
            return text_normalizer(
                s, do_lowercase=True, do_ignore_words=True, symbols_to_keep="'-", do_standardize_numbers=True
            )

        example["target"] = norm_func(example["target"])
        example["prediction"] = norm_func(example["prediction"])
        return example

    result = result.map(map_norm, num_proc=args.preprocessing_num_workers, desc="normalize targets and predictions")

    # filtering out empty targets
    result = result.filter(lambda example: example["target"] != "")

    # load metric
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # compute metrics
    wer_result = wer_metric.compute(references=result["target"], predictions=result["prediction"])
    cer_result = cer_metric.compute(references=result["target"], predictions=result["prediction"])

    # print & log results
    result_str = f"WER: {wer_result}\n" f"CER: {cer_result}"
    print(result_str)

    # log all results in text file. Possibly interesting for analysis
    if log_outputs is not None:

        # with open(f"{args.outdir}/{dataset_id}_eval_results.txt", "w") as f:
        with open(f"{args.outdir}/eval_results.txt", "w") as f:
            f.write(result_str)

        _log_results(result, target_file=f"{args.outdir}/log_targets.txt", pred_file=f"{args.outdir}/log_predictions.txt")


def eval_results(result: Dataset, args: Dict[str, str], do_ignore_words=False):
    df = result.to_pandas()

    def norm_func(s):
        # NB
        return text_normalizer(
            s, do_lowercase=True, do_ignore_words=do_ignore_words, symbols_to_keep="'-", do_standardize_numbers=True
        )

    df["target"] = df["target"].map(norm_func)
    df["prediction"] = df["prediction"].map(norm_func)

    # filtering out empty targets
    df = df[df["target"] != ""]

    df["target"] = df["target"].str.split()
    targets = df.set_index(args.id_column_name)["target"].to_dict()

    df["prediction"] = df["prediction"].str.split()
    predictions = df.set_index(args.id_column_name)["prediction"].to_dict()

    out_dir_ = f"{args.outdir}/wer_summary_without_fillers" if do_ignore_words else f"{args.outdir}/wer_summary"
    compute_wer(targets, predictions, out_dir_, do_print_top_wer=True, do_catastrophic=True)


# def normalize_text(text: str, invalid_chars_regex: str) -> str:
#     """ DO ADAPT FOR YOUR USE CASE. this function normalizes the target text. """

#     text = text.lower()
#     text = re.sub(r"’", "'", text)
#     text = re.sub(invalid_chars_regex, " ", text)
#     text = re.sub(r"\s+", " ", text).strip()

#     return text


def main(args):
    # load dataset
    if args.test_csv_file is not None:
        dataset = load_dataset("csv", data_files=args.test_csv_file)["train"]
    elif args.dataset is not None:
        dataset = load_dataset(args.dataset, args.config, split=args.split, use_auth_token=True)
    else:
        raise ValueError("You have not specified a dataset name nor a custom validation file")

    # Debug: only process the first two examples as a test
    # dataset = dataset.select(range(10))
    # print(dataset[0])
    # quit()

    # NB: when got some real word "nan" in csv file (dekuple)
    def correct_text(example):
        example[args.text_column_name] = (
            example[args.text_column_name] if example[args.text_column_name] is not None else "nan"
        )
        return example

    # dataset = dataset.map(correct_text, num_proc=args.preprocessing_num_workers, desc="Correct readed text")
    # dataset = dataset.sort(args.length_column_name)
    # print(dataset["wrd"][:100])
    # print(None in dataset["wrd"])
    # quit()

    if args.test_csv_file is not None:
        # read audio segments by timestamps
        sr = 8_000

        def read_segments(example):
            start = int(float(example[args.start_column_name]) * sr)
            end = int(float(example[args.end_column_name]) * sr)
            num_frames = end - start

            waveform, sample_rate = torchaudio.load(example[args.audio_column_name], frame_offset=start, num_frames=num_frames)
            # ! read full wavs
            # waveform, _ = torchaudio.load(example[args.audio_column_name], frame_offset=0, num_frames=-1)

            # bh: normalize wav before computing fbanks
            # effects = []
            # # normalize
            # effects.append(["gain", "-n"])
            # # sampling
            # # effects.append(["rate", f"{to_sample_rate}"])
            # # to mono
            # # effects.append(["channels", "1"])
            # converted_waveform, converted_sample_rate = ta_sox.apply_effects_tensor(
            #     waveform, sample_rate, effects
            # )
            # converted_waveform = converted_waveform.squeeze(axis=0)  # mono

            converted_waveform = waveform.squeeze(axis=0)  # mono
            converted_sample_rate = sample_rate

            example[args.audio_column_name] = {
                "path": example[args.audio_column_name],
                "array": converted_waveform.numpy(),
                # "array": waveform,
                "sampling_rate": converted_sample_rate,
            }

            return example

        # """
        dataset = dataset.map(read_segments, num_proc=args.preprocessing_num_workers, desc="read audio segments by timestamps")
        dataset = dataset.cast_column(args.audio_column_name, datasets.features.Audio(sampling_rate=sr))
        print(dataset[0])
        print(dataset[0][args.audio_column_name])
        print(dataset[0][args.audio_column_name]["array"].shape)
        # dataset.save_to_disk("/projects/bhuang/.cache/hf_outputs/asr/hmhm_test/dump_readed")
        # """
        # dataset = datasets.load_from_disk("/projects/bhuang/.cache/hf_outputs/asr/hmhm_test/dump_readed")

    # load processor
    if args.greedy:
        processor = Wav2Vec2Processor.from_pretrained(args.model_id)
        decoder = None
    else:
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(args.model_id)
        decoder = processor.decoder

    # ! reset lm weight
    if args.decoder_alpha is not None:
        decoder.reset_params(alpha=args.decoder_alpha)
        print(f"Reset `decoder_alpha` to {args.decoder_alpha}")

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    sampling_rate = feature_extractor.sampling_rate

    # ! don't normalize waveform
    # feature_extractor.do_normalize = False

    # resample audio
    dataset = dataset.cast_column(args.audio_column_name, Audio(sampling_rate=sampling_rate))

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
    """
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
    """

    # normalize_to_lower = False
    # for token in non_special_tokens:
    #     if token.isalpha() and token.islower():
    #         normalize_to_lower = True
    #         break

    # bh: inference one by one
    """
    # map function to decode audio
    def map_to_pred(batch):
        # prediction = asr(batch["audio"]["array"], chunk_length_s=args.chunk_length_s, stride_length_s=args.stride_length_s)
        prediction = asr(batch[args.audio_column_name]["array"], chunk_length_s=args.chunk_length_s, stride_length_s=args.stride_length_s)

        batch["prediction"] = prediction["text"]
        batch["target"] = normalize_text(batch[args.text_column_name], invalid_chars_regex)
        return batch

    # run inference on all examples
    result = dataset.map(map_to_pred, remove_columns=dataset.column_names)
    """

    # bh: inference by batch
    # """
    # desceding order to early check if OOM
    # dataset = dataset.sort(args.length_column_name, reverse=True)

    predictions = []
    for out in tqdm(
        asr(
            KeyDataset(dataset, args.audio_column_name),
            chunk_length_s=args.chunk_length_s,
            stride_length_s=args.stride_length_s,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        ),
        total=len(dataset),
    ):
        # Exactly the same output as before, but the content are passed
        # as batches to the model
        # print(out)
        predictions.append(out["text"])

    def collect_result(example, idx):
        example["prediction"] = predictions[idx]
        example["target"] = example[args.text_column_name]
        # example["target"] = normalize_text(example[args.text_column_name], invalid_chars_regex)
        return example

    # result = dataset.map(collect_result, remove_columns=dataset.column_names, with_indices=True)
    result = dataset.map(collect_result, with_indices=True)
    # """

    # todo
    # bh: inference by batch with low APIs

    # bh: fake ID if not exists
    if args.id_column_name not in result.features.keys():
        result = result.map(lambda example, idx: {**example, args.id_column_name: f"{idx:09d}"}, with_indices=True)

    # filtering out empty targets
    # result = result.filter(lambda example: example["target"] != "")

    # compute and log_results
    # do not change function below
    log_results(result, args)

    # speechbrain's WER computation util
    eval_results(result, args, do_ignore_words=False)
    eval_results(result, args, do_ignore_words=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers")
    parser.add_argument("--test_csv_file", type=str, default=None, help="test csv path")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        # required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument("--config", type=str, help="Config of the dataset. *E.g.* `'en'`  for Common Voice")
    parser.add_argument("--split", type=str, help="Split of the dataset. *E.g.* `'test'`")
    parser.add_argument("--audio_column_name", type=str, default="audio")
    parser.add_argument("--text_column_name", type=str, default="sentence")
    parser.add_argument("--id_column_name", type=str, default="ID")
    parser.add_argument("--start_column_name", type=str, default="start")
    parser.add_argument("--end_column_name", type=str, default="end")
    parser.add_argument("--length_column_name", type=str, default="length")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
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
    parser.add_argument("--decoder_alpha", type=float, default=None, help="")

    args = parser.parse_args()

    main(args)
