#!/usr/bin/env python
# Copyright 2021  Bofeng Huang

import argparse
import os
import re
from typing import Dict
from tqdm import tqdm
from pathlib import Path

import torch
import torchaudio
import datasets
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
from transformers.pipelines.pt_utils import KeyDataset

tmp_audio_column_name = "path"


def normalize_text(text: str, invalid_chars_regex: str) -> str:
    """ DO ADAPT FOR YOUR USE CASE. this function normalizes the target text. """

    text = text.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(invalid_chars_regex, " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main(args):
    # load dataset
    # dataset = load_dataset(args.dataset, args.config, split=args.split, use_auth_token=True)
    dataset = load_dataset("csv", data_files=args.test_csv_file)["train"]

    # for testing: only process the first two examples as a test
    # dataset = dataset.select(range(10))

    # read audio segments by timestamps
    sr = 8_000

    def read_segments(example):
        start = int(example[args.start_column_name] * sr)
        end = int(example[args.end_column_name] * sr)
        num_frames = end - start

        sig, _ = torchaudio.load(example[args.audio_column_name], frame_offset=start, num_frames=num_frames)
        # ! read full wavs
        # sig, _ = torchaudio.load(example[args.audio_column_name], frame_offset=0, num_frames=-1)

        # todo: casted audio column has no more audio path
        example[tmp_audio_column_name] = example[args.audio_column_name]

        # example[args.audio_column_name] = {
        #     "path": example[args.audio_column_name],
        #     "array": sig.numpy()[0],
        #     # "array": sig[0],
        #     "sampling_rate": sr,
        # }
        example[tmp_audio_column_name] = {
            "path": example[tmp_audio_column_name],
            "array": sig.numpy()[0],
            # "array": sig[0],
            "sampling_rate": sr,
        }

        return example

    dataset = dataset.map(read_segments, num_proc=args.preprocessing_num_workers, desc="read audio segments by timestamps")
    # dataset = dataset.cast_column(args.audio_column_name, datasets.features.Audio(sampling_rate=sr))
    dataset = dataset.cast_column(tmp_audio_column_name, datasets.features.Audio(sampling_rate=sr))
    # print(dataset[0])
    # print(dataset[0][args.audio_column_name])
    # print(dataset[0]["wav"]["array"].shape)
    # quit()

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
    # dataset = dataset.cast_column(args.audio_column_name, Audio(sampling_rate=sampling_rate))
    dataset = dataset.cast_column(tmp_audio_column_name, Audio(sampling_rate=sampling_rate))
    # print(dataset[0])
    # print(dataset[0]["wav"]["array"].shape)

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
            # KeyDataset(dataset, args.audio_column_name),
            KeyDataset(dataset, tmp_audio_column_name),
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
        example["word"] = predictions[idx]
        # todo: some inverse text normalization here if necessary
        # example["target"] = normalize_text(example[args.text_column_name], invalid_chars_regex)
        return example

    # result = dataset.map(collect_result, remove_columns=dataset.column_names, with_indices=True)
    result = dataset.map(collect_result, with_indices=True)
    # """

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    result.to_csv(f"{args.outdir}/predictions.txt", index=False, columns=["ID", args.audio_column_name, args.start_column_name, args.end_column_name, "word"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers")
    parser.add_argument("--test_csv_file", type=str, required=True, help="test csv path")
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument("--config", type=str, help="Config of the dataset. *E.g.* `'en'`  for Common Voice")
    parser.add_argument("--split", type=str, help="Split of the dataset. *E.g.* `'test'`")
    parser.add_argument("--audio_column_name", type=str, default="audio")
    parser.add_argument("--text_column_name", type=str, default="sentence")
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

    args = parser.parse_args()

    main(args)
