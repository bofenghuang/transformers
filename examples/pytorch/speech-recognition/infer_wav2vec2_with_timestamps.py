#!/usr/bin/env python
# Copyright 2023  Bofeng Huang

import argparse
import json
import os
from pathlib import Path

import datasets
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, Dataset, load_dataset
from tqdm import tqdm
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


def read_json_manifest(input_json_file):
    data = []
    with open(input_json_file, "r") as inf:
        for line in inf:
            data.append(json.loads(line.strip()))
    return data


def write_json_manifest(output_json_file, data):
    output_dir = os.path.dirname(output_json_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_json_file, "w") as outf:
        for row in data:
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(args):
    # load dataset
    # dataset = load_dataset(args.dataset, args.config, split=args.split, use_auth_token=True)
    extension = args.input_file_path.rsplit(".", 1)[-1]
    load_kwargs = {}
    if extension == "tsv":
        load_kwargs["sep"] = "\t"
    dataset = load_dataset("csv" if extension == "tsv" else extension, data_files=args.input_file_path, **load_kwargs)["train"]
    print(dataset)

    # data = read_json_manifest(args.input_file_path)
    # print(data)
    # quit()

    # for testing: only process the first two examples as a test
    # dataset = dataset.select(range(10))

    # read audio segments by timestamps
    # sampling_rate = 8_000
    sampling_rate = args.sampling_rate

    def read_segments(example):
        start = int(float(example[args.start_column_name]) * sampling_rate)
        end = int(float(example[args.end_column_name]) * sampling_rate)
        num_frames = end - start

        # waveform, _ = torchaudio.load(example[args.audio_column_name], frame_offset=start, num_frames=num_frames)
        # # ! read full wavs
        # # waveform, _ = torchaudio.load(example[args.audio_column_name], frame_offset=0, num_frames=-1)

        # waveform = waveform.squeeze(axis=0)  # mono

        # example[args.audio_column_name] = {
        #     "path": example[args.audio_column_name],
        #     "array": waveform.numpy(),
        #     "sampling_rate": sampling_rate,
        # }

        waveform, _ = sf.read(example[args.audio_column_name], start=start, frames=num_frames, dtype="float32", always_2d=True)
        waveform = waveform[:, 0]  # mono

        # NB: tmp audio_column_name
        example[f"tmp_{args.audio_column_name}"] = example[args.audio_column_name]

        example[args.audio_column_name] = {
            "path": example[args.audio_column_name],
            "array": waveform,
            "sampling_rate": sampling_rate,
        }

        return example

    dataset = dataset.map(read_segments, num_proc=args.preprocessing_num_workers, desc="read audio segments by timestamps")
    dataset = dataset.cast_column(args.audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate))
    # print(dataset)
    # print(dataset[0][args.audio_column_name])
    # print(dataset[0]["wav"]["array"].shape)
    # quit()

    # load processor
    if args.greedy:
        processor = Wav2Vec2Processor.from_pretrained(args.model_name_or_path)
        decoder = None
    else:
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(args.model_name_or_path)
        decoder = processor.decoder

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    model_sampling_rate = feature_extractor.sampling_rate

    # resample audio
    if sampling_rate != model_sampling_rate:
        dataset = dataset.cast_column(args.audio_column_name, Audio(sampling_rate=model_sampling_rate))
    # print(dataset[0])
    # print(dataset[0]["wav"]["array"].shape)

    if args.device is None:
        args.device = 0 if torch.cuda.is_available() else -1

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCTC.from_pretrained(args.model_name_or_path)

    # asr = pipeline("automatic-speech-recognition", model=args.model_name_or_path, device=args.device)
    pipe = pipeline(
        "automatic-speech-recognition",
        config=config,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        decoder=decoder,
        device=args.device,
    )

    # bh: inference one by one
    """
    # map function to decode audio
    def map_to_pred(batch):
        # prediction = pipe(batch["audio"]["array"], chunk_length_s=args.chunk_length_s, stride_length_s=args.stride_length_s)
        prediction = pipe(batch[args.audio_column_name]["array"], chunk_length_s=args.chunk_length_s, stride_length_s=args.stride_length_s)

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
        pipe(
            KeyDataset(dataset, args.audio_column_name),
            chunk_length_s=args.chunk_length_s,
            stride_length_s=args.stride_length_s,
            # return_timestamps=args.return_timestamps,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        ),
        total=len(dataset),
    ):
        # Exactly the same output as before, but the content are passed
        # as batches to the model
        # print(out)
        # predictions.append(out["text"])

        out_ = {"text": out["text"]}
        # if args.return_nbest:
        #     out_["nbest"] = out["nbest"]
        # if args.return_timestamps:
        #     out_["word_offsets"] = [{"text": chunk["text"], "start": chunk["timestamp"][0], "end": chunk["timestamp"][1]} for chunk in out["chunks"]]

        predictions.append(out_)

    def collect_result(example, idx):
        # example["word"] = predictions[idx]
        example.update(predictions[idx])
        # todo: some inverse text normalization here if necessary
        # example["target"] = normalize_text(example[args.text_column_name], invalid_chars_regex)
        return example

    # result = dataset.map(collect_result, remove_columns=dataset.column_names, with_indices=True)
    result = dataset.map(collect_result, with_indices=True)
    # bh: put back raw audio_column_name
    result = result.remove_columns(args.audio_column_name)
    result = result.rename_column(f"tmp_{args.audio_column_name}", args.audio_column_name)
    print(result)
    print(result[0])
    # """

    os.makedirs(os.path.dirname(args.prediction_file), exist_ok=True)

    extension = args.prediction_file.rsplit(".", 1)[-1]
    if extension in {"csv", "tsv"}:
        write_kwargs = {"sep": "\t"} if extension == "tsv" else {}
        result.to_csv(
            args.prediction_file,
            index=False,
            **write_kwargs,
            # columns=["ID", args.audio_column_name, args.start_column_name, args.end_column_name, "word"],
        )
    elif extension == "json":
        result.to_json(args.prediction_file)
    else:
        raise ValueError("Invalid extension for prediction_file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers"
    )
    parser.add_argument("--input_file_path", type=str, required=True, help="test csv path")
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        help="Dataset name to evaluate the `model_name_or_path`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument("--config", type=str, help="Config of the dataset. *E.g.* `'en'`  for Common Voice")
    parser.add_argument("--split", type=str, help="Split of the dataset. *E.g.* `'test'`")
    parser.add_argument("--sampling_rate", type=int, default=8_000)
    parser.add_argument("--audio_column_name", type=str, default="audio")
    parser.add_argument("--text_column_name", type=str, default="sentence")
    parser.add_argument("--start_column_name", type=str, default="start")
    parser.add_argument("--end_column_name", type=str, default="end")
    parser.add_argument("--length_column_name", type=str, default="length")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    # parser.add_argument("--return_timestamps", type=str, default=None, help="`char` or `word`")
    # parser.add_argument("--return_nbest", action="store_true", help="")
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
    parser.add_argument("--prediction_file", type=str, required=True, help="Save path.")

    args = parser.parse_args()

    main(args)
