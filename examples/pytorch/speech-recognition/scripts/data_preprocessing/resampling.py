#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

import os

import fire
import soundfile as sf
from data_utils import print_dataset_info, write_dataset_to_json
from datasets import load_dataset

from utils.audio_utils import get_waveform

SAMPLE_RATE = 16_000


def main(
    input_file: str,
    # output_file: str,
    num_workers: int = 64,
):
    dataset = load_dataset("json", data_files=input_file, split="train")
    print_dataset_info(dataset)

    def process_function(example):
        audio_file = example["audio_filepath"]
        # todo
        processed_audio_file = audio_file.replace("/audios/", "/audios_16k/")
        os.makedirs(os.path.dirname(processed_audio_file), exist_ok=True)

        # read waveform from audio files
        waveform, _ = get_waveform(
            audio_file,
            mono=True,
            output_sample_rate=SAMPLE_RATE,
            always_2d=False,
        )

        sf.write(processed_audio_file, waveform, samplerate=SAMPLE_RATE, format="wav")

        example["audio_filepath"] = processed_audio_file

        return example

    dataset = dataset.map(process_function, num_proc=num_workers)

    # export
    output_file = input_file.rsplit(".", 1)[0] + "_16k.jsonl"
    write_dataset_to_json(dataset, output_file_path=output_file, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
