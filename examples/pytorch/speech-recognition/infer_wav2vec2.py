#!/usr/bin/env python
# Copyright 2023  Bofeng Huang

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fire
import pandas as pd
import soundfile as sf
import torch
import transformers
from myscripts.audio.audio_utils import get_waveform
from torch.utils.data import DataLoader, Dataset
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


def normalize_text(s):
    s = re.sub(r"\s*'\s*", "'", s)  # standardize when there's a space before/after an apostrophe
    s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space
    return s


class Wav2vec2SpeechDataset(Dataset):
    def __init__(
        self,
        segments: List[Dict],
        processor: Any,
        # device: Union[str, torch.device] = "cpu",
        model_sampling_rate: Optional[int] = None,
        id_column_name: str = "ID",
        audio_column_name: str = "wav",
        start_column_name: str = "start",
        end_column_name: str = "end",
    ):
        self.id_column_name = id_column_name
        self.audio_column_name = audio_column_name
        self.start_column_name = start_column_name
        self.end_column_name = end_column_name

        self.data = self.preprocess(segments)

        self.processor = processor
        self.model_sampling_rate = model_sampling_rate
        # self.normalize_volume = normalize_volume

    def read_csv(self, file_path: str):
        with open(file_path, newline="") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                yield row

    def preprocess(self, segments: List[Dict]):

        data = []
        for audio_path, segments_group in groupby(segments, lambda x: x[self.audio_column_name]):
            sampling_rate = sf.info(audio_path).samplerate
            segments_group = sorted(segments_group, key=lambda x: float(x[self.start_column_name]))

            for segment in segments_group:
                start = int(float(segment[self.start_column_name]) * sampling_rate)
                end = int(float(segment[self.end_column_name]) * sampling_rate)
                n_frames = end - start
                # data.append((segment["wav"], offset, n_frames, segment["wrd"], sampling_rate, segment["ID"],))
                # data.append(
                #     {
                #         "ID": segment[self.id_column_name],
                #         "wav": audio_path,
                #         "start": start,
                #         "frames": n_frames,
                #         # "wrd": segment["wrd"],
                #         "sampling_rate": sampling_rate,
                #     }
                # )
                data.append((segment[self.id_column_name], audio_path, start, n_frames, sampling_rate))

        # todo: sort by length
        data = sorted(data, key=lambda x: x[3], reverse=True)

        return data

    def __getitem__(self, n: int):
        segment_idx, audio_path, start, n_frames, _ = self.data[n]

        # waveform, _ = torchaudio.load(audio_path, frame_offset=start, num_frames=n_frames)
        # # ! read full wavs
        # # waveform, _ = torchaudio.load(audio_path, frame_offset=0, num_frames=-1)
        # waveform = waveform.squeeze(axis=0)  # mono

        # waveform, _ = sf.read(audio_path, start=start, frames=n_frames, dtype="float32", always_2d=True)
        # waveform = waveform[:, 0]  # mono

        waveform, _ = get_waveform(
            audio_path,
            frames=n_frames,
            start=start,
            output_sample_rate=self.model_sampling_rate,
            normalize_volume=False,
            always_2d=False,
            # mono=False,
        )
        # waveform = torch.from_numpy(waveform)

        # todo: feature extractor here
        input_dict = self.processor(waveform, sampling_rate=self.model_sampling_rate)

        # return segment_idx, input_dict["input_values"][0]
        return {"idx": segment_idx, "input_values": input_dict["input_values"][0]}

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class DataCollatorWithPadding:

    processor: transformers.ProcessorMixin
    padding: Union[bool, str, transformers.utils.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        indexs = [feature["idx"] for feature in features]
        input_values = [{"input_values": feature["input_values"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_values,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["idx"] = indexs

        return batch


def main(
    input_file_path: str,
    output_file_path: str,
    model_name_or_path: str,
    fp16: bool = False,
    greedy: bool = False,
    device: Union[str, int] = 0,
    batch_size: int = 8,
    dataloader_num_workers: int = 1,
    id_column_name: str = "ID",
    audio_column_name: str = "wav",
    start_column_name: str = "start",
    end_column_name: str = "end",
    text_column_name: str = "text",
):

    df_data = pd.read_csv(input_file_path, sep="\t")
    # print(df_data.head())
    print(f"Number of segments: {df_data.shape[0]}")

    # debug
    # df_data = df_data.head(10)

    # load processor
    if greedy:
        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        # decoder = None
    else:
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name_or_path)
        # decoder = processor.decoder

    feature_extractor = processor.feature_extractor
    # tokenizer = processor.tokenizer
    model_sampling_rate = feature_extractor.sampling_rate

    # if device is None:
    #     device = 0 if torch.cuda.is_available() else -1
    device = torch.device(device)

    # config = AutoConfig.from_pretrained(model_name_or_path)
    model_args = dict()
    if fp16:
        model_args["torch_dtype"] = torch.float16
    model = AutoModelForCTC.from_pretrained(model_name_or_path, **model_args)
    model = model.to(device)

    dataset = Wav2vec2SpeechDataset(
        df_data.to_dict("records"),
        processor=processor,
        model_sampling_rate=model_sampling_rate,
        id_column_name=id_column_name,
        audio_column_name=audio_column_name,
        start_column_name=start_column_name,
        end_column_name=end_column_name,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
        collate_fn=DataCollatorWithPadding(processor=processor, pad_to_multiple_of=8),
    )

    start_time = time.perf_counter()

    hypotheses = []
    indexes = []

    for input_dict in tqdm(dataloader):
        ids = input_dict["idx"]
        input_values = input_dict["input_values"]
        attention_mask = input_dict["attention_mask"]

        # todo: move to dataset
        if fp16:
            input_values = input_values.half()

        with torch.inference_mode():
            logits = model(input_values.to(device), attention_mask=attention_mask.to(device)).logits

        if greedy:
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentences = processor.batch_decode(predicted_ids)
        else:
            predicted_sentences = processor.batch_decode(logits.cpu().numpy()).text

        hypotheses.extend(predicted_sentences)
        indexes.extend(ids)

    print(f'Inference time: {time.strftime("%Hh%Mm%Ss", time.gmtime(time.perf_counter() - start_time))}')

    # postprocess
    hypotheses = list(map(normalize_text, hypotheses))

    df_data[text_column_name] = df_data[id_column_name].map({i: d for i, d in zip(indexes, hypotheses)})

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df_data.to_csv(output_file_path, index=False, sep="\t")
    print(f"Predicted results have been saved into {output_file_path}")


if __name__ == "__main__":
    fire.Fire(main)
