#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import csv
import json
import os
import re
import torch
import time
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fire
import pandas as pd
import numpy as np
import soundfile as sf
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
)

SAMPLE_RATE = 16_000


def normalize_text(s):
    s = re.sub(r"\s*'\s*", "'", s)  # standardize when there's a space before/after an apostrophe
    s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space
    return s


def jsonl_load(file_path, mode="r"):
    with open(file_path, mode=mode) as f:
        return [json.loads(l.strip()) for l in f]


def jsonl_dump(data, file_path, mode="w", default=str, ensure_ascii=False):
    with open(file_path, mode=mode) as f:
        for item in tqdm(data, desc="Writing to json", total=len(data), unit=" samples"):
            f.write(f"{json.dumps(item, default=default, ensure_ascii=ensure_ascii)}\n")


# Copied and adapted from https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/audio/audio_utils.py#L22
def convert_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
):
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])

    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(_waveform, sample_rate, effects)
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate

    return waveform, sample_rate


# Copied and adapted from https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/audio/audio_utils.py#L69
def get_waveform(
    file_path: str,
    start: int = 0,
    frames: int = -1,
    normalize_volume: bool = False,
    mono: bool = True,
    output_sample_rate: Optional[int] = None,
    normalization: bool = True,
    always_2d: bool = True,
):
    try:
        import soundfile as sf

        # import torchaudio
    except ImportError:
        raise ImportError("Please install soundfile: pip install soundfile")
        # raise ImportError("Please install torchaudio: pip install torchaudio")

    waveform, sample_rate = sf.read(file_path, start=start, frames=frames, dtype="float32", always_2d=True)
    waveform = waveform.T  # T x C -> C x T

    # waveform, sample_rate = torchaudio.load(file_path, channels_first=True, frame_offset=start, num_frames=frames)

    waveform, sample_rate = convert_waveform(
        waveform, sample_rate, normalize_volume=normalize_volume, to_mono=mono, to_sample_rate=output_sample_rate,
    )

    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers

    if not always_2d:
        waveform = waveform.squeeze(axis=0)

    return waveform, sample_rate


class SpeechDataset(Dataset):
    def __init__(
        self,
        segments: List[Dict],
        processor: Any,
        audio_column_name: str = "audio_filepath",
        start_column_name: str = "start",
        end_column_name: str = "end",
        text_column_name: str = "text",
        sample_rate: int = SAMPLE_RATE,
        compute_ctc_loss: bool = False,
    ):
        # self.dataset = jsonl_load(input_file_path)
        self.dataset = segments

        # debug
        # self.dataset = self.dataset[:100]

        print(f"Loaded {len(self.dataset)} examples")

        self.processor = processor
        self.audio_column_name = audio_column_name
        self.start_column_name = start_column_name
        self.end_column_name = end_column_name
        self.text_column_name = text_column_name
        self.sample_rate = sample_rate
        self.compute_ctc_loss = compute_ctc_loss

        # self.dataset = self.preprocess(segments)

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
        sample = self.dataset[n]

        waveform, _ = get_waveform(
            sample[self.audio_column_name], mono=True, output_sample_rate=self.sample_rate, always_2d=False
        )

        input_dict = self.processor(waveform, sampling_rate=self.sample_rate)
        processed_input = dict(input_values=input_dict["input_values"][0])

        if self.compute_ctc_loss:
            processed_input["labels"] = self.processor(text=sample[self.text_column_name])["input_ids"]

        return processed_input

    def __len__(self):
        return len(self.dataset)


@dataclass
class DataCollatorWithPadding:

    processor: transformers.ProcessorMixin
    padding: Union[bool, str, transformers.utils.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    compute_ctc_loss: bool = False

    def __call__(self, features: List[Dict[str, Any]]):
        input_values = [{"input_values": feature["input_values"]} for feature in features]

        batch = self.processor.pad(
            input_values,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        if self.compute_ctc_loss:
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            labels_batch = self.processor.pad(
                labels=label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            # replace padding with -100 to ignore loss correctly
            batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        return batch


def main(
    input_file_path: str,
    model_name_or_path: str = "bhuang/wav2vec2-xls-r-1b-cv9-fr",
    device: Union[str, int] = 0,
    fp16: bool = False, 
    greedy: bool = True,
    batch_size: int = 8,
    dataloader_num_workers: int = 1,
    compute_ctc_loss: bool = False,
    audio_column_name: str = "audio_filepath",
    start_column_name: str = "start",
    end_column_name: str = "end",
    text_column_name: str = "text",
    suffix: str = "_predwav2vec2greedy",
):

    path, ext = input_file_path.rsplit(".", 1)
    output_file_path = f"{path}{suffix}.{ext}"

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

    # torch_dtype = torch.float16 if fp16 else torch.float32

    # config = AutoConfig.from_pretrained(model_name_or_path)
    model_args = dict()
    if fp16:
        model_args["torch_dtype"] = torch.float16
    model = AutoModelForCTC.from_pretrained(
        model_name_or_path,
        # device_map=device,
        # torch_dtype=torch_dtype,
        # low_cpu_mem_usage=True,
        **model_args,
    )
    model.eval()
    model = model.to(device)
    print("Model has been loaded")

    # Activate Torch Scale-Product-Attention (SDPA)
    # Nice to have when GPU doesn't support Flash-Attention
    # todo: doesn't support mask?
    # model = model.to_bettertransformer()

    if compute_ctc_loss:
        model.config.ctc_loss_reduction = "none"
        model.config.ctc_zero_infinity = True

    # load data
    segments = jsonl_load(input_file_path)

    # debug
    # segments = segments[:100]

    # sort by duration
    segments = sorted(segments, key=lambda x: x["duration"], reverse=True)

    dataset = SpeechDataset(
        segments,
        processor=processor,
        audio_column_name=audio_column_name,
        start_column_name=start_column_name,
        end_column_name=end_column_name,
        text_column_name=text_column_name,
        sample_rate=model_sampling_rate,
        compute_ctc_loss=compute_ctc_loss,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
        collate_fn=DataCollatorWithPadding(processor=processor, pad_to_multiple_of=8, compute_ctc_loss=compute_ctc_loss),
    )

    start_time = time.perf_counter()

    hypotheses = []
    losses = []

    for input_dict in tqdm(dataloader):
        input_values = input_dict["input_values"].to(device)
        attention_mask = input_dict["attention_mask"].to(device)
        labels = input_dict["labels"].to(device) if compute_ctc_loss else None

        # todo: move to dataset
        if fp16:
            input_values = input_values.half()

        with torch.inference_mode():
            model_outputs = model(input_values, attention_mask=attention_mask, labels=labels)
            logits = model_outputs.logits
            loss = model_outputs.loss
            if compute_ctc_loss:
                # averaged by the target lengths
                loss = loss / labels.ne(-100).sum(-1)
                loss = loss.tolist()

        if greedy:
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentences = processor.batch_decode(predicted_ids)
        else:
            predicted_sentences = processor.batch_decode(logits.cpu().numpy()).text

        hypotheses.extend(predicted_sentences)
        losses.extend(loss)

    print(f'Inference time: {time.strftime("%Hh%Mm%Ss", time.gmtime(time.perf_counter() - start_time))}')

    # todo: postprocess
    # hypotheses = list(map(normalize_text, hypotheses))
    # df_data[text_column_name] = df_data[id_column_name].map({i: d for i, d in zip(indexes, hypotheses)})

    df = pd.DataFrame(segments)
    df["predicted_text"] = hypotheses
    if compute_ctc_loss:
        df["ctc_loss"] = losses

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    jsonl_dump(df.to_dict("records"), output_file_path)
    print(f"Predicted results have been saved into {output_file_path}")


if __name__ == "__main__":
    fire.Fire(main)
