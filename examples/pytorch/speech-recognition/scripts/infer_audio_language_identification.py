#/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Run audio language identification.

Usage:
python infer_audio_language_identification.py \
    --input_file_path /projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-10/train_asr_processed_dedup256_shuffled.json \
    --output_file_path /projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-10/train_asr_processed_dedup256_shuffled_lang.json \
    --only_first_seconds 5.0 \
    --batch_size 64 \
    --dataloader_num_workers 4
"""

import torch
import json
import fire
import time
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Dict, Any, Union
import math
from speechbrain.pretrained import EncoderClassifier

# import soundfile as sf
from dataclasses import dataclass
from tqdm import tqdm
from torch.nn import functional as F
import pandas as pd

SAMPLE_RATE = 16_000


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
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(waveform, sample_rate, effects)
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
        import torchaudio
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    # waveform, sample_rate = sf.read(file_path, start=start, frames=frames, dtype="float32", always_2d=True)
    # waveform = waveform.T  # T x C -> C x T

    waveform, sample_rate = torchaudio.load(file_path, channels_first=True, frame_offset=start, num_frames=frames)

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
        # input_file_path: str,
        segments: List[Dict],
        audio_column_name: str = "audio_filepath",
        sample_rate: int = SAMPLE_RATE,
        only_first_seconds: Optional[float] = None,
    ):
        # self.dataset = jsonl_load(input_file_path)
        self.dataset = segments

        # debug
        # self.dataset = self.dataset[:100]

        print(f"Loaded {len(self.dataset)} examples")

        self.audio_column_name = audio_column_name
        self.sample_rate = sample_rate
        self.only_first_seconds = only_first_seconds

    def __getitem__(self, n: int):
        sample = self.dataset[n]

        # or speechbrain version
        # signal = language_id.load_audio(sample[self.audio_column_name])
        waveform, _ = get_waveform(
            sample[self.audio_column_name], mono=True, always_2d=False, output_sample_rate=self.sample_rate
        )

        if self.only_first_seconds is not None:
            waveform = waveform[: int(self.only_first_seconds * self.sample_rate)]

        return dict(wav=waveform, wav_len=waveform.shape[0])

    def __len__(self):
        return len(self.dataset)


@dataclass
class DataCollatorWithPadding:
    pad_to_multiple_of: Optional[int] = None
    padding_value: float = 0.0

    def __call__(self, features: List[Dict[str, Any]]):
        wavs, wav_lens = tuple([feature[key] for feature in features] for key in ("wav", "wav_len"))

        if self.pad_to_multiple_of is not None:
            # Pad the longest example to pad_to_multiple_of * N
            max_length_index, max_length = max(enumerate([len(wav) for wav in wavs]), key=lambda x: x[1])
            n_padding = math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of - max_length
            wavs[max_length_index] = torch.cat((wavs[max_length_index], torch.full((n_padding,), self.padding_value)))

        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=self.padding_value)
        wav_lens = torch.LongTensor(wav_lens, device=wavs.device)

        return dict(wavs=wavs, wav_lens=wav_lens)


def main(
    input_file_path: str,
    output_file_path: str,
    model_name_or_path: str = "speechbrain/lang-id-voxlingua107-ecapa",
    device: Union[str, int] = "cuda",
    only_first_seconds: Optional[float] = None,
    label_name: str = "fr: French",
    batch_size: int = 8,
    dataloader_num_workers: int = 1,
    audio_column_name: str = "audio_filepath",
):
    model = EncoderClassifier.from_hparams(source=model_name_or_path, run_opts={"device": device})
    label_index = model.hparams.label_encoder.encode_label(label_name)

    segments = jsonl_load(input_file_path)

    # debug
    # segments = segments[:100]

    dataset = SpeechDataset(
        segments, audio_column_name=audio_column_name, only_first_seconds=only_first_seconds
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
        collate_fn=DataCollatorWithPadding(pad_to_multiple_of=8),
    )

    start_time = time.perf_counter()

    indexed_scores = []
    predicted_labels = []
    predicted_scores = []

    for input_dict in tqdm(dataloader):
        wavs = input_dict["wavs"].to(device)
        wav_lens = input_dict["wav_lens"].to(device)

        with torch.inference_mode():
            logits, _, _, predicted_labels_ = model.classify_batch(wavs, wav_lens)
            scores = F.softmax(logits, dim=-1)
            predicted_scores_, predicted_labels_ids_ = torch.max(scores, dim=-1)
            # predicted_labels_ = model.hparams.label_encoder.decode_torch(predicted_labels_ids_)
            indexed_scores_ = scores[:, label_index]
            indexed_scores.extend(indexed_scores_.tolist())
            predicted_labels.extend(predicted_labels_)
            predicted_scores.extend(predicted_scores_.tolist())

    print(f'Inference time: {time.strftime("%Hh%Mm%Ss", time.gmtime(time.perf_counter() - start_time))}')

    # df = pd.read_json(input_file_path, lines=True)
    df = pd.DataFrame(segments)
    df[f"{label_name}_scores"] = indexed_scores
    df["predicted_labels"] = predicted_labels
    df["predicted_scores"] = predicted_scores

    jsonl_dump(df.to_dict("records"), output_file_path)


if __name__ == "__main__":
    fire.Fire(main)
