#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang

"""Infer whisper models with HF pipeline (with built-in dataloader and sliding windows to infer >30s audio)."""

import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import soundfile as sf
import torch
import numpy as np
from datasets import Audio, load_dataset
from tqdm import tqdm
import fire

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset


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
        waveform,
        sample_rate,
        normalize_volume=normalize_volume,
        to_mono=mono,
        to_sample_rate=output_sample_rate,
    )

    if not normalization:
        waveform *= 2**15  # denormalized to 16-bit signed integers

    if not always_2d:
        waveform = waveform.squeeze(axis=0)

    return waveform, sample_rate


def main(
    model_name_or_path: str,
    output_file_path: str,
    dataset_file: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    dataset_split_name: Optional[str] = None,
    id_column_name: str = "id",
    audio_column_name: str = "audio",
    start_column_name: str = "start",
    duration_column_name: str = "duration",
    text_column_name: str = "text",
    # max_label_length: Optional[str] = 128,
    sort_by_length: bool = False,
    torch_dtype: str = "float16",
    device: Union[str, int] = 0,
    attn_type: Optional[str] = "flash_attn",
    language: Optional[str] = None,
    task: str = "transcribe",
    # return_timestamps: bool = False,
    generation_num_beams: Optional[int] = None,
    chunk_length_s: Optional[float] = None,
    stride_length_s: Optional[float] = None,
    per_device_eval_batch_size: int = 8,
    dataloader_num_workers: int = 1,
    num_processing_workers: int = 1,
):
    if attn_type not in [None, "flash_attn", "flash_attn_2"]:
        raise ValueError(
            f"Argument `attn_type` is set to {attn_type}. Should be one of:1. `None`: default Transformers attention"
            " implementation.2. `flash_attn`: Flash Attention through PyTorch SDPA. Requires `torch>=2.0` and `optimum` to be"
            " installed. Recommended for hardware where Flash Attention 2 is not supported, e.g. Turing GPUs, (T4, RTX"
            " 2080).3. `flash_attn_2`: Flash Attention 2 through the Flash Attention package"
            " https://github.com/Dao-AILab/flash-attention. **Always** recommended on supported hardware (Ampere, Ada, or"
            " Hopper GPUs, e.g., A100, RTX 3090, RTX 4090, H100)."
        )

    if torch_dtype == "float16":
        torch_dtype = torch.float16
    elif torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # load dataset
    if dataset_file is not None:
        ext = dataset_file.rsplit(".", 1)[-1]
        dataset = load_dataset(ext, data_files=dataset_file, split="train")
    elif dataset_name is not None:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split=dataset_split_name,
            # token=token,
            # streaming=True,
        )
    else:
        raise ValueError("You have not specified a dataset name nor a custom validation file")

    print(dataset)

    dataset_features = list(dataset.features.keys())

    # Debug
    dataset = dataset.select(range(100))

    # tmp fix
    dataset = dataset.map(
        lambda x: {f"tmp_{audio_column_name}": x[audio_column_name]},
        num_proc=num_processing_workers,
    )

    # sort to accelerate
    dataset = dataset.sort(duration_column_name, reverse=True) if sort_by_length else dataset

    # load processor
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    # set prefix tokens for tokenizer
    # processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    # tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor
    model_sampling_rate = feature_extractor.sampling_rate

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        # use_safetensors=True,
        use_flash_attention_2=attn_type == "flash_attn_2",
    )

    if attn_type == "flash_attn":
        model = model.to_bettertransformer()

    model.eval()

    # force prefix tokens for generation utils
    # not for EN only models
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task=task,  # no_timestamps=not return_timestamps
    )
    # print(f"Model forced_decoder_ids: {model.config.forced_decoder_ids}")
    # todo: include some other tokens dropped by fine-tuning
    # tmp_config = AutoConfig.from_pretrained("openai/whisper-medium")
    # model.config.suppress_tokens = tmp_config.suppress_tokens
    # print(f"Model `suppress_tokens`: {model.config.suppress_tokens}")

    print("Model has been loaded")

    # read segments
    def get_segment(example):
        sr_ = sf.info(example[audio_column_name]).samplerate
        start = int(float(example[start_column_name]) * sr_)
        frames = int(float(example[duration_column_name]) * sr_)

        waveform, _ = get_waveform(
            example[audio_column_name],
            start=start,
            frames=frames,
            mono=True,
            output_sample_rate=model_sampling_rate,
            always_2d=False,
        )

        example[audio_column_name] = {
            "path": example[audio_column_name],
            "array": waveform,
            "sampling_rate": model_sampling_rate,
        }

        return example

    if start_column_name in dataset_features and duration_column_name in dataset_features:
        dataset = dataset.map(get_segment, num_proc=num_processing_workers)

    dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=model_sampling_rate))

    # inference using pipeline (pros: embedded dataloader)
    # Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = generation_num_beams if generation_num_beams is not None else getattr(model.generation_config, "num_beams", 1)

    gen_kwargs = {
        # "max_length": max_label_length,
        "num_beams": num_beams,
        # "language": language,
        # "task": task,  # confict with pipeline type
        # "return_timestamps": return_timestamps,
    }

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        # max_new_tokens=128,  # todo
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        num_workers=dataloader_num_workers,
        batch_size=per_device_eval_batch_size,
        torch_dtype=torch_dtype,
        # device=device,
        # **gen_kwargs,
    )

    start_time = time.perf_counter()

    predictions = []
    for out in tqdm(
        pipe(
            KeyDataset(dataset, audio_column_name),
            # chunk_length_s=chunk_length_s,
            # stride_length_s=stride_length_s,
            # num_workers=dataloader_num_workers,
            # batch_size=per_device_eval_batch_size,
        ),
        total=len(dataset),
    ):
        # Exactly the same output as before, but the content are passed
        # as batches to the model
        # print(out)
        predictions.append(out["text"])

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Inference time: {hours:.0f}h {minutes:.0f}m {seconds:.2f}s")

    def collect_result(example, idx):
        example["prediction"] = predictions[idx]
        # example["target"] = normalize_text(example[text_column_name], invalid_chars_regex)
        return example

    result = dataset.map(collect_result, with_indices=True, num_proc=num_processing_workers)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w") as manifest_f:
        for _, sample in enumerate(tqdm(result, desc="Saving", total=result.num_rows, unit=" samples")):
            # tmp fix
            # sample[audio_column_name] = sample.pop(audio_column_name)["path"]
            sample[audio_column_name] = sample.pop(f"tmp_{audio_column_name}")
            manifest_f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")


if __name__ == "__main__":
    fire.Fire(main)
