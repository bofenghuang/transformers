#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang

"""Infer whisper models with HF pipeline (with built-in dataloader and sliding windows to infer >30s audio)."""

import json
import os
import time
from typing import Optional, Union

import soundfile as sf
import torch
import numpy as np
from datasets import Audio, load_dataset
from tqdm import tqdm
import fire

from transformers import AutoModelForSpeechSeq2Seq, AutoModelForCausalLM, AutoProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from utils.audio_utils import get_waveform


def main(
    model_name_or_path: str,
    output_file_path: str,
    dataset_file: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    dataset_split_name: Optional[str] = None,
    # id_column_name: str = "id",
    audio_column_name: str = "audio",
    start_column_name: str = "start",
    duration_column_name: str = "duration",
    # text_column_name: str = "text",
    # max_label_length: Optional[str] = 128,
    sort_by_length: bool = False,
    device: Union[str, int] = 0,
    torch_dtype: str = "bfloat16",
    attn_implementation: Optional[str] = None,  # `eager` or `None` for default, `sdpa` for PyTorch SDPA, `flash_attention_2` for FA2
    language: Optional[str] = None,
    task: str = "transcribe",
    return_timestamps: bool = False,
    generation_num_beams: Optional[int] = None,
    assistant_model_name_or_path: Optional[str] = None,
    chunk_length_s: Optional[float] = None,
    stride_length_s: Optional[float] = None,
    per_device_eval_batch_size: int = 8,
    dataloader_num_workers: int = 1,
    num_processing_workers: int = 1,
    max_samples: Optional[int] = None,
):
    # print(locals())
    # quit()

    if torch_dtype == "float16":
        torch_dtype = torch.float16
    elif torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

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
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        device_map=device,
        # use_safetensors=True,
    )
    model.eval()

    # force prefix tokens for generation utils
    # not for EN only
    # tokenizer.set_prefix_tokens(language=language, task=task, predict_timestamps=return_timestamps)
    # model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    #     language=language,
    #     task=task,
    #     no_timestamps=not return_timestamps,
    # )
    # print(f"Model forced_decoder_ids: {model.config.forced_decoder_ids}")
    # todo: include some other tokens dropped by fine-tuning
    # tmp_config = AutoConfig.from_pretrained("openai/whisper-medium")
    # model.config.suppress_tokens = tmp_config.suppress_tokens
    # print(f"Model `suppress_tokens`: {model.config.suppress_tokens}")

    assistant_model = None
    if assistant_model_name_or_path is not None:
        assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map=device,
            # use_safetensors=True,
        )
        assistant_model.eval()

    print("Whisper model has been loaded")

    # load dataset
    if dataset_file is not None:
        ext = dataset_file.rsplit(".", 1)[-1]
        ext = "json" if ext == "jsonl" else ext
        dataset = load_dataset(ext, data_files=dataset_file, split="train")
    elif dataset_name is not None:
        dataset = load_dataset(
            path=dataset_name,
            name=dataset_config_name,
            split=dataset_split_name,
            # streaming=True,
            token=True,
            trust_remote_code=True,
            # num_proc=num_processing_workers,
        )
    else:
        raise ValueError("You have not specified a dataset name nor a custom local dataset file")

    print(dataset)

    dataset_features = list(dataset.features.keys())

    # tmp fix
    dataset = dataset.map(
        lambda x: {f"tmp_{audio_column_name}": x[audio_column_name]},
        num_proc=num_processing_workers,
    )

    # sample
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    # sort by duration to speed up
    if duration_column_name in dataset_features and sort_by_length:
        dataset = dataset.sort(duration_column_name, reverse=True)

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
        dataset = dataset.map(get_segment, num_proc=num_processing_workers, desc="reading segment...")

    # resample, mono
    dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=model_sampling_rate))

    # bh: inference using pipeline (pros: embedded dataloader)
    # Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = generation_num_beams if generation_num_beams is not None else getattr(model.generation_config, "num_beams", 1)

    generate_kwargs = {
        # "max_length": max_label_length,
        "num_beams": num_beams,
        "return_timestamps": return_timestamps,
        "language": language,
        "task": task,  # todo: confict with pipeline type
    }

    if assistant_model is not None:
        generate_kwargs["assistant_model"] = assistant_model

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        torch_dtype=torch_dtype,
        # max_new_tokens=128,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        batch_size=per_device_eval_batch_size,
        num_workers=dataloader_num_workers,
        generate_kwargs=generate_kwargs,
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

    del pipe
    del model

    def collect_result(example, idx):
        example["prediction"] = predictions[idx]
        # example["target"] = normalize_text(example[text_column_name], invalid_chars_regex)
        return example

    dataset = dataset.map(collect_result, with_indices=True, num_proc=num_processing_workers, desc="mapping transcriptions...")

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as fo:
        for sample in tqdm(dataset, desc="Writing to json", unit=" samples"):
            # tmp fix
            # sample[audio_column_name] = sample.pop(audio_column_name)["path"]
            sample[audio_column_name] = sample.pop(f"tmp_{audio_column_name}")
            if not isinstance(sample[audio_column_name], str):
                del sample[audio_column_name]
            fo.write(f"{json.dumps(sample, default=str, ensure_ascii=False)}\n")


if __name__ == "__main__":
    fire.Fire(main)
