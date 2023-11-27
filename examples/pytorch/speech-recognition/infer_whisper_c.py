#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Infer whisper models with vanilla torch dataset (not HF one) and HF accelerate (wrapper to easily extend to DDP or DeepSpeed without modifing code)."""

import copy
import json
import os
import time
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Dict, List, Optional, Union

import fire
import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

SAMPLE_RATE = 16_000

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


class SpeechDataset(Dataset):
    def __init__(
        self,
        segments: Any,
        processor: Any,
        id_column_name: str = "id",
        audio_column_name: str = "audio_filepath",
        start_column_name: str = "start",
        duration_column_name: str = "duration",
        text_column_name: str = "text",
        sample_rate: int = SAMPLE_RATE,
        sort_by_length: bool = False,
    ):
        self.processor = processor
        self.id_column_name = id_column_name
        self.audio_column_name = audio_column_name
        self.start_column_name = start_column_name
        self.duration_column_name = duration_column_name
        self.text_column_name = text_column_name
        self.sample_rate = sample_rate
        self.sort_by_length = sort_by_length

        self.model_input_name = processor.feature_extractor.model_input_names[0]

        self.dataset = self.preprocess(segments)
        # print(f"Loaded {len(self.dataset)} examples")

    def preprocess(self, segments: List[Dict]):
        # convert second to frames
        if self.start_column_name in segments[0] and self.duration_column_name in segments[0]:
            new_segments = []
            for audio_path, segments_group in groupby(segments, lambda x: x[self.audio_column_name]):
                sampling_rate = sf.info(audio_path).samplerate
                # segments_group = sorted(segments_group, key=lambda x: float(x[self.start_column_name]))

                for segment in segments_group:
                    segment[self.start_column_name] = int(float(segment[self.start_column_name]) * sampling_rate)
                    segment[self.duration_column_name] = int(float(segment[self.duration_column_name]) * sampling_rate)
                    new_segments.append(segment)

            segments = new_segments

        if self.sort_by_length:
            if self.duration_column_name not in segments[0]:
                raise ValueError(
                    f"Cannot sort utterances because duration_column_name {self.duration_column_name} doesn't exist"
                )
            # todo: compute duration
            segments = sorted(segments, key=lambda x: x[self.duration_column_name], reverse=True)

        return segments

    def __getitem__(self, n: int):
        sample = self.dataset[n]

        if isinstance(sample[self.audio_column_name], str):
            start = sample.get(self.start_column_name, 0)
            frames = sample.get(self.duration_column_name, -1)

            # read waveform from audio files
            waveform, _ = get_waveform(
                sample[self.audio_column_name],
                start=start,
                frames=frames,
                mono=True,
                output_sample_rate=self.sample_rate,
                always_2d=False,
            )
        else:
            # resample HF datasets Audio sample
            waveform, _ = convert_waveform(
                sample[self.audio_column_name]["array"],
                sample[self.audio_column_name]["sampling_rate"],
                to_mono=True,
                to_sample_rate=self.sample_rate,
            )

        input_dict = self.processor.feature_extractor(waveform, sampling_rate=self.sample_rate)
        processed_input = {self.model_input_name: input_dict[self.model_input_name][0]}

        # keep id
        processed_input["utt_id"] = self.processor.tokenizer(sample[self.id_column_name], add_special_tokens=False).input_ids

        return processed_input

    def __len__(self):
        return len(self.dataset)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    input_padding: Union[bool, str] = True
    target_padding: Union[bool, str] = True
    # max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]):
        # split inputs and labels since they have to be of different lengths and need different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        # label_features = [{"input_ids": feature["labels"]} for feature in features]
        id_features = [{"input_ids": feature["utt_id"]} for feature in features]

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # labels_batch = self.processor.tokenizer.pad(
        #     label_features,
        #     padding=self.target_padding,
        #     # max_length=self.max_target_length,
        #     # pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=self.return_tensors,
        # )

        ids_batch = self.processor.tokenizer.pad(
            id_features,
            padding=self.target_padding,
            # max_length=self.max_target_length,
            # pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # replace padding with -100 to ignore correctly when computing the loss
        # labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        # if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
        #     labels = labels[:, 1:]

        # batch["labels"] = labels
        batch["utt_ids"] = ids_batch["input_ids"]

        return batch


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
    attn_type: Optional[str] = "flash_attn",
    language: Optional[str] = None,
    task: str = "transcribe",
    # return_timestamps: bool = False,
    generation_num_beams: Optional[int] = None,
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

    accelerator = Accelerator()

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

    if id_column_name not in dataset.features.keys():
        with accelerator.local_main_process_first():
            dataset = dataset.map(lambda _, idx: {id_column_name: f"{idx:09d}"}, with_indices=True, num_proc=num_processing_workers)

    # Debug
    # dataset = dataset.select(range(100))

    accelerator.print(dataset)

    # load processor
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    # set prefix tokens for tokenizer
    # processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor
    model_sampling_rate = feature_extractor.sampling_rate
    model_input_name = feature_extractor.model_input_names[0]

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        # device_map=device,
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

    accelerator.print("Whisper model has been loaded")

    speech_dataset = SpeechDataset(
        dataset,
        processor=processor,
        id_column_name=id_column_name,
        audio_column_name=audio_column_name,
        start_column_name=start_column_name,
        duration_column_name=duration_column_name,
        text_column_name=text_column_name,
        sample_rate=model_sampling_rate,
        sort_by_length=sort_by_length,
    )

    # define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,  # <|startoftranscript|>
        input_padding="longest",
        target_padding="longest",
        # max_target_length=max_label_length,
        pad_to_multiple_of=8,
    )

    # Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = generation_num_beams if generation_num_beams is not None else getattr(model.generation_config, "num_beams", 1)

    gen_kwargs = {
        # "max_length": max_label_length,
        "num_beams": num_beams,
        "language": language,
        "task": task,
        # "return_timestamps": return_timestamps,
    }

    eval_preds = []
    # eval_labels = []
    eval_ids = []
    start_time = time.perf_counter()

    eval_dataloader = DataLoader(
        speech_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )

    # Prepare everything with accelerate
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    total_steps = int(len(speech_dataset) / per_device_eval_batch_size / accelerator.num_processes)
    batches = tqdm(eval_dataloader, total=total_steps, desc="Inferring...", disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(batches):
        utt_ids = batch.pop("utt_ids")
        # Generate predictions and pad to max generated length
        generate_fn = model.module.generate if accelerator.num_processes > 1 else model.generate
        generated_ids = generate_fn(batch[model_input_name].to(dtype=torch_dtype), **gen_kwargs)
        generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id)
        # Gather all predictions and targets
        utt_ids, generated_ids = accelerator.gather_for_metrics((utt_ids, generated_ids))
        # utt_ids, generated_ids, labels = accelerator.gather_for_metrics((utt_ids, generated_ids, batch["labels"]))
        eval_preds.extend(generated_ids.cpu().numpy())
        # eval_labels.extend(labels.cpu().numpy())
        eval_ids.extend(tokenizer.batch_decode(utt_ids, skip_special_tokens=True))

        # if step % logging_steps == 0 and step > 0:
        #     # batches.write(f"Saving transcriptions for split {split} step {step}")
        #     accelerator.wait_for_everyone()
        #     eval_preds = tokenizer.batch_decode(eval_preds, skip_special_tokens=True, decode_with_timestamps=return_timestamps)

        #     jsonl_dump(data, output_file_path, mode="a")

    accelerator.wait_for_everyone()
    # accelerator.print(f'Inference time: {time.strftime("%dd%Hh%Mm%Ss", time.gmtime(time.perf_counter() - start_time))}')

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    accelerator.print(f"Inference time: {hours:.0f}h {minutes:.0f}m {seconds:.2f}s")

    accelerator.free_memory()
    del model

    id_pred_mappings = dict(zip(eval_ids, eval_preds))

    def process_function(example):
        # replace padded labels by the padding token
        # for label_ids_ in batch["label_ids"]:
        #     label_ids_[label_ids_ == -100] = tokenizer.pad_token_id

        pred_ids = id_pred_mappings[example[id_column_name]]
        example["prediction"] = tokenizer.decode(pred_ids, skip_special_tokens=True)
        # example["prediction"] = tokenizer.decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=return_timestamps)
        # we do not want to group tokens when computing the metrics
        # example["label_str"] = tokenizer.batch_decode(example["label_ids"], skip_special_tokens=True)

        return example

    with accelerator.local_main_process_first():
        dataset = dataset.map(process_function, num_proc=num_processing_workers)
        # dataset = dataset.remove_columns(set(dataset.column_names) - set([id_column_name, text_column_name, "prediction"]))

    accelerator.print(dataset)

    if accelerator.is_local_main_process:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open(output_file_path, "w") as manifest_f:
            for sample in tqdm(dataset, desc="Saving", total=dataset.num_rows, unit=" samples"):
                manifest_f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")

    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire(main)
