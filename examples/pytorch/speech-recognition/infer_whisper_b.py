#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang


import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import soundfile as sf

import fire
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


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
    max_label_length: Optional[str] = 128,
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
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif torch_dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32

    accelerator = Accelerator(mixed_precision=mixed_precision)

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

    if id_column_name not in dataset.features:
        dataset = dataset.map(
            lambda _, idx: {id_column_name: f"{idx:09d}"}, with_indices=True, num_proc=num_processing_workers
        )

    # Debug
    # dataset = dataset.select(range(20))

    # sort to accelerate
    dataset = dataset.sort(duration_column_name, reverse=True) if sort_by_length else dataset

    # convert to df
    result = copy.deepcopy(dataset)

    # dataset = dataset.to_iterable_dataset(num_shards=data_args.num_shards)  # shard the dataset
    dataset = dataset.to_iterable_dataset()

    # load processor
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    # set prefix tokens for tokenizer
    # processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor
    model_sampling_rate = feature_extractor.sampling_rate

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
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

    # read segments
    def get_segment(example):
        sr_ = sf.info(sample[audio_column_name]).samplerate
        start = int(float(sample[start_column_name]) * sr_)
        frames = int(float(sample[duration_column_name]) * sr_)

        waveform, _ = get_waveform(
            sample[audio_column_name],
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

    if start_column_name in dataset.features and duration_column_name in dataset.features:
        dataset = dataset.map(get_segment)

    dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=model_sampling_rate))

    max_label_length = max_label_length if max_label_length is not None else model.config.max_length
    model_input_name = feature_extractor.model_input_names[0]

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]

        # process targets
        # batch["labels"] = tokenizer(batch[text_column_name]).input_ids
        # batch["labels"] = tokenizer(input_str, max_length=max_label_length, truncation=True).input_ids

        # record the id of the sample as token ids
        batch["utt_id"] = tokenizer(batch[id_column_name], add_special_tokens=False).input_ids

        return batch

    dataset_features = list(dataset.features.keys())
    vectorized_dataset = dataset.map(prepare_dataset, remove_columns=dataset_features)

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

    # Prepare everything with accelerate
    model = accelerator.prepare(model)

    eval_preds = []
    # eval_labels = []
    eval_ids = []
    start_time = time.perf_counter()

    eval_loader = DataLoader(
        vectorized_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )

    eval_loader = accelerator.prepare(eval_loader)
    total_steps = int(result.num_rows / per_device_eval_batch_size / accelerator.num_processes)
    batches = tqdm(eval_loader, total=total_steps, desc="Evaluating...", disable=not accelerator.is_local_main_process)

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
    print(f'Inference time: {time.strftime("%Hh%Mm%Ss", time.gmtime(time.perf_counter() - start_time))}')

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

    result = result.map(process_function, num_proc=num_processing_workers)
    result = result.remove_columns(set(result.column_names) - set([id_column_name, text_column_name, "prediction"]))
    print(result)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as manifest_f:
        for sample in tqdm(result, desc="Saving", total=result.num_rows, unit=" samples"):
            manifest_f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")

    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire(main)
