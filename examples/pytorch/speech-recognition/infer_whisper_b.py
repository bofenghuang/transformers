#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Infer whisper models with vanilla torch dataset (not HF one) and HF accelerate (wrapper to easily extend to DDP or DeepSpeed without modifing code)."""

import json
import os
import time
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Dict, List, Optional, Union

import fire
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Audio, Value, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from utils.audio_utils import get_waveform, convert_waveform

SAMPLE_RATE = 16_000


def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for sample in tqdm(ds_iter, desc="Writing to json", total=dataset.num_rows, unit=" samples"):
            # only save serializable types
            sample = {k: v for k, v in sample.items() if isinstance(v, (str, int, float))}
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")


class SpeechDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        processor: Any,
        audio_column_name: str = "audio_filepath",
        start_column_name: str = "start",
        duration_column_name: str = "duration",
        # text_column_name: str = "text",
        sample_rate: int = SAMPLE_RATE,
        # num_processing_workers: int = 1,
    ):
        self.processor = processor
        self.audio_column_name = audio_column_name
        self.start_column_name = start_column_name
        self.duration_column_name = duration_column_name
        # self.text_column_name = text_column_name
        self.sample_rate = sample_rate
        # self.num_processing_workers = num_processing_workers

        self.model_input_name = processor.feature_extractor.model_input_names[0]

        self.dataset = dataset
        # self.dataset = self.preprocess(dataset)
        # print(f"Loaded {len(self.dataset)} examples")

    # def preprocess(self, dataset: Dataset):
        # convert second to frames
        # if self.start_column_name in segments[0] and self.duration_column_name in segments[0]:
        #     new_segments = []
        #     for audio_path, segments_group in groupby(segments, lambda x: x[self.audio_column_name]):
        #         sampling_rate = sf.info(audio_path).samplerate
        #         # segments_group = sorted(segments_group, key=lambda x: float(x[self.start_column_name]))

        #         for segment in segments_group:
        #             segment[self.start_column_name] = int(float(segment[self.start_column_name]) * sampling_rate)
        #             segment[self.duration_column_name] = int(float(segment[self.duration_column_name]) * sampling_rate)
        #             new_segments.append(segment)

        #     segments = new_segments

        # if self.start_column_name in dataset.features and self.duration_column_name in dataset.features:
        #     # sampling_rate_mappings = {
        #     #     audio_path: sf.info(audio_path).samplerate for audio_path in tqdm(list(set(dataset[self.audio_column_name])))
        #     # }

        #     def _process_function(example):
        #         # sampling_rate = sf.info(example[self.audio_column_name]).samplerate
        #         # sampling_rate = sampling_rate_mappings.get(example[self.audio_column_name])
        #         sampling_rate = SAMPLE_RATE

        #         # todo: map can't set float to int
        #         example[self.start_column_name] = int(float(example[self.start_column_name]) * sampling_rate)
        #         example[self.duration_column_name] = int(float(example[self.duration_column_name]) * sampling_rate)
        #         return example

        #     dataset = dataset.map(
        #         _process_function, num_proc=self.num_processing_workers, desc="converting offset and duration"
        #     )
        #     dataset = dataset.cast_column(self.start_column_name, Value("int64"))
        #     dataset = dataset.cast_column(self.duration_column_name, Value("int64"))

        # get audio duration
        # if self.duration_column_name not in dataset.features and (
        #     self.min_duration is not None or self.max_duration is not None or self.sort_by_length
        # ):
        #     if dataset.features[self.audio_column_name].dtype == "dict":
        #         dataset = dataset.map(
        #             lambda x: {self.duration_column_name: x[self.audio_column_name]["array"].shape[0] / self.sample_rate},
        #             num_proc=self.num_processing_workers,
        #             desc="getting duration",
        #         )
        #     elif dataset.features[self.audio_column_name].dtype == "str":
        #         dataset = dataset.map(
        #             lambda x: {self.duration_column_name: sf.info(x[self.audio_column_name]).duration},
        #             num_proc=self.num_processing_workers,
        #             desc="getting duration",
        #         )
        #     else:
        #         raise NotImplementedError(f"Not implemented type: {dataset.features[self.audio_column_name]}")

        # filter by duration
        # if self.min_duration is not None or self.max_duration is not None:

        #     def _filter_function(example):
        #         if self.min_duration is not None and example[self.duration_column_name] < self.min_duration:
        #             return False
        #         if self.max_duration is not None and example[self.duration_column_name] > self.max_duration:
        #             return False
        #         return True

        #     dataset = dataset.filter(_filter_function, num_proc=self.num_processing_workers)

        # # sort by duration
        # if self.sort_by_length:
        #     # segments = sorted(segments, key=lambda x: x[self.duration_column_name], reverse=True)
        #     dataset = dataset.sort(self.duration_column_name, reverse=True)

        # return dataset

    def __getitem__(self, n: int):
        sample = self.dataset[n]

        if isinstance(sample[self.audio_column_name], str):
            start, frames = 0, -1
            if (start := sample.get(self.start_column_name)) is not None and (
                frames := sample.get(self.duration_column_name)
            ) is not None:
                # sampling_rate = sf.info(sample[self.audio_column_name]).samplerate
                sampling_rate = SAMPLE_RATE
                # convert second to frames
                start = int(float(start) * sampling_rate)
                frames = int(float(frames) * sampling_rate)

            # read waveform from audio files
            waveform, _ = get_waveform(
                sample[self.audio_column_name],
                start=start,
                frames=frames,
                mono=True,
                output_sample_rate=self.sample_rate,
                always_2d=False,
            )
        elif isinstance(sample[self.audio_column_name], dict):
            # resample HF datasets Audio sample
            # accept a waveform of C x T
            # waveform, _ = convert_waveform(
            #     sample[self.audio_column_name]["array"],
            #     sample[self.audio_column_name]["sampling_rate"],
            #     to_mono=True,
            #     to_sample_rate=self.sample_rate,
            # )
            # already resampled by HF Audio feature
            waveform = sample[self.audio_column_name]["array"]
        else:
            raise NotImplementedError(f"Not implemented type: {sample[self.audio_column_name]}")

        input_dict = self.processor.feature_extractor(waveform, sampling_rate=self.sample_rate)
        processed_input = {self.model_input_name: input_dict[self.model_input_name][0]}

        return processed_input

    def __len__(self):
        return len(self.dataset)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    # decoder_start_token_id: int
    input_padding: Union[bool, str] = True
    # target_padding: Union[bool, str] = True
    # max_target_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]):
        # split inputs and labels since they have to be of different lengths and need different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        # label_features = [{"input_ids": feature["labels"]} for feature in features]

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            # pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # labels_batch = self.processor.tokenizer.pad(
        #     label_features,
        #     padding=self.target_padding,
        #     # max_length=self.max_target_length,
        #     # pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=self.return_tensors,
        # )

        # replace padding with -100 to ignore correctly when computing the loss
        # labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        # if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
        #     labels = labels[:, 1:]

        # batch["labels"] = labels

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
    # text_column_name: str = "text",
    # max_label_length: Optional[str] = 128,
    # min_duration: Optional[Union[int, float]] = None,
    # max_duration: Optional[Union[int, float]] = None,
    sort_by_length: bool = False,
    torch_dtype: str = "bfloat16",
    attn_implementation: Optional[str] = None,  # `eager` or `None` for default, `sdpa` for PyTorch SDPA, `flash_attention_2` for FA2
    language: Optional[str] = None,
    task: str = "transcribe",
    return_timestamps: bool = False,
    generation_num_beams: Optional[int] = None,
    per_device_eval_batch_size: int = 8,
    # pad_to_multiple_of: int = 8,
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

    accelerator = Accelerator()
    # kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    # accelerator = Accelerator(kwargs_handlers=[kwargs])

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
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        # device_map=device,
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

    accelerator.print("Whisper model has been loaded")

    # load dataset
    if dataset_file is not None:
        ext = dataset_file.rsplit(".", 1)[-1]
        ext = "json" if ext == "jsonl" else ext
        dataset = load_dataset(ext, data_files=dataset_file, split="train")
    elif dataset_name is not None:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split=dataset_split_name,
            # streaming=True,
            token=True,
            trust_remote_code=True,
            # num_proc=num_processing_workers,
        )
        # resample, mono
        dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=model_sampling_rate, mono=True))
    else:
        raise ValueError("You have not specified a dataset name nor a custom local dataset file")

    accelerator.print(dataset)

    # fake ids (used to attribute predicted text)
    fake_id_column = False
    if id_column_name not in dataset.features.keys():
        with accelerator.local_main_process_first():
            # dataset = dataset.map(
            #     lambda _, idx: {id_column_name: f"{idx:09d}"}, with_indices=True, num_proc=num_processing_workers
            # )
            # using int as id will be cast into torch int
            dataset = dataset.add_column(id_column_name, [f"{i:09d}" for i in range(dataset.num_rows)])
            fake_id_column = True
    elif dataset.features[id_column_name].dtype != "str":
        # cast id to string to tokenize
        dataset = dataset.cast_column(id_column_name, Value("string"))

    # sample
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    # sort by duration to speed up
    if duration_column_name in dataset.features.keys() and sort_by_length:
        dataset = dataset.sort(duration_column_name, reverse=True)

    id_dataset = dataset[id_column_name]

    speech_dataset = SpeechDataset(
        dataset,
        processor=processor,
        audio_column_name=audio_column_name,
        start_column_name=start_column_name,
        duration_column_name=duration_column_name,
        # text_column_name=text_column_name,
        sample_rate=model_sampling_rate,
    )
    accelerator.print(f"Loaded {len(dataset)} samples")

    # define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        # decoder_start_token_id=model.config.decoder_start_token_id,  # <|startoftranscript|>
        input_padding="longest",
        # target_padding="longest",
        # max_target_length=max_label_length,
        # pad_to_multiple_of=pad_to_multiple_of,
    )

    # Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = generation_num_beams if generation_num_beams is not None else getattr(model.generation_config, "num_beams", 1)

    gen_kwargs = {
        # "max_length": max_label_length,
        "num_beams": num_beams,
        "return_timestamps": return_timestamps,
    }
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # forcing the language and task tokens helps multilingual models in their generations
        gen_kwargs.update(
            {
                "language": language,
                "task": task,
            }
        )
    # remove any preset forced decoder ids since these are deprecated
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    eval_dataloader = DataLoader(
        speech_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    id_dataloader = DataLoader(
        id_dataset,
        batch_size=per_device_eval_batch_size * accelerator.num_processes,  # NB
        num_workers=dataloader_num_workers,
    )

    # Prepare everything with accelerate
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    start_time = time.perf_counter()

    eval_pred_ids = []
    eval_preds = []
    # eval_labels = []
    eval_ids = []

    total_steps = int(len(speech_dataset) / per_device_eval_batch_size / accelerator.num_processes)
    batches = tqdm(eval_dataloader, total=total_steps, desc="Inferring...", disable=not accelerator.is_local_main_process)

    for step, (batch, utt_ids) in enumerate(zip(batches, id_dataloader)):
        # Generate predictions and pad to max generated length
        generate_fn = model.module.generate if accelerator.num_processes > 1 else model.generate
        generated_ids = generate_fn(batch[model_input_name].to(dtype=torch_dtype), **gen_kwargs)
        generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id)
        # Gather all predictions and targets
        # NB: all tensors should have the same size at this point
        generated_ids = accelerator.gather_for_metrics((generated_ids))
        # eval_pred_ids.extend(generated_ids.cpu().numpy())
        eval_preds.extend(tokenizer.batch_decode(generated_ids.cpu().numpy(), skip_special_tokens=True, decode_with_timestamps=return_timestamps))
        eval_ids.extend(utt_ids)

        # decoding_steps = 20
        # if step % decoding_steps == 0 and step > 0:
        #     accelerator.wait_for_everyone()
        #     eval_preds.extend(tokenizer.batch_decode(eval_pred_ids, skip_special_tokens=True, decode_with_timestamps=return_timestamps))
        #     eval_pred_ids = []
        # #     jsonl_dump(data, output_file_path, mode="a")

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

        example["prediction"] = id_pred_mappings[example[id_column_name]]
        # pred_ids = id_pred_mappings[example[id_column_name]]
        # example["prediction"] = tokenizer.decode(pred_ids, skip_special_tokens=True)
        # example["prediction"] = tokenizer.decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=return_timestamps)

        # we do not want to group tokens when computing the metrics
        # example["label_str"] = tokenizer.batch_decode(example["label_ids"], skip_special_tokens=True)

        return example

    with accelerator.local_main_process_first():
        dataset = dataset.map(process_function, num_proc=num_processing_workers, desc="mapping transcriptions")
        # dataset = dataset.remove_columns(set(dataset.column_names) - set([id_column_name, text_column_name, "prediction"]))

        if fake_id_column:
            dataset = dataset.remove_columns(id_column_name)

    accelerator.print(dataset)

    if accelerator.is_local_main_process:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        write_dataset_to_json(dataset, output_file_path)

    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire(main)
