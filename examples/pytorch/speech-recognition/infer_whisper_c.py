#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Infer whisper models with HF streaming dataset and HF accelerate for distributed settings (not working together yet)."""

import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import fire
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from utils.audio_utils import get_waveform, convert_waveform


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
    streaming: bool = False,
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
    processing_num_workers: int = 1,
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

    # load dataset
    if dataset_file is not None:
        ext = dataset_file.rsplit(".", 1)[-1]
        ext = "json" if ext == "jsonl" else ext
        dataset = load_dataset(ext, data_files=dataset_file, split="train", streaming=streaming)
    elif dataset_name is not None:
        dataset = load_dataset(
            path=dataset_name,
            name=dataset_config_name,
            split=dataset_split_name,
            streaming=streaming,
            token=True,
            trust_remote_code=True,
            num_proc=processing_num_workers,
        )
    else:
        raise ValueError("You have not specified a dataset name nor a custom validation file")

    dataset_features = list(dataset.features.keys())

    if id_column_name not in dataset_features:
        with accelerator.local_main_process_first():
            dataset = dataset.map(
                lambda _, idx: {id_column_name: f"{idx:09d}"}, with_indices=True, num_proc=processing_num_workers
            )

    # Debug
    # dataset = dataset.select(range(100))

    result = copy.deepcopy(dataset)

    # sort to accelerate
    dataset = dataset.sort(duration_column_name, reverse=True) if sort_by_length else dataset

    # dataset = dataset.to_iterable_dataset(num_shards=data_args.num_shards)  # shard the dataset
    dataset = dataset.to_iterable_dataset()

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
        dataset = dataset.map(get_segment)
    else:
        dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=model_sampling_rate))

    # max_label_length = max_label_length if max_label_length is not None else model.config.max_length
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

    eval_preds = []
    # eval_labels = []
    eval_ids = []
    start_time = time.perf_counter()

    eval_dataloader = DataLoader(
        vectorized_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )

    # Prepare everything with accelerate
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    total_steps = int(result.num_rows / per_device_eval_batch_size / accelerator.num_processes)
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
        result = result.map(process_function, num_proc=processing_num_workers)
        # result = result.remove_columns(set(result.column_names) - set([id_column_name, text_column_name, "prediction"]))

    accelerator.print(result)

    if accelerator.is_local_main_process:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open(output_file_path, "w") as manifest_f:
            for sample in tqdm(result, desc="Saving", total=result.num_rows, unit=" samples"):
                manifest_f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")

    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire(main)
