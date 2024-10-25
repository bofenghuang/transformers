#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import ctc_segmentation as cs
import fire
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import transformers
from transformers import AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from utils.audio_utils import get_waveform
from data_utils import write_dataset_to_json

SAMPLE_RATE = 16_000


def normalize_text(s):
    s = re.sub(r"\s*'\s*", "'", s)  # standardize when there's a space before/after an apostrophe
    s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space
    return s


class SpeechDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        processor: Any,
        audio_column_name: str = "audio_filepath",
        start_column_name: str = "start",
        duration_column_name: str = "duration",
        text_column_name: str = "text",
        sample_rate: int = SAMPLE_RATE,
        compute_ctc_loss: bool = False,
    ):
        self.processor = processor
        self.audio_column_name = audio_column_name
        self.start_column_name = start_column_name
        self.duration_column_name = duration_column_name
        self.text_column_name = text_column_name
        self.sample_rate = sample_rate
        self.compute_ctc_loss = compute_ctc_loss
        self.dataset = dataset

    def __getitem__(self, n: int) -> Dict[str, Any]:
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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
    model_name_or_path: str,
    output_file_path: str,
    dataset_file: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    dataset_split_name: Optional[str] = None,
    audio_column_name: str = "audio_filepath",
    start_column_name: str = "start",
    duration_column_name: str = "duration",
    text_column_name: str = "text",
    sort_by_length: bool = False,
    torch_dtype: str = "float32",
    attn_implementation: Optional[
        str
    ] = None,  # `eager` or `None` for default, `sdpa` for PyTorch SDPA, `flash_attention_2` for FA2
    device: Union[str, int] = 0,
    greedy: bool = True,
    compute_ctc_loss: bool = False,
    batch_size: int = 8,
    pad_to_multiple_of: int = 8,
    dataloader_num_workers: int = 1,
    score_min_mean_over_l: Optional[int] = None,
    num_processing_workers: int = 16,
    max_samples: Optional[int] = None,
):

    torch_dtype = getattr(torch, torch_dtype)

    # load processor
    processor = (
        Wav2Vec2Processor.from_pretrained(model_name_or_path)
        if greedy
        else Wav2Vec2ProcessorWithLM.from_pretrained(model_name_or_path)
    )
    # decoder = None
    # decoder = processor.decoder

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    model_sampling_rate = feature_extractor.sampling_rate

    # config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCTC.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        # low_cpu_mem_usage=True,  # ? make logits NaN
        # device_map=device,
    )
    model.eval()
    model = model.to(device)

    if compute_ctc_loss:
        model.config.ctc_loss_reduction = "none"
        model.config.ctc_zero_infinity = True

    print("Model has been loaded")

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

    print(dataset)

    # sample
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    # sort by duration to speed up
    if duration_column_name in dataset.features.keys() and sort_by_length:
        dataset = dataset.sort(duration_column_name, reverse=True)

    speech_dataset = SpeechDataset(
        dataset,
        processor=processor,
        audio_column_name=audio_column_name,
        start_column_name=start_column_name,
        duration_column_name=duration_column_name,
        text_column_name=text_column_name,
        sample_rate=model_sampling_rate,
        compute_ctc_loss=compute_ctc_loss,
    )

    data_collator = DataCollatorWithPadding(
        processor=processor,
        pad_to_multiple_of=pad_to_multiple_of,
        compute_ctc_loss=compute_ctc_loss,
    )

    dataloader = DataLoader(
        speech_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )

    start_time = time.perf_counter()

    hypotheses = []
    probabilities = []
    losses = []

    for batch in tqdm(dataloader, desc="Inferring..."):
        batch = batch.to(model.device)
        # todo: move to dataset
        if torch_dtype != torch.float32:
            batch["input_values"] = batch["input_values"].to(dtype=torch_dtype)

        with torch.inference_mode():
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss

            # get output lengths after conv layers
            output_lengths = model._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1)).to(torch.long)

            probs = F.softmax(logits, dim=-1)
            probs = probs.cpu().numpy()
            # Use attention mask to ignore padding
            probs = [prob[:output_length] for prob, output_length in zip(probs, output_lengths)]

            if compute_ctc_loss:
                # average loss by the target lengths
                loss = (loss / batch["labels"].ne(-100).sum(-1)).tolist()

        if greedy:
            predicted_ids = torch.argmax(logits, dim=-1)
            # Use attention mask to ignore padding
            predicted_ids = [ids[:output_length] for ids, output_length in zip(predicted_ids, output_lengths)]
            predicted_sentences = processor.batch_decode(predicted_ids)
        else:
            logits = logits.cpu().numpy()
            # Use attention mask to ignore padding
            logits = [log[:output_length] for log, output_length in zip(logits, output_lengths)]
            predicted_sentences = processor.batch_decode(logits).text

        hypotheses.extend(predicted_sentences)
        probabilities.extend(probs)
        if compute_ctc_loss:
            losses.extend(loss)

    print(f'Inference time: {time.strftime("%Hh%Mm%Ss", time.gmtime(time.perf_counter() - start_time))}')

    # Tokenize transcripts
    vocab = tokenizer.get_vocab()
    # unk_id = vocab["<unk>"]
    unk_id = vocab["[UNK]"]
    char_list = list(vocab.keys())

    sample = dataset[0]
    index_duration = int(sample["duration"] * SAMPLE_RATE) / probabilities[0].shape[0] / SAMPLE_RATE

    config = cs.CtcSegmentationParameters(char_list=char_list)
    # config.char_list = char_list
    # config.min_window_size = window_size
    config.index_duration = index_duration
    # config.index_duration = round(index_duration, 4)
    # config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    # Character probabilities over each L frames are accumulated to calculate the confidence score
    # A lower L makes the score more sensitive to error in the transcription, but also errors in the ASR model
    if score_min_mean_over_l is not None:
        config.score_min_mean_over_L = int(score_min_mean_over_l)
    print(f"ctc_segmentation config: {config}")

    def process_function(example, idx):
        # split by word
        transcripts = example[text_column_name].split()
        probs = probabilities[idx]

        tokens = []
        for transcript in transcripts:
            assert len(transcript) > 0
            tok_ids = tokenizer(transcript, return_tensors="np")["input_ids"]
            # tok_ids = np.array(tok_ids, dtype=np.int)
            tok_ids = np.array(tok_ids, dtype=np.int64)
            tokens.append(tok_ids[tok_ids != unk_id])

        try:
            # convert ground truth text or tokens into a matrix
            # ground_truth_mat, utt_begin_indices = cs.prepare_text(config, transcripts)
            ground_truth_mat, utt_begin_indices = cs.prepare_token_list(config, tokens)
            # computes char-wise alignments from the CTC log posterior probabilites
            timings, char_probs, state_list = cs.ctc_segmentation(config, probs, ground_truth_mat)
            # converts char-wise alignments to utterance-wise alignments
            segments = cs.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcripts)

            example["predicted_words"] = [{"text": t, "start": p[0], "end": p[1], "conf": p[2]} for t, p in zip(transcripts, segments)]
        except:
            example["predicted_words"] = []

        return example

    dataset = dataset.map(process_function, with_indices=True, num_proc=num_processing_workers, load_from_cache_file=False)

    dataset = dataset.add_column("predicted_text", hypotheses)
    if compute_ctc_loss:
        dataset = dataset.add_column("predicted_ctc_loss", losses)

    write_dataset_to_json(dataset, output_file_path)


if __name__ == "__main__":
    fire.Fire(main)
