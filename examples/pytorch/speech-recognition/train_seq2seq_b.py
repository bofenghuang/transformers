# /usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import torch
import os
import json
import fire
import time
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Dict, Any, Union
import deepspeed
from dataclasses import dataclass, field
from transformers import set_seed
from contextlib import contextmanager

import datasets
import torch
from datasets import DatasetDict, IterableDatasetDict, interleave_datasets, load_dataset
from torch.utils.data import IterableDataset
import torch.distributed as dist

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)


def get_world_size():
    # Worker RANK and WORLD_SIZE are assigned automatically
    return int(os.getenv("WORLD_SIZE", "1"))


def get_rank():
    # return int(os.getenv('RANK', '0'))
    return dist.get_rank()


def get_local_rank():
    return get_rank() % torch.cuda.device_count()


def is_main_process():
    """Whether or not the current process is the local main process."""
    return get_local_rank() <= 0


@contextmanager
def main_process_first():
    # todo
    if get_world_size() > 1:
        global_rank = get_rank()
        if global_rank != 0:  # other ranks wait first
            dist.barrier()
        yield
        if global_rank == 0:  # then rank 0 waits after it has run the context
            dist.barrier()
    else:
        yield


def print_rank_0(msg: str):
    if dist.get_rank() <= 0:
        print(msg)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        # "input_features"
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            # bh: pass attention_mask for specaugment of whisper models
            # print([feature["attention_mask"] for feature in features])
            # batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])
            batch["attention_mask"] = torch.stack([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


@dataclass
class TrainingArguments:
    ...


def main(
    model_name_or_path: str,
    train_file: str,
    validation_file: str,
    forced_decoder_ids: List[List[int]] = None,
    suppress_tokens: List[int] = None,
    language: str = None,
    task: str = "transcribe",
    use_fast_tokenizer: bool = True,
    audio_column_name: str = "audio",
    text_column_name: str = "text",
    num_shards: int = 1,
    shuffle_buffer_size: int = 500,
    local_rank: int = -1,
    seed: int = 42,
):

    # https://pytorch.org/docs/stable/elastic/run.html
    if local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
        # todo
        # deepspeed.init_distributed(dist_backend="nccl")

    global_rank = dist.get_rank()

    # load config

    # seed
    set_seed(seed)

    # load data
    raw_datasets = IterableDatasetDict()
    ext = train_file.rsplit(".", 1)[-1]
    raw_datasets["train"] = load_dataset(ext, data_files=train_file, split="train")
    # faster than loading with streaming=True
    raw_datasets["train"] = raw_datasets["train"].to_iterable_dataset(num_shards=num_shards)  # shard the dataset
    ext = validation_file.rsplit(".", 1)[-1]
    raw_datasets["eval"] = load_dataset(ext, data_files=validation_file, split="train")
    raw_datasets["eval"] = raw_datasets["eval"].to_iterable_dataset(num_shards=num_shards)

    raw_datasets = raw_datasets.cast_column(audio_column_name, datasets.features.Audio(sampling_rate=16_000))
    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())

    # 5. Load pretrained model, tokenizer, and feature extractor
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(model_name_or_path)

    config.update(
        {
            "forced_decoder_ids": forced_decoder_ids,
            "suppress_tokens": suppress_tokens,
            # "use_cache": use_cache,
            # "encoder_layerdrop": encoder_layerdrop,
            # "decoder_layerdrop": decoder_layerdrop,
            # "dropout": dropout,
            # "attention_dropout": attention_dropout,
            # "activation_dropout": activation_dropout,
        }
    )

    # SpecAugment for whisper models
    # bh: SpecAugment parameters
    # if getattr(config, "model_type", None) == "whisper":
    #     config.update(
    #         {
    #             "apply_spec_augment": apply_spec_augment,
    #             "mask_time_prob": mask_time_prob,
    #             "mask_time_length": mask_time_length,
    #             "mask_time_min_masks": mask_time_min_masks,
    #             "mask_feature_prob": mask_feature_prob,
    #             "mask_feature_length": mask_feature_length,
    #             "mask_feature_min_masks": mask_feature_min_masks,
    #         }
    #     )

    # load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer,)
    # load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, config=config,)

    if language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=language, task=task)

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    model_input_name = feature_extractor.model_input_names[0]
    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=forward_attention_mask
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = batch[text_column_name]
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    with main_process_first():
        vectorized_datasets = raw_datasets.map(prepare_dataset, remove_columns=raw_datasets_features,).with_format(
            "torch"
        )

        # manually shuffle if streaming (done by the trainer for non-streaming)
        vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
            buffer_size=shuffle_buffer_size, seed=seed,
        )

    # filter training data that is shorter than min_input_length or longer than
    # max_input_length
    # def is_audio_in_length_range(length):
    #     return min_input_length < length < max_input_length

    # if do_train:
    #     vectorized_datasets["train"] = vectorized_datasets["train"].filter(
    #         is_audio_in_length_range,
    #         input_columns=["input_length"],
    #     )

    # max_label_length = model.config.max_target_positions

    # def is_labels_in_length_range(labels):
    #     return len(labels) < max_label_length

    # if do_train:
    #     # todo: do in another preprocess script
    #     vectorized_datasets = vectorized_datasets.filter(
    #         is_labels_in_length_range,
    #         input_columns=["labels"],
    #     )

    # load optimizer

    # load scheduler

    # DataLoaders creation:
    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=default_data_collator, sampler=train_sampler, batch_size=per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, sampler=eval_sampler, batch_size=per_device_eval_batch_size,
    )

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return losses, perplexity

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay, lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # train
    # todo resume
    print_rank_0("***** Running training *****", global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{num_train_epochs} *****", global_rank,
    )
    perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", global_rank)

    for epoch in range(num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            global_rank,
        )
        model.train()

        for step, batch in enumerate(train_dataloader):
            start = time.time()
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            if print_loss:
                print(f"Epoch: {epoch}, Step: {step}, Rank: {dist.get_rank()}, loss = {loss}")
            model.backward(loss)
            model.step()
            end = time.time()
            if dist.get_rank() == 0:
                print_throughput(model.model, args, end - start, global_rank)

            if step == training_debug_steps:
                break

        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch+1}/{num_train_epochs} *****", global_rank,
        )
        eval_losses, perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"eval_loss: {eval_losses}", global_rank)
        print_rank_0(f"ppl: {perplexity}", global_rank)
        model.tput_timer.update_epoch_count()

    if output_dir is not None:
        print_rank_0("saving the final model ...", global_rank)
        model = convert_lora_to_linear_layer(model)

        if global_rank == 0:
            save_hf_format(model, tokenizer, args)
    # save


if __name__ == "__main__":
    fire.Fire(main)
