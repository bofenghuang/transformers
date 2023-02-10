#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang

import datasets
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

SEED = 10


# bh: stat ds duration
def get_duration(ds_, audio_column_name, preprocessing_num_workers):
    def get_duration_(example):
        example["duration"] = example[audio_column_name]["array"].shape[0] / example[audio_column_name]["sampling_rate"]
        return example
    processed_ds_ = ds_.map(get_duration_, remove_columns=ds_.column_names, num_proc=preprocessing_num_workers)
    return sum(processed_ds_["duration"]) / 3600


def normalize_dataset(ds, text_column_name=None):
    if text_column_name is not None and text_column_name != "sentence":
        ds = ds.rename_column(text_column_name, "sentence")
    # resample to specified sampling rate
    # in order to merge ds
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    # normalise columns to ["audio", "sentence"]
    ds = ds.remove_columns(set(ds.features.keys()) - set(["audio", "sentence"]))
    return ds


# todo: interleave
def load_train_datasets(model_args, data_args, training_args):
    ds_mcv = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "fr",
        split="train+validation",
        # use_auth_token=model_args.use_auth_token,
        use_auth_token=True,
    )
    ds_mcv = normalize_dataset(ds_mcv)

    # bh: no punc, no case
    ds_mls = load_dataset(
        # "multilingual_librispeech",
        "facebook/multilingual_librispeech",
        "french",
        split="train+validation",
    )
    ds_mls = normalize_dataset(ds_mls, text_column_name="text")

    ds_voxpopuli = load_dataset(
        # "polinaeterna/voxpopuli",
        "facebook/voxpopuli",
        "fr",
        split="train+validation",
    )
    # todo: raw_text
    ds_voxpopuli = normalize_dataset(ds_voxpopuli, text_column_name="normalized_text")

    ds_fleurs = load_dataset("google/fleurs", "fr_fr", split="train+validation")
    # todo: raw_transcription
    ds_fleurs = normalize_dataset(ds_fleurs, text_column_name="transcription")

    ds_african_accented_fr = load_dataset(
        "gigant/african_accented_french",
        "fr",
        split="train",
    )
    ds_african_accented_fr = normalize_dataset(ds_african_accented_fr)

    data_files = [
        "/projects/bhuang/corpus/speech/multilingual-tedx/fr-fr/output_asr/data_train.csv",
        "/projects/bhuang/corpus/speech/multilingual-tedx/fr-fr/output_asr/data_valid.csv",
    ]
    ds_mtedx = load_dataset("csv", data_files=data_files)["train"]
    ds_mtedx = normalize_dataset(ds_mtedx, text_column_name="normalized_text")

    # bh: no punc, no case
    data_files = ["/projects/bhuang/corpus/speech/media-speech/data.csv"]
    ds_mediaspeech = load_dataset("csv", data_files=data_files)["train"]
    ds_mediaspeech = normalize_dataset(ds_mediaspeech, text_column_name="normalized_text")

    # ? dup
    # data_files = ["/projects/bhuang/corpus/speech/m-ailabs/data.csv"]
    # ds_mailabs = load_dataset("csv", data_files=data_files)["train"]
    # ds_mailabs = normalize_dataset(ds_mailabs, text_column_name="normalized_text")

    all_train_datasets = [
        ds_mcv,
        ds_mls,
        ds_voxpopuli,
        ds_fleurs,
        ds_african_accented_fr,
        ds_mtedx,
        ds_mediaspeech,
    ]
    ds_train = concatenate_datasets(all_train_datasets)
    print(ds_train)

    ds_train = ds_train.filter(
        lambda example: example is not None,
        # num_proc=data_args.preprocessing_num_workers,
        num_proc=4,
        input_columns=["sentence"],
    )
    print(ds_train)

    # NB: shuffle concatenated dataset
    # ds_train = ds_train.shuffle(training_args.seed)
    ds_train = ds_train.shuffle(SEED)

    return ds_train


def load_test_datasets(model_args, data_args, training_args):
    ds_mcv = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "fr",
        split="test",
        # use_auth_token=model_args.use_auth_token,
        use_auth_token=True,
    )
    ds_mcv = normalize_dataset(ds_mcv)

    ds_mls = load_dataset(
        # "multilingual_librispeech",
        "facebook/multilingual_librispeech",
        "french",
        split="test",
    )
    ds_mls = normalize_dataset(ds_mls, text_column_name="text")

    ds_voxpopuli = load_dataset(
        # "polinaeterna/voxpopuli",
        "facebook/voxpopuli",
        "fr",
        split="test",
    )
    ds_voxpopuli = normalize_dataset(ds_voxpopuli, text_column_name="normalized_text")

    ds_fleurs = load_dataset("google/fleurs", "fr_fr", split="train+validation")
    # todo: raw_transcription
    ds_fleurs = normalize_dataset(ds_fleurs, text_column_name="transcription")

    ds_african_accented_fr = load_dataset(
        "gigant/african_accented_french",
        "fr",
        split="test",
    )
    ds_african_accented_fr = normalize_dataset(ds_african_accented_fr)

    data_files = [
        "/projects/bhuang/corpus/speech/multilingual-tedx/fr-fr/output_asr/data_test.csv",
    ]
    ds_mtedx = load_dataset("csv", data_files=data_files)["train"]
    ds_mtedx = normalize_dataset(ds_mtedx, text_column_name="normalized_text")

    all_test_datasets = [
        ds_mcv,
        ds_mls,
        ds_voxpopuli,
        ds_fleurs,
        ds_african_accented_fr,
        ds_mtedx,
    ]
    # all_test_datasets = [ds_mcv]
    ds_test = concatenate_datasets(all_test_datasets)
    print(ds_test)

    ds_test = ds_test.filter(
        lambda example: example is not None,
        # num_proc=data_args.preprocessing_num_workers,
        num_proc=4,
        input_columns=["sentence"],
    )
    print(ds_test)

    return ds_test
