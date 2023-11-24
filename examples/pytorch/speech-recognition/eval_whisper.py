#!/usr/bin/env python
# Copyright 2022  Bofeng Huang

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import json
import time
import torch
import torchaudio
import torchaudio.sox_effects as ta_sox
from datasets import Audio, Dataset, load_dataset
from tqdm import tqdm

import evaluate
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

# NB
# from utils.normalize_french import FrenchTextNormalizer
# from utils.normalize_french_zaion import FrenchTextNormalizer


# text_normalizer = FrenchTextNormalizer()

# sys.path.append("/home/bhuang/my-scripts")
# from myscripts.data.text.wer.compute_wer import compute_wer


def log_results(result: Dataset, args: Dict[str, str]):
    """DO NOT CHANGE. This function computes and logs the result metrics."""

    log_outputs = args.log_outputs

    def _log_results(result: Dataset, target_file, pred_file):
        # log all results in text file. Possibly interesting for analysis
        with open(pred_file, "w") as p, open(target_file, "w") as t:
            # mapping function to write output
            # def write_to_file(batch, i):
            #     p.write(f"{i}" + "\n")
            #     p.write(batch["prediction"] + "\n")
            #     t.write(f"{i}" + "\n")
            #     t.write(batch["target"] + "\n")

            # ! to adjust for your cases
            def write_to_file(batch, i):
                # p.write(batch[args.id_column_name] + "\t" + str(batch[args.start_column_name]) + "\t" + str(batch[args.end_column_name]) + "\t" + batch["prediction"] + "\n")
                # t.write(batch[args.id_column_name] + "\t" + str(batch[args.start_column_name]) + "\t" + str(batch[args.end_column_name]) + "\t" + batch["target"] + "\n")
                p.write(batch[args.id_column_name] + "\t" + batch["prediction"] + "\n")
                t.write(batch[args.id_column_name] + "\t" + batch["target"] + "\n")

            result.map(write_to_file, with_indices=True)

    if log_outputs is not None:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        _log_results(
            result, target_file=f"{args.outdir}/log_targets_raw.txt", pred_file=f"{args.outdir}/log_predictions_raw.txt"
        )

    def map_norm(example):
        def norm_func(s):
            # NB
            return text_normalizer(
                s, do_lowercase=True, do_ignore_words=True, symbols_to_keep="'-", do_standardize_numbers=True
            )

        example["target"] = norm_func(example["target"])
        example["prediction"] = norm_func(example["prediction"])
        return example

    result = result.map(map_norm, num_proc=args.preprocessing_num_workers, desc="normalize targets and predictions")

    # filtering out empty targets
    result = result.filter(lambda example: example["target"] != "")

    # load metric
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # compute metrics
    wer_result = wer_metric.compute(references=result["target"], predictions=result["prediction"])
    cer_result = cer_metric.compute(references=result["target"], predictions=result["prediction"])

    # print & log results
    result_str = f"WER: {wer_result}\nCER: {cer_result}"
    print(result_str)

    # log all results in text file. Possibly interesting for analysis
    if log_outputs is not None:
        # with open(f"{args.outdir}/{dataset_id}_eval_results.txt", "w") as f:
        with open(f"{args.outdir}/eval_results.txt", "w") as f:
            f.write(result_str)

        _log_results(result, target_file=f"{args.outdir}/log_targets.txt", pred_file=f"{args.outdir}/log_predictions.txt")


def eval_results(result: Dataset, args: Dict[str, str], do_ignore_words=False):
    df = result.to_pandas()

    def norm_func(s):
        # NB
        return text_normalizer(
            s, do_lowercase=True, do_ignore_words=do_ignore_words, symbols_to_keep="'-", do_standardize_numbers=True
        )

    df["target"] = df["target"].map(norm_func)
    df["prediction"] = df["prediction"].map(norm_func)

    # filtering out empty targets
    df = df[df["target"] != ""]

    df["target"] = df["target"].str.split()
    targets = df.set_index(args.id_column_name)["target"].to_dict()

    df["prediction"] = df["prediction"].str.split()
    predictions = df.set_index(args.id_column_name)["prediction"].to_dict()

    out_dir_ = f"{args.outdir}/wer_summary_without_fillers" if do_ignore_words else f"{args.outdir}/wer_summary"
    compute_wer(targets, predictions, out_dir_, do_print_top_wer=True, do_catastrophic=True)


def main(args):
    # load dataset
    if args.input_file_path is not None:
        ext = args.input_file_path.rsplit(".", 1)[-1]
        dataset = load_dataset(ext, data_files=args.input_file_path, split="train")
    elif args.dataset is not None:
        dataset = load_dataset(args.dataset, args.config, split=args.split, use_auth_token=True)
    else:
        raise ValueError("You have not specified a dataset name nor a custom validation file")

    print(dataset)

    # Debug
    # dataset = dataset.select(range(1000))

    device = "cuda:0" if torch.cuda.is_available() and args.fp16 else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # load processor
    # set prefix tokens for tokenizer
    processor = AutoProcessor.from_pretrained(args.model_id, language=args.language, task=args.task)
    model_sampling_rate = processor.feature_extractor.sampling_rate

    # tmp fix
    dataset = dataset.map(
        lambda x: {f"tmp_{args.audio_column_name}": x[args.audio_column_name]}, num_proc=args.preprocessing_num_workers, keep_in_memory=True
    )

    # resample audio
    dataset = dataset.cast_column(args.audio_column_name, Audio(sampling_rate=model_sampling_rate))

    # whisper doesn't take segments longer than 30s
    # todo: max 448 tokens
    max_input_length = 30 * model_sampling_rate
    # min_input_length = 1 * model_sampling_rate

    def is_audio_in_length_range(example):
        length = example[args.audio_column_name]["array"].shape[0]
        # return length > min_input_length and length < max_input_length
        return length < max_input_length

    # dataset = dataset.filter(is_audio_in_length_range, num_proc=args.preprocessing_num_workers, keep_in_memory=True)
    # print(f"Filtered to {dataset.num_rows:,d} utterances")

    # if args.device is None:
    #     args.device = 0 if torch.cuda.is_available() else -1

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        device_map=device,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        # use_safetensors=True,
        # use_flash_attention_2=True,
    )
    model.eval()
    # model = model.to(args.device)
    print("Model has been loaded")

    # Activate Torch Scale-Product-Attention (SDPA)
    # Nice to have when GPU doesn't support Flash-Attention
    model = model.to_bettertransformer()

    # force prefix tokens for generation utils
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    print(f"Model forced_decoder_ids: {model.config.forced_decoder_ids}")
    # todo: include some other tokens dropped by fine-tuning
    # tmp_config = AutoConfig.from_pretrained("openai/whisper-medium")
    # model.config.suppress_tokens = tmp_config.suppress_tokens
    # print(f"Model `suppress_tokens`: {model.config.suppress_tokens}")

    # inference using pipeline (pros: embedded dataloader)
    # """
    # asr = pipeline("automatic-speech-recognition", model=args.model_id, device=args.device, **model_args)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        # todo
        # max_new_tokens=128,
        # todo
        # chunk_length_s=args.chunk_length_s,
        # stride_length_s=args.stride_length_s,
        # num_workers=args.num_workers,
        # batch_size=args.batch_size,
        torch_dtype=torch_dtype,
        # device=args.device,
    )

    # desceding order to early check if OOM
    dataset = dataset.sort(args.length_column_name, reverse=True, keep_in_memory=True)

    start_time = time.perf_counter()

    predictions = []
    for out in tqdm(
        asr(
            KeyDataset(dataset, args.audio_column_name),
            chunk_length_s=args.chunk_length_s,
            stride_length_s=args.stride_length_s,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        ),
        total=len(dataset),
    ):
        # Exactly the same output as before, but the content are passed
        # as batches to the model
        # print(out)
        predictions.append(out["text"])

    print(f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}.")

    def collect_result(example, idx):
        example["prediction"] = predictions[idx]
        # example["target"] = normalize_text(example[args.text_column_name], invalid_chars_regex)
        return example

    # result = dataset.map(collect_result, remove_columns=dataset.column_names, with_indices=True)
    result = dataset.map(collect_result, with_indices=True, keep_in_memory=True)
    # """

    # bh: inference by batch with low-level APIs
    """
    gen_greedy = {"do_sample": False, "num_beams": 1}
    gen_greedy_sampling = {"do_sample": True, "num_beams": 1}
    # While nucleus sampling can generate text free of repetitions, the semantic coherence of the generated text is not well-maintained.
    gen_nucleus = {"do_sample": True, "num_beams": 1, "top_p": 0.95, "top_k": 0}
    # reducing the temperature brings nucleus sampling closer to greedy search, which can be seen as a trade-off between greedy search and nucleus sampling.
    gen_nucleus_temperature = {"do_sample": True, "num_beams": 1, "top_p": 0.95, "top_k": 0, "temperature": 0.7}

    gen_beam = {"do_sample": False, "num_beams": 5}
    gen_beam_sampling = {"do_sample": True, "num_beams": 5}
    gen_beam_group = {"do_sample": False, "num_beams": 10, "num_beam_groups": 2}

    # When generating output, contrastive search jointly considers
    # (i) the probability predicted by the language model to maintain the semantic coherence
    # between the generated text and the prefix text
    # (ii) the similarity with respect to the previous context to avoid model degeneration.
    gen_contrastive_search = {"top_k": 6, "penalty_alpha": 0.6}

    gen_kwargs = {
        "max_new_tokens": 225,
        # "max_new_tokens": 40,
        "num_beams": args.gen_num_beams,
        # "repetition_penalty"
        # "length_penalty"
        # "no_repeat_ngram_size"
        # "bad_words_ids"
        # "num_return_sequences"
    }

    def map_to_pred(batch):
        # bh: synchronised process and forward, this can be improved by dataloader
        inputs = processor(
            [example["array"] for example in batch[args.audio_column_name]],
            sampling_rate=model_sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features
        input_features = input_features.to(args.device)

        if args.fp16:
            input_features = input_features.half()

        generated_ids = model.generate(inputs=input_features, **gen_kwargs)
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        batch["prediction"] = transcriptions
        # normalize prediction
        # batch["prediction"] = [norm_func(prediction) for prediction in transcriptions]

        # batch["target"] = batch[args.text_column_name]
        # normalize target
        # batch["target"] = [norm_func(target) for target in batch[args.text_column_name]]

        return batch

    result = dataset.map(map_to_pred, batched=True, batch_size=args.batch_size, keep_in_memory=True)
    """

    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)

    with open(args.output_file_path, "w") as manifest_f:
        for _, sample in enumerate(tqdm(result, desc="Saving", total=result.num_rows, unit=" samples")):
            # tmp fix
            # sample[args.audio_column_name] = sample.pop(args.audio_column_name)["path"]
            sample[args.audio_column_name] = sample.pop(f"tmp_{args.audio_column_name}")
            manifest_f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")

    # bh: fake ID if not exists
    # if args.id_column_name not in result.features.keys():
    #     result = result.map(lambda example, idx: {**example, args.id_column_name: f"{idx:09d}"}, with_indices=True)

    # filtering out empty targets
    # result = result.filter(lambda example: example["target"] != "")

    # compute and log_results
    # do not change function below
    # todo
    # log_results(result, args)

    # speechbrain's WER computation util
    # todo
    # eval_results(result, args, do_ignore_words=False)
    # eval_results(result, args, do_ignore_words=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers"
    )
    parser.add_argument("--input_file_path", type=str, default=None, help="Path to input test file")
    parser.add_argument("--output_file_path", type=str, default=None, help="Path to save results")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        # required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument("--config", type=str, help="Config of the dataset. *E.g.* `'en'`  for Common Voice")
    parser.add_argument("--split", type=str, help="Split of the dataset. *E.g.* `'test'`")
    parser.add_argument("--audio_column_name", type=str, default="audio")
    parser.add_argument("--text_column_name", type=str, default="sentence")
    parser.add_argument("--id_column_name", type=str, default="ID")
    parser.add_argument("--start_column_name", type=str, default="start")
    parser.add_argument("--end_column_name", type=str, default="end")
    parser.add_argument("--length_column_name", type=str, default="length")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--chunk_length_s",
        type=float,
        default=None,
        help="Chunk length in seconds. Defaults to None. For long audio files a good value would be 5.0 seconds.",
    )
    parser.add_argument(
        "--stride_length_s",
        type=float,
        default=None,
        help="Stride of the audio chunks. Defaults to None. For long audio files a good value would be 1.0 seconds.",
    )
    parser.add_argument("--log_outputs", action="store_true", help="If defined, write outputs to log file for analysis.")
    parser.add_argument("--greedy", action="store_true", help="If defined, the LM will be ignored during inference.")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    # parser.add_argument("--outdir", type=str, required=True, help="Save path.")
    parser.add_argument("--language", type=str, default="french", help="Language token")
    parser.add_argument("--task", type=str, default="transcribe", help="Task token")
    parser.add_argument("--fp16", action="store_true", help="Downcast model and data to fp16")
    parser.add_argument("--gen_num_beams", type=int, default=1)

    args = parser.parse_args()

    main(args)
