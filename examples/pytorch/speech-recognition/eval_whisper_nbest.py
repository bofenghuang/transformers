#!/usr/bin/env python
# Copyright 2021  Bofeng Huang

import argparse
import os
import re
from typing import Dict
from tqdm import tqdm
from pathlib import Path

import torch
import torchaudio
import datasets
from datasets import Audio, Dataset, load_dataset, load_metric

from transformers import AutoConfig, AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor, pipeline
from transformers.pipelines.pt_utils import KeyDataset

import torchaudio.sox_effects as ta_sox

# from utils.normalize_text_fr_zaion import FrenchTextNormalizer
from utils.normalize_text_hf_sprint import FrenchTextNormalizer

text_normalizer = FrenchTextNormalizer()


def log_results(result: Dataset, args: Dict[str, str]):
    """DO NOT CHANGE. This function computes and logs the result metrics."""

    log_outputs = args.log_outputs
    # dataset_id = "_".join(args.dataset.split("/") + [args.config, args.split])
    # dataset_id = Path(args.test_csv_file).stem

    # load metric
    wer = load_metric("wer")
    cer = load_metric("cer")

    # compute metrics
    # wer_result = wer.compute(references=result["target"], predictions=result["prediction"])
    # cer_result = cer.compute(references=result["target"], predictions=result["prediction"])
    wer_result = wer.compute(references=result["target"], predictions=result["prediction_0"])
    cer_result = cer.compute(references=result["target"], predictions=result["prediction_0"])

    # print & log results
    result_str = f"WER: {wer_result}\n" f"CER: {cer_result}"
    print(result_str)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # with open(f"{args.outdir}/{dataset_id}_eval_results.txt", "w") as f:
    with open(f"{args.outdir}/eval_results.txt", "w") as f:
        f.write(result_str)

    # log all results in text file. Possibly interesting for analysis
    if log_outputs is not None:
        # pred_file = f"{args.outdir}/log_{dataset_id}_predictions.txt"
        # target_file = f"{args.outdir}/log_{dataset_id}_targets.txt"
        pred_file = f"{args.outdir}/log_predictions.txt"
        target_file = f"{args.outdir}/log_targets.txt"

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
                # p.write(batch[args.id_column_name] + "\t" + batch["prediction"] + "\n")
                p.write(batch[args.id_column_name] + "\t" + "\t".join([batch[f"prediction_{nbest_i}"] for nbest_i in range(10)]) + "\n")
                t.write(batch[args.id_column_name] + "\t" + batch["target"] + "\n")

            result.map(write_to_file, with_indices=True)


def normalize_text(text: str, invalid_chars_regex: str) -> str:
    """DO ADAPT FOR YOUR USE CASE. this function normalizes the target text."""

    text = text.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(invalid_chars_regex, " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main(args):
    # load dataset
    if args.test_csv_file is not None:
        dataset = load_dataset("csv", data_files=args.test_csv_file)["train"]
    elif args.dataset is not None:
        dataset = load_dataset(args.dataset, args.config, split=args.split, use_auth_token=True)
    else:
        raise ValueError("You have not specified a dataset name nor a custom validation file")

    # Debug: only process the first two examples as a test
    # dataset = dataset.select(range(10))

    # when got some real word "nan" in csv file
    # def correct_text(example):
    #     example[args.text_column_name] = example[args.text_column_name] if example[args.text_column_name] is not None else "nan"
    #     return example
    # dataset = dataset.map(correct_text, num_proc=args.preprocessing_num_workers, desc="Correct readed text")
    # dataset = dataset.sort(args.length_column_name)
    # print(dataset["wrd"][:100])
    # print(None in dataset["wrd"])
    # quit()

    if args.test_csv_file is not None:
        # read audio segments by timestamps
        sr = 8_000

        def read_segments(example):
            start = int(example[args.start_column_name] * sr)
            end = int(example[args.end_column_name] * sr)
            num_frames = end - start

            waveform, sample_rate = torchaudio.load(example[args.audio_column_name], frame_offset=start, num_frames=num_frames)
            # ! read full wavs
            # waveform, _ = torchaudio.load(example[args.audio_column_name], frame_offset=0, num_frames=-1)

            # bh: normalize wav before computing fbanks
            # effects = []
            # # normalize
            # effects.append(["gain", "-n"])
            # # sampling
            # # effects.append(["rate", f"{to_sample_rate}"])
            # # to mono
            # # effects.append(["channels", "1"])
            # converted_waveform, converted_sample_rate = ta_sox.apply_effects_tensor(
            #     waveform, sample_rate, effects
            # )
            # converted_waveform = converted_waveform.squeeze(axis=0)  # mono

            converted_waveform = waveform.squeeze(axis=0)  # mono
            converted_sample_rate = sample_rate

            example[args.audio_column_name] = {
                "path": example[args.audio_column_name],
                "array": converted_waveform.numpy(),
                # "array": waveform,
                "sampling_rate": converted_sample_rate,
            }

            return example

        """
        dataset = dataset.map(read_segments, num_proc=args.preprocessing_num_workers, desc="read audio segments by timestamps")
        dataset = dataset.cast_column(args.audio_column_name, datasets.features.Audio(sampling_rate=sr))
        # print(dataset[0])
        # print(dataset[0][args.audio_column_name])
        # print(dataset[0]["wav"]["array"].shape)
        # quit()
        # dataset.save_to_disk("/projects/bhuang/.cache/hf_outputs/asr/hmhm_test/dump_readed")
        """
        # dataset = datasets.load_from_disk("/projects/bhuang/.cache/hf_outputs/asr/hmhm_test/dump_readed")

        # bh: whisper doesn't take segments longer than 30s
        max_input_length = 30 * 16_000
        min_input_length = 1 * 16_000

        def is_audio_in_length_range(example):
            length = example[args.audio_column_name]["array"].shape[0]
            # return length > min_input_length and length < max_input_length
            return length < max_input_length

        # dataset = dataset.filter(is_audio_in_length_range, num_proc=args.preprocessing_num_workers)
        # dataset.save_to_disk("/projects/bhuang/.cache/hf_outputs/asr/hmhm_test/dump_readed_max30s")
        dataset = datasets.load_from_disk("/projects/bhuang/.cache/hf_outputs/asr/hmhm_test/dump_readed_max30s")

    # load processor
    # processor = AutoProcessor.from_pretrained(args.model_id)
    # processor = AutoProcessor.from_pretrained(args.model_id, language="french", task="transcribe")
    # bh: set prefix tokens for tokenizer
    processor = AutoProcessor.from_pretrained(args.model_id, language=args.language, task=args.task)
    # todo: how to boost performance? ngram prefix beam search? shallow fusion?
    decoder = None

    feature_extractor = processor.feature_extractor
    # tokenizer = processor.tokenizer
    sampling_rate = feature_extractor.sampling_rate

    # resample audio
    dataset = dataset.cast_column(args.audio_column_name, Audio(sampling_rate=sampling_rate))
    # print(dataset[0])
    # print(dataset[0]["wav"]["array"].shape)

    # load eval pipeline
    if args.device is None:
        args.device = 0 if torch.cuda.is_available() else -1

    # config = AutoConfig.from_pretrained(args.model_id)
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id)
    model_args = {}
    if args.fp16:
        model_args["torch_dtype"] = torch.float16
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, **model_args)
    # todo bh: load model in fp16
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, torch_dtype=torch.float16)

    # bh: force prefix tokens for generation utils
    # model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="fr", task="transcribe")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    print(f"Model `forced_decoder_ids`: {model.config.forced_decoder_ids}")
    # todo: include some other tokens dropped by fine-tuning
    # tmp_config = AutoConfig.from_pretrained("openai/whisper-medium")
    # model.config.suppress_tokens = tmp_config.suppress_tokens
    # print(f"Model `suppress_tokens`: {model.config.suppress_tokens}")

    model.eval()
    model = model.to(args.device)

    # todo: test new pipeline
    # asr = pipeline("automatic-speech-recognition", model=args.model_id, device=args.device)
    # asr = pipeline(
    #     "automatic-speech-recognition",
    #     config=config,
    #     model=model,
    #     tokenizer=tokenizer,
    #     feature_extractor=feature_extractor,
    #     decoder=decoder,
    #     device=args.device,
    # )

    # build normalizer config
    # tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # tokens = [x for x in tokenizer.convert_ids_to_tokens(range(0, tokenizer.vocab_size))]
    # special_tokens = [
    #     tokenizer.pad_token,
    #     tokenizer.word_delimiter_token,
    #     tokenizer.unk_token,
    #     tokenizer.bos_token,
    #     tokenizer.eos_token,
    # ]
    # non_special_tokens = [x for x in tokens if x not in special_tokens]
    # invalid_chars_regex = f"[^\s{re.escape(''.join(set(non_special_tokens)))}]"

    # normalize_to_lower = False
    # for token in non_special_tokens:
    #     if token.isalpha() and token.islower():
    #         normalize_to_lower = True
    #         break

    # bh: inference one by one
    """
    # map function to decode audio
    def map_to_pred(batch):
        # prediction = asr(batch["audio"]["array"], chunk_length_s=args.chunk_length_s, stride_length_s=args.stride_length_s)
        # prediction = asr(batch[args.audio_column_name]["array"], chunk_length_s=args.chunk_length_s, stride_length_s=args.stride_length_s)
        # todo
        prediction = asr(batch[args.audio_column_name]["array"])

        # batch["prediction"] = prediction["text"]
        batch["prediction"] = text_normalizer(prediction["text"])

        # batch["target"] = normalize_text(batch[args.text_column_name], invalid_chars_regex)
        batch["target"] = batch[args.text_column_name]
        return batch

    # run inference on all examples
    # result = dataset.map(map_to_pred, remove_columns=dataset.column_names)
    # bh: keep ID for tracking
    result = dataset.map(map_to_pred)
    """

    # bh: inference by batch
    """
    # desceding order to early check if OOM
    # dataset = dataset.sort(args.length_column_name, reverse=True)

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

    def collect_result(example, idx):
        example["prediction"] = predictions[idx]
        example["target"] = normalize_text(example[args.text_column_name], invalid_chars_regex)
        return example

    # result = dataset.map(collect_result, remove_columns=dataset.column_names, with_indices=True)
    result = dataset.map(collect_result, with_indices=True)
    """

    # bh: inference by batch with low APIs
    # """
    gen_greedy = {"do_sample": False, "num_beams": 1}
    gen_greedy_sampling = {"do_sample": True, "num_beams": 1}
    gen_greedy_sampling_nbest = {**gen_greedy_sampling, "return_dict_in_generate": True, "output_scores": True, "num_return_sequences": 10}
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
        **gen_greedy_sampling_nbest,
        # "repetition_penalty"
        # "length_penalty"
        # "no_repeat_ngram_size"
        # "bad_words_ids"
        # "num_return_sequences"
    }

    def map_to_pred(batch):
        # bh: synchronised process and forward, this can be improved by dataloader
        inputs = processor(
            [example["array"] for example in batch[args.audio_column_name]], sampling_rate=16_000, return_tensors="pt"
        )
        input_features = inputs.input_features
        input_features = input_features.to(args.device)

        if args.fp16:
            input_features = input_features.half()

        # generated_ids = model.generate(inputs=input_features, **gen_kwargs)
        outputs_ = model.generate(inputs=input_features, **gen_kwargs)
        generated_ids = outputs_.sequences  # shape BS x NBEST

        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # batch["prediction"] = transcriptions
        # normalize prediction
        predictions = [text_normalizer(prediction) for prediction in transcriptions]

        nbest = gen_greedy_sampling_nbest["num_return_sequences"]
        for nbest_i in range(nbest):
            batch[f"prediction_{nbest_i}"] = [predictions[i + nbest_i] for i in range(0, len(predictions), nbest)]

        # batch["target"] = batch[args.text_column_name]
        # normalize target
        # batch["target"] = normalize_text(batch[args.text_column_name], invalid_chars_regex)
        batch["target"] = [text_normalizer(target) for target in batch[args.text_column_name]]

        return batch

    result = dataset.map(map_to_pred, batched=True, batch_size=args.batch_size)
    # """

    # fake ID if not exists
    if args.id_column_name not in result.features.keys():
        result = result.map(lambda example, idx: {**example, args.id_column_name: f"{idx:09d}"}, with_indices=True)

    # filtering out empty targets
    # result = result.filter(lambda example: example["target"] != "")

    # compute and log_results
    # do not change function below
    log_results(result, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers")
    parser.add_argument("--test_csv_file", type=str, default=None, help="test csv path")
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
    parser.add_argument("--outdir", type=str, required=True, help="Save path.")
    parser.add_argument("--language", type=str, default="french", help="Language token")
    parser.add_argument("--task", type=str, default="transcribe", help="Task token")
    parser.add_argument("--fp16", action="store_true", help="Downcast model and data to fp16")

    args = parser.parse_args()

    main(args)
