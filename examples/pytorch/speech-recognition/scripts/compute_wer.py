#!/usr/bin/env python
# Copyright 2023  Bofeng Huang

import sys

sys.path.append("/home/bhuang/myscripts")

import json
import re

# from jiwer import wer
import evaluate
import fire
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from jiwer import process_words
from nltk import ngrams

from asr_metric_calculation.compute_wer import compute_wer
from text_normalization.normalize_french import FrenchTextNormalizer
# from .hf_dataset_processing.file_utils import write_dataset_to_json

disable_progress_bar()

# from nltk.stem.snowball import SnowballStemmer
# from nltk import word_tokenize
# Download the nltk data for tokenization
# import nltk
# nltk.download("punkt")
# Initialize SnowballStemmer with the specified language
# stemmer = SnowballStemmer("french")


# def stem_sentence(sentence):
#     # Tokenize the sentence into words
#     words = word_tokenize(sentence)

#     # Stem each word in the sentence
#     stemmed_words = [stemmer.stem(word) for word in words]

#     # Join the stemmed words back into a sentence
#     stemmed_sentence = " ".join(stemmed_words)

#     return stemmed_sentence


def compute_metrics(references, predictions, ngram_degree=5):
    wer_output = process_words(references, predictions)

    num_ref_words = sum([len(ref) for ref in wer_output.references])

    wer_norm = 100 * wer_output.wer
    ier_norm = 100 * wer_output.insertions / num_ref_words
    ser_norm = 100 * wer_output.substitutions / num_ref_words
    der_norm = 100 * wer_output.deletions / num_ref_words

    all_ngrams = list(ngrams(" ".join(predictions).split(), ngram_degree))
    repeated_ngrams_per_dataset = len(all_ngrams) - len(set(all_ngrams))

    repeated_ngrams_per_utt = 0
    for prediction_ in predictions:
        all_ngrams_ = list(ngrams(prediction_.split(), ngram_degree))
        repeated_ngrams_per_utt += len(all_ngrams_) - len(set(all_ngrams_))

    return {
        "wer": wer_norm,
        "ier": ier_norm,
        "ser": ser_norm,
        "der": der_norm,
        # f"repeated_{ngram_degree}grams": repeated_ngrams,
        f"per_dataset_repeated_{ngram_degree}grams": repeated_ngrams_per_dataset,
        f"per_utterance_repeated_{ngram_degree}grams": repeated_ngrams_per_utt,
        "num_ref_words": num_ref_words,
        "num_ref_sentences": len(references),
    }


def main(
    input_file_path: str,
    output_dir: str,
    # suffix: str = "_wer",
    id_column_name: str = "id",
    target_column_name: str = "text",
    prediction_column_name: str = "prediction",
    ngram_degree: int = 5,
    num_processing_workers: int = 32,
    do_ignore_words: bool = False,
):
    path, ext = input_file_path.rsplit(".", 1)
    # output_file_path = f"{path}{suffix}.{ext}"
    dataset = load_dataset(ext, data_files=input_file_path, split="train")
    print(dataset)

    # Debug
    # dataset = dataset.select(range(1000))

    normalizer = FrenchTextNormalizer()

    def process_function(example, idx):
        def normalize_(s):
            # s = re.sub(r"<[0-9\.]+>", "", s)  # remove timstamps
            s = normalizer(
                s, do_lowercase=True, do_ignore_words=do_ignore_words, symbols_to_keep="'", do_num2text=True
            )  # w/o "-"

            return s

        example[f"normalized_{target_column_name}"] = normalize_(example[target_column_name])
        example[f"normalized_{prediction_column_name}"] = normalize_(example[prediction_column_name])

        example[f"{target_column_name}_split"] = example[target_column_name].split()
        example[f"{prediction_column_name}_split"] = example[prediction_column_name].split()
        example[f"normalized_{target_column_name}_split"] = example[f"normalized_{target_column_name}"].split()
        example[f"normalized_{prediction_column_name}_split"] = example[f"normalized_{prediction_column_name}"].split()

        # split into characters
        # norm_ = lambda s: " ".join(re.sub(r"\s+", "", s))
        # example[f"normalized_{target_column_name}"] = norm_(example[f"normalized_{target_column_name}"])
        # example[f"normalized_{prediction_column_name}"] = norm_(example[f"normalized_{prediction_column_name}"])

        # example["wer"] = wer(
        #     example[f"normalized_{target_column_name}"], example[f"normalized_{prediction_column_name}"]
        # )

        # example["stemmed_wer"] = wer(
        #     stem_sentence(example[f"normalized_{target_column_name}"]),
        #     stem_sentence(example[f"normalized_{prediction_column_name}"]),
        # )

        if id_column_name not in example:
            example[id_column_name] = f"{idx:09d}"

        return example

    dataset = dataset.map(
        process_function,
        with_indices=True,
        num_proc=num_processing_workers,
        # remove_columns=raw_datasets.column_names,
        load_from_cache_file=False,
        desc="process",
    )
    # print(dataset)

    # write_dataset_to_json(dataset, output_file_path)
    # print(f"The processed data is saved into {output_file_path}")

    # filtering out empty targets
    dataset = dataset.filter(
        lambda x: len(x) > 0, input_columns=f"normalized_{target_column_name}", num_proc=num_processing_workers
    )

    assert dataset.num_rows == len(set(dataset[id_column_name]))

    # load metric
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    overall_wer = wer_metric.compute(references=dataset[target_column_name], predictions=dataset[prediction_column_name])
    norm_overall_wer = wer_metric.compute(
        references=dataset[f"normalized_{target_column_name}"], predictions=dataset[f"normalized_{prediction_column_name}"]
    )
    overall_cer = cer_metric.compute(references=dataset[target_column_name], predictions=dataset[prediction_column_name])
    norm_overall_cer = cer_metric.compute(
        references=dataset[f"normalized_{target_column_name}"], predictions=dataset[f"normalized_{prediction_column_name}"]
    )
    print(f"WER: {overall_wer:.4f}, Normalized WER: {norm_overall_wer:.4f}")
    print(f"CER: {overall_cer:.4f}, Normalized CER: {norm_overall_cer:.4f}")

    result = compute_metrics(
        dataset[f"normalized_{target_column_name}"], dataset[f"normalized_{prediction_column_name}"], ngram_degree=ngram_degree
    )
    print("\nNORMALIZED METRICS")
    print(json.dumps(result, indent=4))

    # speechbrain-style wer and alignment
    targets = dict(zip(dataset[id_column_name], dataset[f"{target_column_name}_split"]))
    predictions = dict(zip(dataset[id_column_name], dataset[f"{prediction_column_name}_split"]))
    compute_wer(targets, predictions, f"{output_dir}/wer_summary", do_print_top_wer=True, do_catastrophic=True)

    targets = dict(zip(dataset[id_column_name], dataset[f"normalized_{target_column_name}_split"]))
    predictions = dict(zip(dataset[id_column_name], dataset[f"normalized_{prediction_column_name}_split"]))
    compute_wer(targets, predictions, f"{output_dir}/normalized_wer_summary", do_print_top_wer=True, do_catastrophic=True)


if __name__ == "__main__":
    # main()  # noqa pylint: disable=no-value-for-parameter
    fire.Fire(main)
