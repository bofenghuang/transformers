#!/usr/bin/env python
# Copyright 2023  Bofeng Huang

import json
import os
import sys
import time
from pathlib import Path

import evaluate
import fire
import pandas as pd
from tqdm import tqdm

from .predict_ner_case_punkt import TokenClassificationPredictor
from utils.dataset_ner_case_punkt import load_data_files


def read_json_manifest(input_json_file):
    data = []
    with open(input_json_file, "r") as inf:
        for line in inf:
            data.append(json.loads(line.strip()))
    return data


def write_json_manifest(output_json_file, data):
    output_dir = os.path.dirname(output_json_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_json_file, "w") as outf:
        for row in data:
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(
    model_name_or_path,
    normalizer_file,
    input_dir,
    input_filename,
    output_suffix,
    text_column_name="text",
    output_text_column_name="punctuated_text",
):

    start_time = time.time()

    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/camembert-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/camembert-large_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/xlm-roberta-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/xlm-roberta-large_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc/camembert-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc/xlm-roberta-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc-multilingual/xlm-roberta-large_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc-multilingual_plus/xlm-roberta-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc-multilingual_plus/xlm-roberta-large_ft"
    # test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/test"
    # test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/tmp"
    # test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/en_europarl/data/test"
    # test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/es_europarl/data/test"
    # test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/de_europarl/data/test"
    # test_data_dir = "/home/bhuang/corpus/text/internal/punctuation/2022-10-05/data"

    tc = TokenClassificationPredictor(
        model_name_or_path,
        device=0,
        # normalizer_file="./normalizer.json",
        normalizer_file=normalizer_file,
        do_pre_normalize=True,
        stride=100,
        # batch_size=64,
        # aggregation_strategy=AggregationStrategy.LAST,
    )

    print(f"Model from {model_name_or_path} has been loaded")

    # Inference CSV file
    """
    # input_file = "/home/bhuang/corpus/text/internal/punctuation/2022-11-29/data_dekuple_200.csv"
    # output_file = "/home/bhuang/corpus/text/internal/punctuation/2022-11-29/data_dekuple_200_generated.csv"
    # input_file = "/home/bhuang/corpus/text/internal/punctuation/2022-11-29/data_carglass_200.csv"
    # output_file = "/home/bhuang/corpus/text/internal/punctuation/2022-11-29/data_carglass_200_generated.csv"
    df = pd.read_csv(input_file, sep="\t")
    print(df.head())
    # df = df.head(5)
    # df["wrd"] = df["wrd"].map(lambda x: tc(x)[0])
    # df["text"] = df["text"].map(lambda x: tc(x)[0])
    df[output_text_column_name] = df[text_column_name].map(lambda x: tc(x)[0])
    print(df.head())
    # df[["ID", "wrd"]].to_csv(output_file, index=False, sep="\t")
    # df[["utt", "text"]].to_csv(output_file, index=False, sep="\t")
    df.to_csv(output_file, index=False, sep="\t")
    """

    # Iterate files
    # """
    for p in Path(input_dir).rglob(input_filename):
        print(f"Punctuate the segments in file {p.as_posix()}")

        json_data = read_json_manifest(p.as_posix())
        for row in tqdm(json_data):
            if row[text_column_name]:
                row[output_text_column_name] = tc(row[text_column_name])[0]
            else:
                row[output_text_column_name] = row[text_column_name]

        output_json_path = f'{p.parent / f"{p.stem}{output_suffix}.json"}'
        write_json_manifest(output_json_path, json_data)
    # """

    """
    sentences = [
        "ca va",
        "what's up bro",
        "bonjour comment ca va",
        "bonjour j'aimerais savoir quelle est la réponse quelle était la question déjà",
        "qué tal",
        "que es la pregunta",
        "what's up",
    ]
    for sentence in sentences:
        # print(tc(sentence))
        res = tc.predict(sentence)
        print(tc.prediction_to_text(res)[0])
        for r in res[0]:
            print(r)
    """

    """
    # don't forget replacers cc
    # test_ds = load_data_files(test_data_dir, task_config="punc", replacers={"EXCLAMATION": "PERIOD"}, preprocessing_num_workers=16)
    # test_ds = load_data_files(test_data_dir, task_config="eos", preprocessing_num_workers=16)
    test_ds = load_data_files(
        test_data_dir,
        task_config=["case", "punc"],
        replacers={"case": {"OTHER": "LOWER"}, "punc": {"EXCLAMATION": "PERIOD", "COLON": "COMMA"}},
        preprocessing_num_workers=16,
    )

    def predict_(examples):
        sentences = [" ".join(words) for words in examples["word"]]
        # examples["hypothese"] = [[pred_["entity"] for pred_ in pred] for pred in tc.predict(sentences)]
        predictions = tc.predict(sentences)
        examples["case_hypothese"] = [[pred_["case_entity"] for pred_ in pred] for pred in predictions]
        examples["punc_hypothese"] = [[pred_["punct_entity"] for pred_ in pred] for pred in predictions]
        examples["word_pred"] = [[pred_["word"] for pred_ in pred] for pred in predictions]
        return examples

    # test_ds = test_ds.select(range(10))
    # test_ds = test_ds.select([1058])
    test_ds = test_ds.map(predict_, batched=True, batch_size=256)
    # print(test_ds)
    # print(test_ds[0])
    # print(test_ds[1])
    # quit()

    # debug
    # for w, w_p in zip(test_ds[0]["word"], test_ds[0]["word_pred"]):
    #     print(w, w_p)
    # quit()

    # metric = evaluate.load("seqeval")
    # results = metric.compute(predictions=test_ds["hypothese"], references=test_ds["label"])
    # print(results)

    # tmp_outdir = f"{model_name_or_path}/results/predict_words"
    if not os.path.exists(tmp_outdir):
        os.makedirs(tmp_outdir)

    with open(f"{tmp_outdir}/case_references.txt", "w") as writer:
        for labels_ in test_ds["case_label"]:
            writer.write(" ".join(labels_) + "\n")

    with open(f"{tmp_outdir}/case_predictions.txt", "w") as writer:
        for prediction_ in test_ds["case_hypothese"]:
            writer.write(" ".join(prediction_) + "\n")

    with open(f"{tmp_outdir}/punc_references.txt", "w") as writer:
        for labels_ in test_ds["punc_label"]:
            writer.write(" ".join(labels_) + "\n")

    with open(f"{tmp_outdir}/punc_predictions.txt", "w") as writer:
        for prediction_ in test_ds["punc_hypothese"]:
            writer.write(" ".join(prediction_) + "\n")
    """

    print(f"Run time: {(time.time() - start_time): .2f}s")


if __name__ == "__main__":
    fire.Fire(main)