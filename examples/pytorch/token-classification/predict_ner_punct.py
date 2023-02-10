#!/usr/bin/env python
# Copyright 2021  Bofeng Huang

import os
import re
import unicodedata
import fire
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, BatchEncoding


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}")


class AggregationStrategy(ExplicitEnum):
    """All the valid aggregation strategies for TokenClassificationPipeline"""

    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    LAST = "last"
    AVERAGE = "average"
    MAX = "max"


def remove_symbols_and_diacritics(s: str, keep=""):
    r"""Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        c if c in keep else "" if unicodedata.category(c) == "Mn" else " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKD", s)
    )


class TokenClassificationPredictor:

    punc_label2text = {"0": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}

    def __init__(self, model_name_or_path: str, device: Union[torch.device, str, int] = "cpu", **kwargs) -> None:
        # load config
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # ! only for camembert large
        # this model has a wrong maxlength value, so we need to set it manually
        # if model_name_or_path == "camembert/camembert-large":
        self.tokenizer.model_max_length = 512
        # device
        self.device = self.configure_device(device)
        # load model
        # todo: onnx
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        model.eval()
        self.model = model.to(self.device)

        # todo: set by config
        # kwargs.update(
        #     {
        #         "max_length": 20,
        #         "stride": 10,
        #         "batch_size": 4,
        #         "overlap": 10,
        #         "aggregation_strategy": AggregationStrategy.LAST,
        #     }
        # )

        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)

    def _sanitize_parameters(
        self,
        do_pre_normalize: bool = False,
        max_length: Optional[int] = None,
        stride: Optional[int] = None,
        batch_size: Optional[int] = None,
        overlap: Optional[int] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
    ):
        preprocess_params = {}
        if do_pre_normalize:
            preprocess_params["do_pre_normalize"] = do_pre_normalize
        if max_length is not None:
            preprocess_params["max_length"] = max_length
        if stride is not None:
            preprocess_params["stride"] = stride

        forward_params = {}
        if batch_size is not None:
            forward_params["batch_size"] = batch_size

        postprocess_params = {}
        if overlap is not None:
            postprocess_params["overlap"] = overlap
        if aggregation_strategy is not None:
            postprocess_params["aggregation_strategy"] = aggregation_strategy

        return preprocess_params, forward_params, postprocess_params

    def configure_device(self, device: Union[torch.device, str, int]) -> torch.device:
        if isinstance(device, torch.device):
            return device
        elif isinstance(device, str):
            return torch.device(device)
        elif device < 0:
            return torch.device("cpu")
        else:
            return torch.device(f"cuda:{device}")

    def predict(self, inputs, **kwargs):
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)

        return outputs

    def __call__(self, inputs, **kwargs):
        return self.prediction_to_text(self.predict(inputs, **kwargs))

    @staticmethod
    def pre_normalize_text(s: str):
        s = s.lower()  # remove existing case

        # same standarization when preparing training data
        s = re.sub(r"(?<=\w)\s+'", r"'", s)  # standardize when there's a space before an apostrophe
        s = re.sub(r"(?<!aujourd)(?<=\w)'\s*(?=\w)", "' ", s)  # add an espace after an apostrophe (except)

        s = remove_symbols_and_diacritics(s, keep="'-")  # remove existing punc

        s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space
        return s

    @staticmethod
    def post_normalize_text(s: str):
        s = re.sub(r"\s+'\s+", "'", s)  # standardize when there's a space before/after an apostrophe
        s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space
        return s

    def preprocess(
        self, inputs: Union[str, List[str]], do_pre_normalize: bool = False, max_length: Optional[int] = None, stride: int = 0
    ):
        if isinstance(inputs, str):
            inputs = [inputs]

        if do_pre_normalize:
            inputs = [TokenClassificationPredictor.pre_normalize_text(inputs_) for inputs_ in inputs]

        tokenized_inputs = self.tokenizer(
            inputs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            stride=stride,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        tokenized_inputs["sentence"] = [inputs[i] for i in tokenized_inputs["overflow_to_sample_mapping"]]
        return tokenized_inputs

    def _forward(self, model_inputs):
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping")
        overflow_to_sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        features = model_inputs.to(self.device)
        with torch.inference_mode():
            outputs = self.model(**features)
            logits = outputs[0]

        return {
            "logits": logits.to(torch.device("cpu")),
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "overflow_to_sample_mapping": overflow_to_sample_mapping,
            **model_inputs.to(torch.device("cpu")),
        }

    def forward(self, model_inputs, batch_size: Optional[int] = None):
        sentences = model_inputs.pop("sentence")

        if batch_size is not None:
            model_inputs = [
                BatchEncoding({k: v[i : i + batch_size] for k, v in model_inputs.items()})
                for i in range(0, len(model_inputs["overflow_to_sample_mapping"]), batch_size)
            ]
        else:
            model_inputs = [model_inputs]
        # print(model_inputs)

        model_outputs = []
        for model_inputs_ in model_inputs:
            model_outputs.append(self._forward(model_inputs_))
        # print(model_outputs)

        # gather batched outputs
        # list of dict back to dict of list
        model_outputs = {
            k: torch.concat([model_outputs_[k] for model_outputs_ in model_outputs], dim=0) for k in model_outputs[0].keys()
        }
        model_outputs["sentence"] = sentences
        # print(model_outputs)

        # debug
        # import pickle
        # with open("tmp.pkl", "wb") as f:
        #     pickle.dump(model_outputs, f)

        return model_outputs

    def postprocess(
        self, model_outputs, overlap: int = 0, aggregation_strategy: AggregationStrategy = AggregationStrategy.LAST
    ):
        # convert tensors to np arrays
        model_outputs = {k: v.numpy() if torch.is_tensor(v) else v for k, v in model_outputs.items()}

        # debug
        # for input_ids_, logits_ in zip(model_outputs["input_ids"], model_outputs["logits"]):
        #     for i, l in zip(input_ids_, logits_):
        #         print(self.tokenizer.convert_ids_to_tokens([i]), l)
        #     print("\n")
        # quit()

        # gather splitted sub results
        sentences, input_ids, logits, offset_mapping = self.gather_sub_results(model_outputs, overlap)
        # print(self.tokenizer.decode(input_ids[0]))
        # quit()

        entities = []
        for sentence_, input_ids_, logits_, offset_mapping_ in zip(sentences, input_ids, logits, offset_mapping):

            maxes = np.max(logits_, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits_ - maxes)
            scores_ = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

            pre_entities_ = self.gather_pre_entities(sentence_, input_ids_, scores_, offset_mapping_)
            entities_ = self.aggregate_words(pre_entities_, aggregation_strategy)

            entities.append(entities_)

        # todo: group entities if across words

        return entities

    def gather_sub_results(self, model_outputs, overlap: int = 0):
        sentences = []
        aggregated_input_ids = []
        aggregated_logits = []
        aggregated_offset_mapping = []

        # sorted unique
        for sentence_id in np.unique(model_outputs["overflow_to_sample_mapping"]):
            indexes_ = model_outputs["overflow_to_sample_mapping"] == sentence_id
            input_ids_sentence_splits = model_outputs["input_ids"][indexes_]
            logits_sentence_splits = model_outputs["logits"][indexes_]
            offset_mapping_sentence_splits = model_outputs["offset_mapping"][indexes_]
            special_tokens_mask_sentence_splits = model_outputs["special_tokens_mask"][indexes_]

            overlap_ = overlap

            input_ids_sentence = []
            logits_sentence = []
            offset_mapping_sentence = []
            for i, (input_ids_, logits_, offset_mapping_, special_tokens_mask_) in enumerate(
                zip(
                    input_ids_sentence_splits,
                    logits_sentence_splits,
                    offset_mapping_sentence_splits,
                    special_tokens_mask_sentence_splits,
                )
            ):
                # use last batch completely
                if i == len(logits_sentence_splits) - 1:
                    overlap_ = 0

                non_special_indexes = special_tokens_mask_ != 1
                # num_non_overlap = len(input_ids_) - overlap_
                num_non_overlap = sum(non_special_indexes) - overlap_
                input_ids_sentence.append(input_ids_[non_special_indexes][:num_non_overlap])
                logits_sentence.append(logits_[non_special_indexes][:num_non_overlap])
                offset_mapping_sentence.append(offset_mapping_[non_special_indexes][:num_non_overlap])

            # sentences.append(model_outputs["sentence"][next(i for i, v in enumerate(indexes_) if v)])
            sentences.append(model_outputs["sentence"][np.argwhere(indexes_)[0][0]])
            aggregated_input_ids.append(np.concatenate(input_ids_sentence, axis=0))
            aggregated_logits.append(np.concatenate(logits_sentence, axis=0))
            aggregated_offset_mapping.append(np.concatenate(offset_mapping_sentence, axis=0))

        # aggregated_input_ids = np.stack(aggregated_input_ids, axis=0)
        # aggregated_logits = np.stack(aggregated_logits, axis=0)
        # aggregated_offset_mapping = np.stack(aggregated_offset_mapping, axis=0)

        return sentences, aggregated_input_ids, aggregated_logits, aggregated_offset_mapping

    def gather_pre_entities(
        self, sentence: str, input_ids: np.ndarray, scores: np.ndarray, offset_mapping: np.ndarray
    ) -> List[dict]:
        pre_entities = []

        last_word = None
        for idx, token_scores in enumerate(scores):
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            # normally there won't be None here
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                word_ref = sentence[start_ind:end_ind]
                # todo: This is a fallback heuristic. This will fail most likely on any kind of text + punctuation mixtures that will be considered "words". Non word aware models cannot do better than this unfortunately.
                # todo: "l' ue" gives unexpected token (l'", "▁", "ue") (not "_") whose offset mapping is wrong
                # todo: "ue" doesn't apply to HF rules, and we may need prediction of the token "▁" 
                # is_subword = start_ind > 0 and " " not in sentence[start_ind - 1 : start_ind + 1]
                is_subword = start_ind > 0 and " " not in sentence[start_ind - 1 : start_ind + 1] if last_word != "▁" else True
                # print(f"-{word}-{word_ref}-{start_ind}-{end_ind}-{sentence[start_ind - 1 : start_ind + 1]}-{is_subword}")

                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    # why set to False
                    # is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            # # todo: words "aujourd' hui" gives unexpected token "▁" (not "_") whose offset mapping is wrong
            # # if int(input_ids[idx]) == 21 and offset_mapping[idx][1] - offset_mapping[idx][0] == 1:
            # if word == "▁":
            #     continue
            pre_entities.append(pre_entity)

            last_word = word

        # debug
        # for e in pre_entities:
        #     print(e)
        # quit()

        return pre_entities

    def aggregate_word(
        self, entities: List[dict], aggregation_strategy: AggregationStrategy = AggregationStrategy.LAST
    ) -> dict:
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.LAST:
            scores = entities[-1]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity

    def aggregate_words(
        self, entities: List[dict], aggregation_strategy: AggregationStrategy = AggregationStrategy.LAST
    ) -> List[dict]:
        r"""Override tokens from a given word that disagree to force agreement on word boundaries."""
        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
                word_group = [entity]
        # Last item
        word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities

    def prediction_to_text(self, predictions: List[List[dict]]):
        new_sentences = []
        for predictions_ in predictions:
            new_sentence = ""
            for pred_word in predictions_:
                # todo
                pred_word_text = pred_word["word"] + TokenClassificationPredictor.punc_label2text.get(pred_word["entity"])
                new_sentence += pred_word_text + " "

            # todo: rule based correction
            # Append trailing period if doesnt exist.
            # if new_sentence[-1].isalnum():

            # new_sentence = new_sentence.strip()
            new_sentence = TokenClassificationPredictor.post_normalize_text(new_sentence)
            new_sentences.append(new_sentence)
        return new_sentences


def main(model_name_or_path):
    import evaluate
    import pandas as pd
    import sys
    from dataset_ner_punct import load_data_files

    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/camembert-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/camembert-large_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/xlm-roberta-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/xlm-roberta-large_ft"
    test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/test"
    # test_data_dir = "/home/bhuang/corpus/text/internal/punctuation/2022-10-05/data"

    tc = TokenClassificationPredictor(
        model_name_or_path,
        0,
        # batch_size=64,
        stride=100,
        overlap=100,
        aggregation_strategy=AggregationStrategy.LAST,
    )

    # csv_path = "/home/bhuang/corpus/text/internal/punctuation/2022-10-05/data.csv"
    # to_csv_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc_old/camembert-base-ft/res/preds.tsv"
    # df = pd.read_csv(csv_path, sep=";")
    # print(df.shape)
    # df.dropna(subset=["input", "reference"], how="any", inplace=True)
    # print(df.shape)
    # print(df.head())
    # df["hypothesis"] = df["input"].map(lambda x: tc([x])[0])
    # tmp_outdir = os.path.dirname(to_csv_path)
    # if not os.path.exists(tmp_outdir):
    #     os.makedirs(tmp_outdir)
    # df.to_csv(to_csv_path, index=False, sep="\t")
    # sys.path.append("/home/bhuang/my-scripts")
    # from myscripts.data.text.wer.get_alignment import stats_wer
    # # df["reference"] = df["reference"].map()
    # df["reference"] = df["reference"].str.lower()
    # refs = dict(zip(df["id"], [l.split() for l in df["reference"]]))
    # hyps = dict(zip(df["id"], [l.split() for l in df["hypothesis"]]))
    # stats_wer(refs, hyps, tmp_outdir)
    # quit()

    # word = ["président", "de", "la", "séance", "d'", "aujourd'", "hui", "pour", "confirmer"]
    # # word = ["président", "de", "la", "séance", "d'", "aujourd'hui", "pour", "confirmer"]
    # sentence = " ".join(word)
    # # sentence = "président de la séance d'aujourd'hui pour confirmer"
    # sentence = "d'accord effectivement le véhicule il est de deux mille le vingt-cinq janvier deux mille très bien donc j'ai toutes les informations nécessaires pour la déclaration de bris de glace pour la facturation donc y aura bien sûr la franchise de soixante de soixante euros à régler le jour de l'intervention la facture et la déclaration de bris de glace sera envoyé en même temps une fois l'intervention fini pardon à l'assurance et je vous rappelle le rendez vous donc du dix-neuf janvier à neuf heures trente à mireille lauze pour le remplacement de la lunette arrière sur le sujet qui est ce que vous avez des questions monsieur"
    # print(tc(sentence))
    # res = tc.predict(sentence)
    # print(tc.prediction_to_text(res)[0])
    # hypothese = [pred_["entity"] for pred_ in res[0]]
    # word_pred = [pred_["word"] for pred_ in res[0]]
    # # print(res[0])
    # for r in res[0]:
    #     print(r)
    # for w, w_p in zip(word, word_pred):
    #     print(w, w_p)
    # quit()

    # don't forget replacers cc
    # test_ds = load_data_files(test_data_dir, task_config="punc", replacers={"EXCLAMATION": "PERIOD"}, preprocessing_num_workers=16)
    test_ds = load_data_files(test_data_dir, task_config="eos", preprocessing_num_workers=16)
    # test_ds = load_data_files(test_data_dir, task_config="case", replacers={"OTHER": "LOWER"}, preprocessing_num_workers=16)

    def predict_(examples):
        sentences = [" ".join(words) for words in examples["word"]]
        # examples["hypothese"] = [[pred_["entity"] for pred_ in pred] for pred in tc.predict(sentences)]
        predictions = tc.predict(sentences)
        examples["hypothese"] = [[pred_["entity"] for pred_ in pred] for pred in predictions]
        examples["word_pred"] = [[pred_["word"] for pred_ in pred] for pred in predictions]
        return examples

    # test_ds = test_ds.select(range(10))
    test_ds = test_ds.map(predict_, batched=True, batch_size=128)
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

    tmp_outdir = f"{model_name_or_path}/results/predict_words"
    if not os.path.exists(tmp_outdir):
        os.makedirs(tmp_outdir)

    with open(f"{tmp_outdir}/references.txt", "w") as writer:
        for labels_ in test_ds["label"]:
            writer.write(" ".join(labels_) + "\n")

    with open(f"{tmp_outdir}/predictions.txt", "w") as writer:
        for prediction_ in test_ds["hypothese"]:
            writer.write(" ".join(prediction_) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
