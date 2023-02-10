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

from model_ner_punct_multiple_choice import RobertaForRecasePunct
from normalize_text import TextNormalizer


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


def remove_symbols(s: str, keep=""):
    r"""Replace any other markers, symbols, punctuations with a space, keeping diacritics."""
    return "".join(
        c if c in keep else " " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s)
    )


class TokenClassificationPredictor:

    threshold_case_capitalization = 0.6
    case_label2text_functions = {
        "CAPITALIZE": lambda x, score: x.lower().capitalize()
        if score > TokenClassificationPredictor.threshold_case_capitalization
        else x.lower(),
        "LOWER": lambda x, _: x.lower(),
        "UPPER": lambda x, _: x.upper(),
    }
    # punc_label2text = {"0": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}
    # todo: add threshold
    threshold_punctuation = -1
    punct_label2text_functions = {
        "0": lambda _: "",
        "COMMA": lambda score: "," if score > TokenClassificationPredictor.threshold_punctuation else "",
        "PERIOD": lambda score: "." if score > TokenClassificationPredictor.threshold_punctuation else "",
        # todo multilingual
        "QUESTION": lambda score: " ?" if score > TokenClassificationPredictor.threshold_punctuation else "",
    }

    def __init__(
        self,
        model_name_or_path: str,
        device: Union[torch.device, str, int] = "cpu",
        normalizer_file: Optional[str] = None,
        **kwargs
    ) -> None:
        # load config
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        # todo: hf change this somewhere is from_pretrained()
        self.config.case_id2label = {int(k): v for k, v in self.config.case_id2label.items()}
        self.config.punc_id2label = {int(k): v for k, v in self.config.punc_id2label.items()}
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # NB: only for camembert large
        # if model_name_or_path == "camembert/camembert-large":
        # this model has a wrong maxlength value, so we need to set it manually
        self.tokenizer.model_max_length = 512
        # device
        self.device = self.configure_device(device)
        # load model
        # todo: onnx
        # model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        model = RobertaForRecasePunct.from_pretrained(
            model_name_or_path,
            num_case_labels=len(self.config.case_id2label.keys()),
            num_punc_labels=len(self.config.punc_id2label.keys()),
        )
        model.eval()
        self.model = model.to(self.device)

        # load normalization patterns
        self.text_normalizer = TextNormalizer(normalizer_file) if normalizer_file is not None else None

        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)

    def _sanitize_parameters(
        self,
        do_pre_normalize: bool = False,
        max_length: Optional[int] = None,
        stride: Optional[int] = None,
        batch_size: Optional[int] = None,
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
        # use the same value for stride and overlap
        if stride is not None:
            postprocess_params["overlap"] = stride
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

    def pre_normalize_text(self, s: str):
        s = s.lower()  # remove existing case

        # use the same standarization when preparing training data
        s = re.sub(r"(?<=\w)\s+'", r"'", s)  # standardize when there's a space before an apostrophe
        s = re.sub(r"(?<!aujourd)(?<=\w)'\s*(?=\w)", "' ", s)  # add an espace after an apostrophe (except "aujourd")

        s = remove_symbols(s, keep="'-")  # remove existing punct

        s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space
        return s

    def post_normalize_text(self, s: str):

        if self.text_normalizer is not None:
            s = self.text_normalizer(s)

        # todo
        s = re.sub(r"\s*'\s*", "'", s)  # standardize when there's a space before/after an apostrophe
        s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space
        return s

    def preprocess(
        self, inputs: Union[str, List[str]], do_pre_normalize: bool = False, max_length: Optional[int] = None, stride: int = 0
    ):
        if isinstance(inputs, str):
            inputs = [inputs]

        if do_pre_normalize:
            inputs = [self.pre_normalize_text(inputs_) for inputs_ in inputs]

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
            # logits = outputs[0]

        return {
            # "logits": logits.to(torch.device("cpu")),
            "case_logits": outputs["case_logits"].to(torch.device("cpu")),
            "punc_logits": outputs["punc_logits"].to(torch.device("cpu")),
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

        return model_outputs

    def postprocess(
        self, model_outputs, overlap: int = 0, aggregation_strategy: AggregationStrategy = AggregationStrategy.LAST
    ):
        # convert tensors to np arrays
        model_outputs = {k: v.numpy() if torch.is_tensor(v) else v for k, v in model_outputs.items()}

        # debug
        # for input_ids_, case_logits_, punc_logits_ in zip(model_outputs["input_ids"], model_outputs["case_logits"], model_outputs["punc_logits"]):
        #     for i, c_l, p_l in zip(input_ids_, case_logits_, punc_logits_):
        #         print(self.tokenizer.convert_ids_to_tokens([i]), c_l, p_l)
        #     print("\n")
        # quit()

        # gather splitted sub results
        sentences, input_ids, case_logits, punc_logits, offset_mapping = self.gather_sub_results(model_outputs, overlap)
        # print(self.tokenizer.decode(input_ids[0]))
        # quit()

        entities = []
        for sentence_, input_ids_, case_logits_, punc_logits_, offset_mapping_ in zip(
            sentences, input_ids, case_logits, punc_logits, offset_mapping
        ):

            case_maxes = np.max(case_logits_, axis=-1, keepdims=True)
            case_shifted_exp = np.exp(case_logits_ - case_maxes)
            case_scores_ = case_shifted_exp / case_shifted_exp.sum(axis=-1, keepdims=True)

            punc_maxes = np.max(punc_logits_, axis=-1, keepdims=True)
            punc_shifted_exp = np.exp(punc_logits_ - punc_maxes)
            punct_scores_ = punc_shifted_exp / punc_shifted_exp.sum(axis=-1, keepdims=True)

            # debug
            # for input_i_, case_s_, punc_s_ in zip(input_ids_, case_scores_, punct_scores_):
            #     print(input_i_, self.tokenizer.convert_ids_to_tokens([input_i_]), case_s_, punc_s_)
            # quit()

            pre_entities_ = self.gather_pre_entities(sentence_, input_ids_, case_scores_, punct_scores_, offset_mapping_)
            entities_ = self.aggregate_words(pre_entities_, aggregation_strategy)

            entities.append(entities_)

        # todo: group entities if across words

        return entities

    def gather_sub_results(self, model_outputs, overlap: int = 0):
        sentences = []
        aggregated_input_ids = []
        aggregated_case_logits = []
        aggregated_punc_logits = []
        aggregated_offset_mapping = []

        # sorted unique
        for sentence_id in np.unique(model_outputs["overflow_to_sample_mapping"]):
            indexes_ = model_outputs["overflow_to_sample_mapping"] == sentence_id
            input_ids_sentence_splits = model_outputs["input_ids"][indexes_]
            case_logits_sentence_splits = model_outputs["case_logits"][indexes_]
            punc_logits_sentence_splits = model_outputs["punc_logits"][indexes_]
            offset_mapping_sentence_splits = model_outputs["offset_mapping"][indexes_]
            special_tokens_mask_sentence_splits = model_outputs["special_tokens_mask"][indexes_]

            overlap_ = overlap

            input_ids_sentence = []
            case_logits_sentence = []
            punc_logits_sentence = []
            offset_mapping_sentence = []
            for i, (input_ids_, case_logits_, punc_logits_, offset_mapping_, special_tokens_mask_) in enumerate(
                zip(
                    input_ids_sentence_splits,
                    # logits_sentence_splits,
                    case_logits_sentence_splits,
                    punc_logits_sentence_splits,
                    offset_mapping_sentence_splits,
                    special_tokens_mask_sentence_splits,
                )
            ):
                # use last batch completely
                # if i == len(logits_sentence_splits) - 1:
                if i == len(case_logits_sentence_splits) - 1:
                    overlap_ = 0

                non_special_indexes = special_tokens_mask_ != 1
                # num_non_overlap = len(input_ids_) - overlap_
                num_non_overlap = sum(non_special_indexes) - overlap_
                input_ids_sentence.append(input_ids_[non_special_indexes][:num_non_overlap])
                case_logits_sentence.append(case_logits_[non_special_indexes][:num_non_overlap])
                punc_logits_sentence.append(punc_logits_[non_special_indexes][:num_non_overlap])
                offset_mapping_sentence.append(offset_mapping_[non_special_indexes][:num_non_overlap])

            # sentences.append(model_outputs["sentence"][next(i for i, v in enumerate(indexes_) if v)])
            sentences.append(model_outputs["sentence"][np.argwhere(indexes_)[0][0]])
            aggregated_input_ids.append(np.concatenate(input_ids_sentence, axis=0))
            aggregated_case_logits.append(np.concatenate(case_logits_sentence, axis=0))
            aggregated_punc_logits.append(np.concatenate(punc_logits_sentence, axis=0))
            aggregated_offset_mapping.append(np.concatenate(offset_mapping_sentence, axis=0))

        # aggregated_input_ids = np.stack(aggregated_input_ids, axis=0)
        # aggregated_logits = np.stack(aggregated_logits, axis=0)
        # aggregated_offset_mapping = np.stack(aggregated_offset_mapping, axis=0)

        return sentences, aggregated_input_ids, aggregated_case_logits, aggregated_punc_logits, aggregated_offset_mapping

    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        case_scores: np.ndarray,
        punct_scores: np.ndarray,
        offset_mapping: np.ndarray,
    ) -> List[dict]:
        pre_entities = []

        last_word = None
        for idx, input_id in enumerate(input_ids):
            word = self.tokenizer.convert_ids_to_tokens(int(input_id))
            # normally there won't be None here
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                word_ref = sentence[start_ind:end_ind]
                # todo: This is a fallback heuristic. This will fail most likely on any kind of text + punctuation mixtures that will be considered "words". Non word aware models cannot do better than this unfortunately.
                # is_subword = start_ind > 0 and " " not in sentence[start_ind - 1 : start_ind + 1]
                # todo: "l' ue" gives unexpected token (l'", "▁", "ue") (not "_") whose offset mapping is wrong
                # todo: "ue" doesn't apply to HF rules, and we may need prediction of the token "▁"
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
                "case_scores": case_scores[idx],
                "punct_scores": punct_scores[idx],
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)

            last_word = word

        # debug
        # i = -1
        # for e in pre_entities:
        #     if not e["is_subword"]:
        #         i += 1
        #     print(i, e)
        # quit()

        return pre_entities

    def aggregate_prediction(
        self, entities: List[dict], entity_score_name: str, aggregation_strategy: AggregationStrategy, id2label: dict
    ):
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0][entity_score_name]
            idx = scores.argmax()
            score = scores[idx]
            entity = id2label[idx]
        elif aggregation_strategy == AggregationStrategy.LAST:
            scores = entities[-1][entity_score_name]
            idx = scores.argmax()
            score = scores[idx]
            entity = id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity[entity_score_name].max())
            scores = max_entity[entity_score_name]
            idx = scores.argmax()
            score = scores[idx]
            entity = id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity[entity_score_name] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")

        return entity, score

    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])

        # todo
        case_entity, case_score = self.aggregate_prediction(
            entities, "case_scores", AggregationStrategy.FIRST, self.config.case_id2label
        )
        punct_entity, punct_score = self.aggregate_prediction(
            entities, "punct_scores", AggregationStrategy.LAST, self.config.punc_id2label
        )

        new_entity = {
            "case_entity": case_entity,
            "case_score": case_score,
            "punct_entity": punct_entity,
            "punct_score": punct_score,
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
                recase_funct_ = self.case_label2text_functions.get(pred_word["case_entity"])
                repunct_funct_ = self.punct_label2text_functions.get(pred_word["punct_entity"])
                pred_word_text = recase_funct_(pred_word["word"], pred_word["case_score"]) + repunct_funct_(
                    pred_word["punct_score"]
                )
                new_sentence += pred_word_text + " "

            # todo: inverse label for es
            # todo: rule based correction
            # Append trailing period if doesnt exist.
            # if new_sentence[-1].isalnum():

            # post normalize text
            new_sentence = self.post_normalize_text(new_sentence)
            new_sentences.append(new_sentence)
        return new_sentences


# def main(model_name_or_path, test_data_dir, tmp_outdir):
def main():
    import evaluate
    import pandas as pd
    import sys

    from dataset_ner_punct_multiple_choice import load_data_files

    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/camembert-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/camembert-large_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/xlm-roberta-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/punc/xlm-roberta-large_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc/camembert-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc/xlm-roberta-base_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc-multilingual/xlm-roberta-large_ft"
    # model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc-multilingual_plus/xlm-roberta-base_ft"
    model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc-multilingual_plus/xlm-roberta-large_ft"
    # test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/test"
    # test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/tmp"
    # test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/en_europarl/data/test"
    # test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/es_europarl/data/test"
    # test_data_dir = "/projects/bhuang/corpus/text/flaubert/raw/de_europarl/data/test"
    # test_data_dir = "/home/bhuang/corpus/text/internal/punctuation/2022-10-05/data"

    tc = TokenClassificationPredictor(
        model_name_or_path,
        device=0,
        normalizer_file="./normalizer.json",
        do_pre_normalize=True,
        stride=100,
        # batch_size=64,
        # aggregation_strategy=AggregationStrategy.LAST,
    )

    # Inference CSV file
    # # input_file = "/home/bhuang/corpus/text/internal/punctuation/2022-11-29/data_dekuple_200.csv"
    # # output_file = "/home/bhuang/corpus/text/internal/punctuation/2022-11-29/data_dekuple_200_generated.csv"
    # input_file = "/home/bhuang/corpus/text/internal/punctuation/2022-11-29/data_carglass_200.csv"
    # output_file = "/home/bhuang/corpus/text/internal/punctuation/2022-11-29/data_carglass_200_generated.csv"
    # df = pd.read_csv(input_file, sep="\t")
    # print(df.head())
    # # df = df.head(5)
    # # df["wrd"] = df["wrd"].map(lambda x: tc(x)[0])
    # df["text"] = df["text"].map(lambda x: tc(x)[0])
    # print(df.head())
    # # df[["ID", "wrd"]].to_csv(output_file, index=False, sep="\t")
    # df[["utt", "text"]].to_csv(output_file, index=False, sep="\t")
    # quit()

    # Alignments
    # sys.path.append("/home/bhuang/my-scripts")
    # from myscripts.data.text.wer.get_alignment import stats_wer
    # df["reference"] = df["reference"].map()
    # df["reference"] = df["reference"].str.lower()
    # refs = dict(zip(df["id"], [l.split() for l in df["reference"]]))
    # hyps = dict(zip(df["id"], [l.split() for l in df["hypothesis"]]))
    # stats_wer(refs, hyps, tmp_outdir)
    # quit()

    # Single sentence
    # sentence = "président de la séance d'aujourd'hui pour confirmer"
    # sentence = "politiques étrangères de l' ue seront ce les"
    # sentence = "d'accord effectivement le véhicule il est de deux mille le vingt-cinq janvier deux mille très bien donc j'ai toutes les informations nécessaires pour la déclaration de bris de glace pour la facturation donc y aura bien sûr la franchise de soixante de soixante euros à régler le jour de l'intervention la facture et la déclaration de bris de glace sera envoyé en même temps une fois l'intervention fini pardon à l'assurance et je vous rappelle le rendez vous donc du dix-neuf janvier à neuf heures trente à mireille lauze pour le remplacement de la lunette arrière sur le sujet qui est ce que vous avez des questions monsieur"
    # sentence = "bonjour j'aimerais savoir quelle est la réponse quelle était la question déjà"
    # sentence = "ca va"
    # sentence = "what's up bro"
    # sentence = "bonjour comment ca va"
    # res = tc.predict(sentence)
    # print(tc.prediction_to_text(res)[0])
    # for r in res[0]:
    #     print(r)
    # quit()

    # Multiple sentences
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
    quit()

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


if __name__ == "__main__":
    fire.Fire(main)
