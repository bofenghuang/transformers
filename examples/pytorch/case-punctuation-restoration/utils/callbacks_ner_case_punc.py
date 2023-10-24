#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import numpy as np
from typing import List, Optional, Union

from transformers.integrations import WandbCallback

def gather_pre_entities(
        tokenizer,
        sentence: str,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        case_scores: np.ndarray,
        punc_scores: np.ndarray,
        offset_mapping: np.ndarray,
    ) -> List[dict]:
        pre_entities = []

        last_word = None
        for idx, (input_id_, attention_mask_) in enumerate(input_ids, attention_mask):

            if attention_mask_ == 0:
                break

            word = tokenizer.convert_ids_to_tokens(int(input_id_))
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

                if int(input_id_) == tokenizer.unk_token_id:
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
                "punc_scores": punc_scores[idx],
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


class WandbProgressResultsCallback(WandbCallback):
    def __init__(self, trainer, sample_dataset): 
        super().__init__()

        self.trainer = trainer
        self.sample_dataset = sample_dataset
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        super().on_log(args, state, control, model, logs)

        predictions = self.trainer.predict(self.sample_dataset)
        case_logits, punc_logits = predictions.predictions

        case_maxes = np.max(case_logits, axis=-1, keepdims=True)
        case_shifted_exp = np.exp(case_logits - case_maxes)
        case_scores = case_shifted_exp / case_shifted_exp.sum(axis=-1, keepdims=True)

        punc_maxes = np.max(punc_logits, axis=-1, keepdims=True)
        punc_shifted_exp = np.exp(punc_logits - punc_maxes)
        punc_scores = punc_shifted_exp / punc_shifted_exp.sum(axis=-1, keepdims=True)

        for example, case_scores_, punc_scores_ in zip(self.sample_dataset, case_scores, punc_scores):
            sentence = example["text"]
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            
            



        predictions = decode_predictions(self.trainer, predictions)
        measures_df = compute_measures(predictions, self.records_df["sentence"].tolist())
        records_df = pd.concat([self.records_df, measures_df], axis=1)
        records_df["prediction"] = predictions
        records_df["step"] = state.global_step
        records_table = self._wandb.Table(dataframe=records_df)
        self._wandb.log({"sample_predictions": records_table})
        
    # def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
