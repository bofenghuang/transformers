#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class CasePuncOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    # todo
    # case_loss: Optional[torch.FloatTensor] = None
    # punc_loss: Optional[torch.FloatTensor] = None
    case_logits: torch.FloatTensor = None
    punc_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaForCasePunc(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_case_labels, num_punc_labels):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.num_case_labels = num_case_labels
        self.num_punc_labels = num_punc_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.case_classifier = nn.Linear(config.hidden_size, num_case_labels)
        self.punc_classifier = nn.Linear(config.hidden_size, num_punc_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        case_labels: Optional[torch.LongTensor] = None,
        punc_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CasePuncOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)
        case_logits = self.case_classifier(sequence_output)
        punc_logits = self.punc_classifier(sequence_output)

        # logits = torch.stack([case_logits, punc_logits], dim=2)

        loss = None
        # todo
        # case_loss = None
        # punc_loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if case_labels is not None and punc_labels is not None:
            # CE
            loss_fct = CrossEntropyLoss()
            # Focal loss
            # from .losses import MultiClassFocalLoss
            # loss_fct = MultiClassFocalLoss(gamma=2.0)
            # CB CE
            # from .losses import ClassBalancedCrossEntroy
            # loss_fct_punc = ClassBalancedCrossEntroy(num_samples=self.num_samples_punc, beta=0.99)

            # coeffs = 1 / np.sqrt(self.num_samples_punc)
            # coeffs = coeffs / coeffs.sum()
            # coeffs = torch.from_numpy(coeffs).float().cuda()

            # weighted loss for punctuation
            # class_weights = [0.02325581, 0.23255814, 0.02325581, 0.23255814, 0.23255814, 0.02325581, 0.23255814]
            # class_weights = [1, 10, 1, 10, 10, 1, 10]
            # class_weights = [1, 2, 1, 2, 2, 1, 2]
            # class_weights = [1, 1, 1, 2, 1, 1, 1]
            class_weights = [1, 1, 2, 1, 1, 1]
            class_weights = np.asarray(class_weights)
            class_weights = torch.from_numpy(class_weights).float().cuda()

            loss_fct_punc = CrossEntropyLoss(weight=class_weights)

            case_loss = loss_fct(case_logits.view(-1, self.num_case_labels), case_labels.view(-1))
            # punc_loss = loss_fct(punc_logits.view(-1, self.num_punc_labels), punc_labels.view(-1))
            punc_loss = loss_fct_punc(punc_logits.view(-1, self.num_punc_labels), punc_labels.view(-1))

            # loss = case_loss + punc_loss
            loss = (case_loss + punc_loss) / 2.0
            # todo: weighted sum
            # loss = alpha * case_loss + (1 - alpha) * punc_loss

        if not return_dict:
            # output = (logits,) + outputs[2:]
            output = (case_logits, punc_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CasePuncOutput(
            loss=loss,
            # todo
            # case_loss=case_loss,
            # punc_loss=punc_loss,
            # logits=logits,
            case_logits=case_logits,
            punc_logits=punc_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    model = RobertaForCasePunc.from_pretrained("camembert-base", num_case_labels=3, num_punc_labels=4)
    input_tensors = torch.randint(10000, (1, 32))
    case_labels = torch.randint(3, (1, 32))
    punc_labels = torch.randint(4, (1, 32))
    outputs = model(input_tensors, case_labels=case_labels, punc_labels=punc_labels)
    print(outputs)
