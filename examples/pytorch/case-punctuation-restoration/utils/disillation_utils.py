# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Adapted from:
- https://www.philschmid.de/knowledge-distillation-bert-transformers
- https://lewtun.github.io/blog/weeknotes/nlp/huggingface/transformers/2021/01/17/wknotes-distillation-and-generation.html#fn-1
"""

from dataclasses import dataclass, field
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import TrainingArguments, Trainer


@dataclass
class DistillationTrainingArguments(TrainingArguments):
    alpha: float = field(default=0.5, metadata={"help": "Weight for distill loss."})
    temperature: float = field(default=2.0, metadata={"help": "Distillation temperature."})


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     Commpute loss by adding distill loss by soft target and student loss by hard target.

    #     General version.
    #     """
    #     # compute student output
    #     outputs_student = model(**inputs)
    #     loss_student = outputs_student.loss

    #     # compute teacher output
    #     with torch.no_grad():
    #         outputs_teacher = self.teacher(**inputs)

    #     # assert size
    #     assert outputs_student.logits.size() == outputs_teacher.logits.size()

    #     # Soften probabilities and compute distillation loss
    #     loss_fct = nn.KLDivLoss(reduction="batchmean")

    #     loss_logits = loss_fct(
    #         F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
    #         F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
    #     ) * (self.args.temperature**2)

    #     # Return weighted student loss
    #     loss = self.args.alpha * loss_logits + (1.0 - self.args.alpha) * loss_student

    #     return (loss, outputs_student) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Commpute loss by adding distill loss by soft target and student loss by hard target.

        Multi head version.
        """
        # compute student output
        outputs_student = model(**inputs)
        loss_student = outputs_student.loss
        case_loss_student = outputs_student.case_loss
        punc_loss_student = outputs_student.punc_loss
        case_logits_student = outputs_student.case_logits
        punc_logits_student = outputs_student.punc_logits

        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
            case_logits_teacher = outputs_teacher.case_logits
            punc_logits_teacher = outputs_teacher.punc_logits

        # assert size
        assert case_logits_student.size() == case_logits_teacher.size()
        assert punc_logits_student.size() == punc_logits_teacher.size()

        # Soften probabilities and compute distillation loss
        # 1. kl loss as distillation loss
        """
        loss_fct = nn.KLDivLoss(reduction="batchmean")

        case_loss = loss_fct(
            F.log_softmax(case_logits_student / self.args.temperature, dim=-1),
            F.softmax(case_logits_teacher / self.args.temperature, dim=-1),
        ) * (self.args.temperature**2)
        punc_loss = loss_fct(
            F.log_softmax(punc_logits_student / self.args.temperature, dim=-1),
            F.softmax(punc_logits_teacher / self.args.temperature, dim=-1),
        ) * (self.args.temperature**2)
        """

        # 2. ce loss as distillation loss ??
        """
        # loss_fct = nn.CrossEntropyLoss()
        # case_loss = loss_fct(case_logits_student.view(-1, self.num_case_labels), case_labels.view(-1))
        """

        # 3. mse loss as distillation loss
        loss_fct = nn.MSELoss()

        case_loss = loss_fct(case_logits_student, case_logits_teacher)
        punc_loss = loss_fct(punc_logits_student, punc_logits_teacher)


        # multi task
        # distil_loss = case_loss + punc_loss
        distil_loss = (case_loss + punc_loss) / 2.0

        # Return weighted student loss
        loss = self.args.alpha * distil_loss + (1.0 - self.args.alpha) * loss_student

        self.callback_handler.callbacks[2]._wandb.log(
            {
                "case_loss_student": case_loss_student,
                "punc_loss_student": punc_loss_student,
                "case_loss_ditill": case_loss,
                "punc_loss_ditill": punc_loss,
                "distil_loss": distil_loss,
                "student_loss": loss_student,
                "final_loss": loss,
            }
        )

        return (loss, outputs_student) if return_outputs else loss