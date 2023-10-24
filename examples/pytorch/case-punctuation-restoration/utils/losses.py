# coding=utf-8
# Copyright 2022  Bofeng Huang

import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiClassFocalLoss(nn.Module):
    r"""Focal loss for multi-class classification.
    Adapted from
        - https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
        - https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
        - https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super(MultiClassFocalLoss, self).__init__()
        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            torch.pow((1.0 - prob), self.gamma) * log_prob, target_tensor, weight=self.weight, reduction=self.reduction
        )


class ClassBalancedCrossEntroy(nn.Module):
    r"""Implementation of the class-balanced loss as described in `"Class-Balanced Loss Based on Effective Number
    of Samples" <https://arxiv.org/pdf/1901.05555.pdf>`_.
    Given a loss function :math:`\mathcal{L}`, the class-balanced loss is described by:
    .. math::
        CB(p, y) = \frac{1 - \beta}{1 - \beta^{n_y}} \mathcal{L}(p, y)
    where :math:`p` is the predicted probability for class :math:`y`, :math:`n_y` is the number of training
    samples for class :math:`y`, and :math:`\beta` is exponential factor.

    Adapted from:
        - https://github.com/vandit15/Class-balanced-loss-pytorch/blob/921ccb8725b1eb0903b2c22a1a752a594fcae138/class_balanced_loss.py#L73
        - https://github.com/fcakyon/balanced-loss/blob/main/balanced_loss/losses.py
    """

    def __init__(
        self, num_samples: Union[np.ndarray, torch.Tensor], beta: float = 0.99, reduction: str = "mean"
    ) -> None:
        super().__init__()

        self.beta = beta
        self.reduction = reduction

        self.num_classes = num_samples.shape[0]

        # effective number
        cb_weights = (1 - beta) / (1 - beta**num_samples)
        # normalize
        cb_weights = cb_weights / cb_weights.sum() * self.num_classes
        self.cb_weights = cb_weights
        print(f"CB weights have been set to {self.cb_weights}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # See more details in reduction of CE
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        cb_weights = torch.as_tensor(self.cb_weights, device=logits.device).float()

        print(logits.shape)
        print(targets_one_hot.shape)
        print(cb_weights.shape)
        print(cb_weights)

        return F.cross_entropy(input=logits, target=targets_one_hot, weight=cb_weights, reduction=self.reduction)

    def __repr__(self) -> str:
        # return f"{self.__class__.__name__}({self.criterion.__repr__()}, beta={self.beta})"
        return f"{self.__class__.__name__}(beta={self.beta})"
