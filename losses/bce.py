"""
Binary Cross Entropy
"""
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from typing import Optional
from functools import partial

__all__ = ["BinaryCrossEntropy"]


class BinaryCrossEntropy(_Loss):
    def __init__(self):        
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss