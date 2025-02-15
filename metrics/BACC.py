from sklearn.metrics import balanced_accuracy_score
import warnings
import numpy as np
from .base import Metric
import torch

def calc_BalancedAccuracy(truth, pred, threshold=0.5, **kwargs):
    # Ensure inputs are PyTorch tensors
    if not torch.is_tensor(truth):
        truth = torch.tensor(truth)
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred)

    # Flatten the arrays
    pred = (pred > threshold).int()
    gt = truth.flatten().int()
    pd = pred.flatten().int()

    # Compute BACC via scikit-learn
    bacc = balanced_accuracy_score(gt.cpu().numpy(), pd.cpu().numpy())

    # Return BACC score
    return torch.tensor([bacc], dtype=torch.double).to(pred.device).mean()


class BACC(Metric):
    __name__ = "BACC_Custom"

    def __init__(self, eps=1e-15, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold

    def forward(self, y_pr, y_gt):
        return calc_BalancedAccuracy(
            pred=y_pr,
            truth=y_gt,
            eps=self.eps,
            threshold=self.threshold,
        )