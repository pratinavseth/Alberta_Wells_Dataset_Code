import segmentation_models_pytorch as smp
from .bce import BinaryCrossEntropy
from .dice import DiceLoss
from .focal import FocalLoss
import torch.nn as nn


def get_lossfn(config):
    if config.loss_type == "Tversky":
        loss = smp.losses.TverskyLoss(mode="binary")
    elif config.loss_type == "Jaccard":
        loss = smp.losses.JaccardLoss(mode="binary")
    elif config.loss_type == "Dice":
        loss = smp.losses.DiceLoss(mode="binary")        
    elif config.loss_type == "Focal":
        loss = smp.losses.FocalLoss(mode="binary",gamma=config.gamma)
        loss.__name__ = "Focal Loss"
    elif config.loss_type == "Focal-MC":
        loss = smp.losses.FocalLoss(mode="multiclass")
        loss.__name__ = "Focal Loss"
    elif config.loss_type == "Lovasz":
        loss = smp.losses.LovaszLoss(mode="binary")    
    elif config.loss_type == "SoftBCEWithLogits":
        loss = smp.losses.SoftBCEWithLogitsLoss()
    elif config.loss_type == "SoftCrossEntropy":
        loss = smp.losses.SoftCrossEntropyLoss()
    elif config.loss_type == "MCC":
        loss = smp.losses.MCCLoss()
    elif config.loss_type == "BCE":
        loss = BinaryCrossEntropy()
    elif config.loss_type == "Focal-Custom":
        loss = FocalLoss(alpha=config.alpha,gamma=config.gamma)
    elif config.loss_type == "DiceLoss-Custom":
        loss = DiceLoss()
    elif config.loss_type == "CE_Torch":
        loss = nn.CrossEntropyLoss()
    return loss