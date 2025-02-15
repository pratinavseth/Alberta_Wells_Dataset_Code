"""
U-Net Model with variouys backbones from SMP
"""
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.smp_model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=4,
            classes=1,
            activation='sigmoid',)

    def forward(self, x):
        return self.smp_model(x)