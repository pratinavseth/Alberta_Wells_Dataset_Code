import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        # Flatten the predictions and labels
        output_flat = output.view(-1)
        target_flat = target.view(-1)

        # Calculate the intersection and union
        intersection = torch.sum(output_flat * target_flat)
        union = torch.sum(output_flat) + torch.sum(target_flat)

        # Calculate the Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Calculate the Dice loss
        loss = 1.0 - dice
        return loss