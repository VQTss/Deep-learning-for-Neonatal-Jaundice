import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights


class EfficientNetB0Regression(nn.Module):
    """EfficientNet-B0 for Regression"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        print(f"EfficientNet-B0 - Pretrain: {pretrained}")
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = nn.Sequential(
            backbone.features,
            backbone.avgpool
        )
        self.fc = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(1)