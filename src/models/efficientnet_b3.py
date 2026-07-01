import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights


class EfficientNetB3Regression(nn.Module):
    """EfficientNet-B3 for Regression"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        print(f"EfficientNet-B3 - Pretrain: {pretrained}")
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        self.features = nn.Sequential(
            backbone.features,
            backbone.avgpool
        )
        self.fc = nn.Linear(1536, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(1)