import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights


class MobileNetV3SmallRegression(nn.Module):
    """MobileNetV3-Small for Regression"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        print(f"MobileNetV3-Small - Pretrain: {pretrained}")
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = nn.Sequential(
            backbone.features,
            backbone.avgpool
        )
        self.fc = nn.Linear(576, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(1)