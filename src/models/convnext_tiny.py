import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights


class ConvNeXtTinyRegression(nn.Module):
    """ConvNeXt-Tiny for Regression"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        print(f"ConvNeXt-Tiny - Pretrain: {pretrained}")
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        self.features = nn.Sequential(
            backbone.features,
            backbone.avgpool
        )
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(1)