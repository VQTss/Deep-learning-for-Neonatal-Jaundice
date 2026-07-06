"""Multi-branch Regression model.

Mỗi branch là 1 backbone 3-channel nhận 1 color space riêng.
Features từ các branch được concat theo channel dim, qua MLP head.

Cùng kiến trúc backbone cho mọi branch (vd 2 ResNet18 hoặc 2 EfficientNet-B0).
Mỗi branch giữ pretrained weights riêng (không share weight) vì input khác nhau.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


# ==================== Backbone Builders ====================
# Mỗi builder trả về (features_module, out_dim, layer_name_prefix)

def _build_resnet18(pretrained: bool):
    w = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    bb = tv_models.resnet18(weights=w)
    # Trả về features (all layers trừ FC cuối) + out_dim
    features = nn.Sequential(*list(bb.children())[:-1])  # global avgpool output: (B, 512, 1, 1)
    return features, 512


def _build_convnext_tiny(pretrained: bool):
    w = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    bb = tv_models.convnext_tiny(weights=w)
    features = nn.Sequential(bb.features, bb.avgpool)
    return features, 768


def _build_efficientnet_b0(pretrained: bool):
    w = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    bb = tv_models.efficientnet_b0(weights=w)
    features = nn.Sequential(bb.features, bb.avgpool)
    return features, 1280


def _build_efficientnet_b3(pretrained: bool):
    w = tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    bb = tv_models.efficientnet_b3(weights=w)
    features = nn.Sequential(bb.features, bb.avgpool)
    return features, 1536


def _build_mobilenetv3_small(pretrained: bool):
    w = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    bb = tv_models.mobilenet_v3_small(weights=w)
    features = nn.Sequential(bb.features, bb.avgpool)
    return features, 576


BACKBONE_BUILDERS = {
    "resnet18": (_build_resnet18, 512),
    "convnext_tiny": (_build_convnext_tiny, 768),
    "efficientnet_b0": (_build_efficientnet_b0, 1280),
    "efficientnet_b3": (_build_efficientnet_b3, 1536),
    "mobilenetv3_small": (_build_mobilenetv3_small, 576),
}


# ==================== MultiBranchRegression ====================

class MultiBranchRegression(nn.Module):
    """N parallel backbones, concat features, MLP head.

    Args:
        spaces: List các color spaces (len=N), phải khớp với số channels input (N*3).
        backbone_name: Tên backbone (vd "resnet18"). Mọi branch dùng cùng backbone.
        pretrained: Load ImageNet pretrained weights cho mỗi branch.
        head_hidden: Hidden dim của MLP head (default 256).
        dropout: Dropout rate trong head (default 0.2).
    """

    def __init__(
        self,
        spaces: list,
        backbone_name: str,
        pretrained: bool = True,
        head_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if backbone_name not in BACKBONE_BUILDERS:
            raise ValueError(f"Unknown backbone: {backbone_name}. Valid: {list(BACKBONE_BUILDERS.keys())}")
        if not spaces or len(spaces) < 2:
            raise ValueError(f"MultiBranchRegression cần >=2 spaces, got: {spaces}")

        self.spaces = list(spaces)
        self.backbone_name = backbone_name
        builder, out_dim = BACKBONE_BUILDERS[backbone_name]

        # Mỗi branch là 1 instance riêng (không share weight vì input khác nhau)
        self.backbones = nn.ModuleList([builder(pretrained)[0] for _ in self.spaces])
        self.out_dim = out_dim
        self.n_branches = len(self.spaces)

        # Head: MLP đơn giản
        self.head = nn.Sequential(
            nn.Linear(self.n_branches * out_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        print(
            f"MultiBranchRegression: {self.n_branches} branches x {backbone_name} "
            f"(pretrained={pretrained}) | per-branch out_dim={out_dim} | "
            f"head_hidden={head_hidden}, dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape (B, N*3, H, W) - mỗi nhóm 3 channels là 1 color space.
        Returns:
            Tensor shape (B,) - regression output.
        """
        assert x.shape[1] == self.n_branches * 3, (
            f"Expected {self.n_branches * 3} channels, got {x.shape[1]}"
        )

        # Tách channels cho mỗi branch
        branch_features = []
        for i, backbone in enumerate(self.backbones):
            x_branch = x[:, i * 3:(i + 1) * 3, :, :]  # (B, 3, H, W)
            feat = backbone(x_branch)  # (B, out_dim, 1, 1) cho hầu hết
            feat = torch.flatten(feat, 1)  # (B, out_dim)
            branch_features.append(feat)

        # Concat features từ tất cả branches
        fused = torch.cat(branch_features, dim=1)  # (B, N * out_dim)

        out = self.head(fused)  # (B, 1)
        return out.squeeze(1)