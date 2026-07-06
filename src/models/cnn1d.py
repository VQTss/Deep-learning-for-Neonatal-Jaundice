"""1D-CNN Regression model.

Biến ảnh 2D (H, W, C) thành flattened 1D vector, sau đó dùng 1D convolutions
để học hierarchical patterns từ flattened pixel features.
So với 2D CNN backbone:
- 1D convolutions chỉ học local patterns dọc theo flattened sequence.
- Không có spatial inductive bias (rotation/translation invariance yếu hơn).
- Phù hợp khi muốn khám phá features không gian theo chiều flattened.
"""
import torch
import torch.nn as nn


class CNN1DRegression(nn.Module):
    """1D-CNN Regression.

    Input: (B, C, H, W) — ảnh RGB/channels bất kỳ.
    Pipeline:
        1. Flatten thành (B, C*H*W) — coi như 1D sequence có C*H*W timesteps.
        2. Reshape thành (B, 1, C*H*W) — single-channel 1D signal.
        3. 1D conv layers với ReLU + BatchNorm + Dropout.
        4. GAP over time dimension → (B, n_filters_last).
        5. MLP head → scalar regression output.

    Args:
        in_channels: Số channels đầu vào (default 3 cho RGB).
        pretrained: Không áp dụng cho 1DCNN (placeholder, luôn False).
        hidden_dims: Danh sách hidden channels cho từng 1D conv layer.
                     Default [512, 256, 128, 64] tạo 4 conv blocks.
        kernel_sizes: Kernel size cho từng conv layer (default [7, 5, 3, 3]).
        dropout: Dropout rate (default 0.3).
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,  # 1DCNN không có pretrained weights
        hidden_dims: list = None,
        kernel_sizes: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3, 3]

        assert len(hidden_dims) == len(kernel_sizes), (
            f"hidden_dims and kernel_sizes must have same length, "
            f"got {len(hidden_dims)} vs {len(kernel_sizes)}"
        )

        self.in_channels = in_channels
        self.pretrained = pretrained  # placeholder

        # Build 1D conv blocks
        conv_blocks = []
        in_ch = 1  # flattened thành single-channel signal

        for i, (out_ch, ks) in enumerate(zip(hidden_dims, kernel_sizes)):
            conv_blocks.append(self._make_conv_block(in_ch, out_ch, ks, dropout, is_last=(i == len(hidden_dims) - 1)))
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.n_filters_last = hidden_dims[-1]

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(self.n_filters_last, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self._init_weights()

        print(
            f"CNN1DRegression: in_channels={in_channels} | "
            f"hidden_dims={hidden_dims} | kernel_sizes={kernel_sizes} | "
            f"dropout={dropout}"
        )

    def _make_conv_block(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dropout: float,
        is_last: bool = False,
    ) -> nn.Module:
        """1D conv block: Conv1d → BatchNorm → ReLU → Dropout.
        Skip BatchNorm ở layer cuối nếu muốn raw features cho head.
        """
        if is_last:
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, C, H, W)
        Returns:
            Tensor (B,) — regression output
        """
        B, C, H, W = x.shape

        # Flatten ảnh thành 1D sequence
        x = x.view(B, C * H * W)           # (B, C*H*W)
        x = x.unsqueeze(1)                  # (B, 1, C*H*W)

        # 1D conv blocks
        x = self.conv_blocks(x)             # (B, n_filters_last, 1)
        x = x.squeeze(-1)                    # (B, n_filters_last)

        # MLP head
        x = self.head(x)                     # (B, 1)
        return x.squeeze(1)                  # (B,)


# ==================== Model Registry ====================
# Đăng ký vào train.py get_model() bằng tên "cnn1d"
# =========================================================
# Ví dụ sử dụng:
#   model = CNN1DRegression(in_channels=3, pretrained=False, hidden_dims=[512,256,128,64])
# Hoặc trong train.py thêm:
#   elif model_name == "cnn1d":
#       from models.cnn1d import CNN1DRegression
#       return CNN1DRegression(in_channels=3)
# =========================================================
