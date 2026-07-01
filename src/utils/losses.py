import torch
import torch.nn as nn


def losses_function(name: str = "smoothl1", **kwargs) -> nn.Module:
    name = name.lower().replace("-", "_")

    if name in ("smoothl1", "smooth_l1", "huber"):
        return nn.SmoothL1Loss(**kwargs)
    if name in ("mse", "l2"):
        return nn.MSELoss()
    if name in ("l1", "mae"):
        return nn.L1Loss()

    raise ValueError(
        f"Unknown loss name: '{name}'. "
        f"Supported: smoothl1, mse, l1, logcosh, quantile"
    )