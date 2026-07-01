import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
)

def get_scheduler(optimizer, name: str = "cosine", **kwargs):
    name = name.lower() if name else "none"

    if name in ("none", None):
        return None

    elif name == "step":
        return StepLR(optimizer, **kwargs)

    elif name == "multistep":
        return MultiStepLR(optimizer, **kwargs)

    elif name in ("cosine", "cosineannealing"):
        return CosineAnnealingLR(optimizer, **kwargs)

    elif name in ("cosine_warm", "cosinewarm"):
        return CosineAnnealingWarmRestarts(optimizer, **kwargs)

    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, **kwargs)

    else:
        raise ValueError(f"Unknown scheduler: '{name}'")