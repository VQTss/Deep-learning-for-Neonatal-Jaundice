import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(backend: str = "nccl"):
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def wrap_ddp(model: torch.nn.Module, local_rank: int) -> DDP:

    return DDP(model, device_ids=[local_rank], output_device=local_rank)


def print_on_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)
