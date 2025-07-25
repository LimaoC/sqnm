import torch
from torch import Tensor


def unflatten(flat_vec: Tensor, param_shapes: dict) -> dict:
    out = dict()
    offset = 0
    for name, shape in param_shapes.items():
        numel = torch.prod(torch.tensor(shape)).item()
        out[name] = flat_vec[offset : offset + numel].view(shape)
        offset += numel
    return out


def flatten(params: dict) -> Tensor:
    return torch.cat([param.flatten() for param in params.values()])
