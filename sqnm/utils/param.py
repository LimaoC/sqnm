from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


def unflatten(
    flat_vec: Tensor, param_shapes: dict[str, torch.Size]
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = dict()
    offset = 0
    for name, shape in param_shapes.items():
        numel = int(torch.prod(torch.tensor(shape)).item())
        out[name] = flat_vec[offset : offset + numel].view(shape)
        offset += numel
    return out


def flatten(params: dict) -> Tensor:
    return torch.cat([param.flatten() for param in params.values()])


def grad_vec(params: Iterable[nn.Parameter]) -> Tensor:
    """Concatenates gradients from params into a 1D tensor"""
    grads = []
    for param in params:
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)
