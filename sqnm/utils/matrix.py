import torch
from torch import Tensor


def block_tensor(A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> Tensor:
    """
    Creates a block tensor
    A B
    C D
    from tensors A, B, C, D
    """
    top_half, bot_half = torch.cat((A, B, C, D), dim=1).t().chunk(2)
    return torch.cat((top_half, bot_half), dim=1).t()
