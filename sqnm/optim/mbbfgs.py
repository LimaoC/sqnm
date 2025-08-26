"""
Multi-batch BFGS (MB-BFGS)

Berahas, A. S., Nocedal, J., & Takáč, M. (2016). A Multi-Batch L-BFGS Method for Machine
    Learning.
"""

import logging
from typing import Callable

import torch
from torch import Tensor

from ..line_search.strong_wolfe_line_search import strong_wolfe_line_search
from .sqn_base import SQNBase

logger = logging.getLogger(__name__)


class MBBFGS(SQNBase):
    LINE_SEARCH_FNS = ["strong_wolfe"]

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        line_search_fn: str | None = None,
        history_size: int = 20,
    ):
        if line_search_fn is not None and line_search_fn not in self.LINE_SEARCH_FNS:
            raise ValueError(f"MB-BFGS only supports one of: {self.LINE_SEARCH_FNS}")

        defaults = dict(
            lr=lr,
            line_search_fn=line_search_fn,
            history_size=history_size,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float],
        overlap_fn: Callable[[Tensor], Tensor],
        fn: Callable[[Tensor], Tensor] | None = None,
    ):
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        lr = group["lr"]
        line_search_fn = group["line_search_fn"]
        m = group["history_size"]

        state = self.state[self._params[0]]
        k = state["num_iters"]
        sy_history = state["sy_history"]

        if line_search_fn is not None and fn is None:
            raise ValueError("fn parameter is needed for line search")

        ################################################################################

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        orig_loss = closure()  # Populate gradients
        xk = self._get_param_vector()
        gradk = self._get_grad_vector()

        # NOTE: Termination criterion?

        if k == 0:
            pk = -gradk  # Gradient descent for first iteration
        else:
            # NOTE: Can't reliably check if pk is a descent direction here
            pk = -self._two_loop_recursion(gradk)

        if line_search_fn == "strong_wolfe":
            assert fn is not None
            # Choose step size to satisfy strong Wolfe conditions
            grad_fn = torch.func.grad(fn)
            alpha_k = strong_wolfe_line_search(fn, grad_fn, xk, pk)
        elif line_search_fn == "prob_wolfe":
            # TODO:
            pass
        else:
            # Use fixed step size
            alpha_k = lr

        xk_next = xk + alpha_k * pk
        self._set_param_vector(xk_next)

        sk = alpha_k * pk
        grad_overlap_fn = torch.func.grad(overlap_fn)
        yk = grad_overlap_fn(xk_next) - grad_overlap_fn(xk)
        sy_history[state["num_sy_pairs"] % m] = (sk, yk)
        state["num_sy_pairs"] += 1

        state["num_iters"] += 1
        return orig_loss
