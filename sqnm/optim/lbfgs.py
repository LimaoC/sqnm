"""
Limited-memory BFGS (L-BFGS)

REF: Algorithm 7.5 in Numerical Optimization by Nocedal and Wright
"""

import logging
from typing import Callable

import torch
from torch import Tensor

from ..line_search.strong_wolfe import strong_wolfe_line_search
from .sqn_base import SQNBase

logger = logging.getLogger(__name__)


class LBFGS(SQNBase):
    def __init__(
        self,
        params,
        lr: float = 1,
        history_size: int = 20,
        grad_tol: float = 1e-4,
        line_search_fn: str | None = None,
    ):
        """
        Limited-memory BFGS (L-BFGS)

        Parameters:
            params: iterable of parameters to optimize
            lr: learning rate, only used if line_search_fn is None
            history_size: history size, usually 2 <= m <= 30
            grad_tol: termination tolerance for gradient norm
            line_search_fn: line search function to use, either None for fixed step
                size, or "strong_wolfe" for strong Wolfe line search
        """
        if lr <= 0:
            raise ValueError("L-BFGS learning rate must be positive")
        if line_search_fn is not None and line_search_fn != "strong_wolfe":
            raise ValueError("L-BFGS only supports strong Wolfe line search")

        defaults = dict(
            lr=lr,
            history_size=history_size,
            grad_tol=grad_tol,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Callable[[], float], fn: Callable[[Tensor], Tensor] | None = None
    ) -> float:
        """
        Perform a single L-BFGS iteration.

        Parameters:
            closure: A closure that re-evaluates the model and returns the loss.
            fn: A pure function that computes the loss for a given input.
        """
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        lr = group["lr"]
        m = group["history_size"]
        grad_tol = group["grad_tol"]
        line_search_fn = group["line_search_fn"]
        k = state["num_iters"]
        sy_history = state["sy_history"]

        ################################################################################

        if line_search_fn == "strong_wolfe" and fn is None:
            raise ValueError("fn parameter is needed for strong Wolfe line search")

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        orig_loss = closure()  # Populate gradients
        xk = self._get_param_vector()
        grad = self._get_grad_vector()

        if grad.norm() < grad_tol:
            return orig_loss

        if k == 0:
            pk = -grad  # Gradient descent for first iteration
        else:
            pk = -self._two_loop_recursion(grad)
            if grad.dot(pk) >= 0:
                logger.warning("p_k is not a descent direction.")

        if line_search_fn == "strong_wolfe":
            assert fn is not None
            # Choose step size to satisfy strong Wolfe conditions
            grad_fn = torch.func.grad(fn)
            alpha_k = strong_wolfe_line_search(fn, grad_fn, xk, pk)
            xk_next = xk + alpha_k * pk
        else:
            # Use fixed step size
            xk_next = xk + lr * pk

        # Compute and store next iterates
        self._set_param_vector(xk_next)
        closure()  # Recompute gradient after setting new param
        grad_next = self._get_grad_vector()
        sy_history[k % m] = (xk_next - xk, grad_next - grad)

        state["num_iters"] += 1
        return orig_loss
