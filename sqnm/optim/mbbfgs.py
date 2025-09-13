"""
Multi-batch BFGS (MB-BFGS)

Berahas, A. S., Nocedal, J., & Takáč, M. (2016). A Multi-Batch L-BFGS Method for Machine
    Learning.
"""

import logging
from typing import Any, Callable, cast

import torch
from torch import Tensor

from sqnm.line_search import prob_line_search, strong_wolfe_line_search
from sqnm.optim.sqn_base import SQNBase

logger = logging.getLogger(__name__)


class MBBFGS(SQNBase):
    LINE_SEARCH_FNS = ["strong_wolfe", "prob_wolfe"]
    EXCEPTION_STRATS = ["skip_update"]

    def __init__(
        self,
        params,
        batch_size: int,
        overlap_batch_size: int,
        lr: float = 1e-3,
        line_search_fn: str | None = None,
        history_size: int = 20,
        weight_decay: float = 0.0,
        exception_strat: str | None = None,
        eps: float = 1e-8,
    ):
        """
        Multi-Batch BFGS (MB-BFGS)

        Parameters:
            params: iterable of parameters to optimize
            batch_size: used to keep track of number of function evaluations
            overlap_batch_size: used to keep track of number of function evaluations
            lr: learning rate, ignored if line_search_fn is not None
            line_search_fn: line search function to use, either None for fixed step
                size, or one of MBBFGS.LINE_SEARCH_FNS
            history_size: history size, usually 2 <= m <= 30
            weight_decay: l2 regularisation term
            exception_strat: strategy to employ if curvature condition is violated,
                either None for no strategy or one of MBBFGS.EXCEPTION_STRATS
            eps: if exception_strat == "skip_update", the update is skipped if
                s.dot(y) < eps * ||s||^2
        """
        if line_search_fn is not None and line_search_fn not in self.LINE_SEARCH_FNS:
            raise ValueError(f"MB-BFGS only supports one of: {self.LINE_SEARCH_FNS}")
        if exception_strat is not None and exception_strat not in self.EXCEPTION_STRATS:
            raise ValueError(f"MB-BFGS only supports one of: {self.EXCEPTION_STRATS}")

        defaults = dict(
            batch_size=batch_size,
            overlap_batch_size=overlap_batch_size,
            lr=lr,
            line_search_fn=line_search_fn,
            history_size=history_size,
            weight_decay=weight_decay,
            exception_strat=exception_strat,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], Tensor],
        overlap_fn: Callable[[Tensor], Tensor],
        fn: Callable[[Tensor], Tensor] | Callable[[Tensor, bool], Any] | None = None,
    ):
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        batch_size = group["batch_size"]
        overlap_batch_size = group["overlap_batch_size"]
        lr = group["lr"]
        line_search_fn = group["line_search_fn"]
        m = group["history_size"]
        weight_decay = group["weight_decay"]
        exception_strat = group["exception_strat"]
        eps = group["eps"]

        state = self.state[self._params[0]]
        k = state["num_iters"]
        s_hist = state["s_hist"]
        y_hist = state["y_hist"]

        if line_search_fn is not None and fn is None:
            raise ValueError("fn parameter is needed for line search")

        ################################################################################

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        orig_loss = closure()  # Populate gradients
        loss = float(orig_loss)
        state["func_evals"] += 2 * batch_size  # Forward + backward pass

        xk = self._get_param_vector()
        gradk = self._get_grad_vector()
        if weight_decay != 0:
            gradk.add_(xk, alpha=weight_decay)

        # NOTE: Termination criterion?

        if k == 0:
            pk = -gradk  # Gradient descent for first iteration
        else:
            # NOTE: Can't reliably check if pk is a descent direction here
            pk = -self._two_loop_recursion(gradk)

        if line_search_fn == "strong_wolfe":
            fn = cast(Callable[[Tensor], Tensor], fn)
            grad_and_val_fn = torch.func.grad_and_value(fn)
            alpha_k, ls_func_evals = strong_wolfe_line_search(
                grad_and_val_fn, xk, pk, loss, gradk
            )
            state["func_evals"] += ls_func_evals * batch_size
        elif line_search_fn == "prob_wolfe":
            fn = cast(Callable[[Tensor, bool], Any], fn)
            grad_and_val_fn = torch.func.grad_and_value(lambda x: fn(x, False))
            var_f0, var_df0 = fn(xk, True)
            # Don't need function handle to return variances in line search
            alpha_k, _, _, ls_func_evals = prob_line_search(
                grad_and_val_fn, xk, pk, loss, gradk, var_f0, var_df0
            )
            state["func_evals"] += ls_func_evals * batch_size
        else:
            # Use fixed step size
            alpha_k = lr

        sk = alpha_k * pk
        xk_next = xk + sk
        grad_overlap_fn = torch.func.grad(overlap_fn)
        yk = grad_overlap_fn(xk_next) - grad_overlap_fn(xk)
        state["func_evals"] += 4 * overlap_batch_size  # Forward + backward pass

        if exception_strat == "skip_update" and sk.dot(yk) < eps * sk.dot(sk):
            # Curvature condition violated, skip this update
            return orig_loss

        self._add_param_vector(sk)
        s_hist[state["num_sy_pairs"] % m] = sk
        y_hist[state["num_sy_pairs"] % m] = yk
        state["num_sy_pairs"] += 1

        state["num_iters"] += 1
        return orig_loss
