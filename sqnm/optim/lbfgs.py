"""
Limited-memory BFGS (L-BFGS)

REF: Algorithm 7.5 in Numerical Optimization by Nocedal and Wright
"""

import logging
from typing import Callable, cast

import torch
from torch import Tensor

from sqnm.line_search import strong_wolfe_line_search
from sqnm.optim.sqn_base import SQNBase

logger = logging.getLogger(__name__)


class LBFGS(SQNBase):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        line_search_fn: str | None = None,
        history_size: int = 20,
        grad_tol: float = 1e-4,
    ):
        """
        Limited-memory BFGS (L-BFGS)

        Parameters:
            params: iterable of parameters to optimize
            lr: learning rate, ignored if line_search_fn is not None
            line_search_fn: line search function to use, either None for fixed step
                size, or "strong_wolfe" for strong Wolfe line search
            history_size: history size, usually 2 <= m <= 30
            grad_tol: termination tolerance for gradient norm
        """
        if line_search_fn is not None and line_search_fn != "strong_wolfe":
            raise ValueError("L-BFGS only supports strong Wolfe line search")

        defaults = dict(
            lr=lr,
            line_search_fn=line_search_fn,
            history_size=history_size,
            grad_tol=grad_tol,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], Tensor],
        fn: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        """
        Perform a single L-BFGS iteration.

        Parameters:
            closure: A closure that re-evaluates the model and returns the loss.
            fn: A pure function that computes the loss for a given input.
        """
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        lr = group["lr"]
        m = group["history_size"]
        grad_tol = group["grad_tol"]
        line_search_fn = group["line_search_fn"]

        state = self.state[self._params[0]]
        k: int = state["num_iters"]
        s_hist: Tensor = state["s_hist"]
        y_hist: Tensor = state["y_hist"]
        alphas: list[float] = state["alphas"]
        sdotys: list[Tensor] = state["sdotys"]

        if line_search_fn == "strong_wolfe" and fn is None:
            raise ValueError("fn parameter is needed for strong Wolfe line search")

        ################################################################################

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        orig_loss = closure()  # Populate gradients
        loss = float(orig_loss)

        xk = self._get_param_vector()
        gradk = self._get_grad_vector()

        if gradk.norm() < grad_tol:
            return orig_loss

        if k == 0:
            pk = -gradk  # Gradient descent for first iteration
        else:
            pk = -self._two_loop_recursion(gradk)
            if gradk.dot(pk) >= 0:
                logger.warning("p_k is not a descent direction.")

        if line_search_fn == "strong_wolfe":
            fn = cast(Callable[[Tensor], Tensor], fn)
            grad_and_val_fn = torch.func.grad_and_value(fn)
            alpha_k, ls_func_evals = strong_wolfe_line_search(
                grad_and_val_fn, xk, pk, loss, gradk
            )
        else:
            # Use fixed step size
            alpha_k = lr

        # Compute and store next iterates
        sk = alpha_k * pk
        self._add_param_vector(sk)
        closure()  # Recompute gradient after setting new param
        gradk_next = self._get_grad_vector()
        yk = gradk_next - gradk

        # Update state
        alphas.append(alpha_k)
        sdotys.append(sk.dot(yk))
        s_hist[state["num_sy_pairs"] % m] = sk
        y_hist[state["num_sy_pairs"] % m] = yk
        state["num_sy_pairs"] += 1
        state["num_iters"] += 1

        return orig_loss
