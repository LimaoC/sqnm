"""
Online limited-memory BFGS (oL-BFGS)

REF: Schraudolph, N. N., Yu, J., & Günter, S. (2007). A Stochastic Quasi-Newton Method
    for Online Convex Optimization.
"""

import logging
from typing import Any, Callable

import torch
from torch import Tensor

from ..line_search.prob_line_search import prob_line_search
from ..line_search.strong_wolfe_line_search import strong_wolfe_line_search
from .sqn_base import SQNBase

logger = logging.getLogger(__name__)


class OLBFGS(SQNBase):
    LINE_SEARCH_FNS = ["strong_wolfe", "prob_wolfe"]

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        line_search_fn: str | None = None,
        history_size: int = 20,
        reg_term: float = 0.0,
        step_size_weight: float = 1.0,
    ):
        """
        Online limited-memory BFGS (oL-BFGS)

        Parameters:
            params: iterable of parameters to optimize
            lr: learning rate, ignored if line_search_fn is not None
            line_search_fn: line search function to use, either None for fixed step
                size, or one of OLBFGS.LINE_SEARCH_FNS
            history_size: history size, usually 2 <= m <= 30
            reg_term: regularisation parameter in gradient difference term
            step_size_weight: step size scaling factor
        """
        if line_search_fn is not None and line_search_fn not in self.LINE_SEARCH_FNS:
            raise ValueError(f"o-LBFGS only supports one of: {self.LINE_SEARCH_FNS}")

        defaults = dict(
            lr=lr,
            line_search_fn=line_search_fn,
            history_size=history_size,
            reg_term=reg_term,
            step_size_weight=step_size_weight,
        )
        super().__init__(params, defaults)

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:  # type: ignore[override]
        """
        Two loop recursion for computing H_k * grad

        This differs from the standard two loop recursion in that H_k^(0) is computed
        from the (s, y) pairs from the (up to) history_size most recent iterates,
        instead of just the most recent (s, y) pair.
        """
        group = self.param_groups[0]
        line_search_fn = group["line_search_fn"]
        m = group["history_size"]
        c = group["step_size_weight"]

        state = self.state[self._params[0]]
        k = state["num_iters"]
        sy_history = state["sy_history"]

        q = grad.clone()
        alphas = torch.zeros(m)
        s_prev, y_prev = sy_history[(k - 1) % m]
        history_idxs = range(max(k - m, 0), k)
        const = 0
        for i in reversed(history_idxs):
            s_prev, y_prev = sy_history[i % m]
            alphas[i - (k - m)] = s_prev.dot(q) / s_prev.dot(y_prev)
            q -= alphas[i - (k - m)] * y_prev
            const += s_prev.dot(y_prev) / y_prev.dot(y_prev)
        if line_search_fn is None:
            alphas[-1] *= c  # Scale alpha_{k-1} by step_size_weight
        r = const / min(k, m) * q
        for i in history_idxs:
            s_prev, y_prev = sy_history[i % m]
            beta = y_prev.dot(r) / s_prev.dot(y_prev)
            r += (alphas[i - (k - m)] - beta) * s_prev
        return r

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float],
        fn: Callable[[Tensor, bool], Any] | None = None,
    ) -> float:
        """
        Perform a single oL-BFGS iteration.

        Parameters:
            closure: A closure that re-evaluates the model and returns the loss.
            fn: A pure function that computes the loss for a given input. Required if
                using line search. For linesearch_fn == "prob_wolfe", the function
                should also take a boolean parameter which, if True, also returns the
                gradient, loss variance, and gradient variance.
        """
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        lr = group["lr"]
        line_search_fn = group["line_search_fn"]
        m = group["history_size"]
        reg_term = group["reg_term"]
        c = group["step_size_weight"]

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
            assert fn is not None
            f0, df0, var_f0, var_df0 = fn(xk, True)
            # Don't need function handle to return variances in line search
            alpha_k = prob_line_search(
                lambda x: fn(x, False), xk, pk, f0, df0, var_f0, var_df0
            )
        else:
            # Use fixed step size
            alpha_k = lr / c

        xk_next = xk + alpha_k * pk
        self._set_param_vector(xk_next)
        closure()  # Recompute gradient after setting new param
        gradk_next = self._get_grad_vector()

        sk = alpha_k * pk
        yk = gradk_next - gradk + reg_term * sk
        sy_history[state["num_sy_pairs"] % m] = (sk, yk)
        state["num_sy_pairs"] += 1

        state["num_iters"] += 1
        return orig_loss
