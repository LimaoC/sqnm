"""
Online limited-memory BFGS (oL-BFGS)

REF: Schraudolph, N. N., Yu, J., & Günter, S. (2007). A Stochastic Quasi-Newton Method
    for Online Convex Optimization.
"""

import logging
from typing import Any, Callable, cast

import torch
from torch import Tensor

from sqnm.line_search import prob_line_search, strong_wolfe_line_search
from sqnm.optim.sqn_base import SQNBase

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
        c: float = 1.0,
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
            c: step size scaling factor
        """
        if line_search_fn is not None and line_search_fn not in self.LINE_SEARCH_FNS:
            raise ValueError(f"o-LBFGS only supports one of: {self.LINE_SEARCH_FNS}")

        defaults = dict(
            lr=lr,
            line_search_fn=line_search_fn,
            history_size=history_size,
            reg_term=reg_term,
            c=c,
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
        c = group["c"]

        state = self.state[self._params[0]]
        k = state["num_iters"]
        h = min(k, m)  # Number of curvature pairs to use
        idxs = torch.arange(k - h, k) % m
        s = state["s_hist"][idxs]  # [h, d]
        y = state["y_hist"][idxs]  # [h, d]

        q = grad.clone()
        sy = torch.sum(s * y, dim=1)  # [h], precompute s.dot(y) for each pair
        alphas = torch.zeros(h, device=grad.device)
        for i in reversed(range(h)):
            alphas[i] = s[i].dot(q) / sy[i]
            q -= alphas[i] * y[i]
        if line_search_fn is None:
            alphas[-1] *= c  # Scale alpha_{k-1} by c
        r = (torch.sum(sy / torch.sum(y * y, dim=1)) / h) * q
        for i in range(h):
            beta = y[i].dot(r) / sy[i]
            r += (alphas[i] - beta) * s[i]
        return r

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float],
        fn: Callable[[Tensor], Tensor] | Callable[[Tensor, bool], Any] | None = None,
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
        c = group["c"]

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
            fn = cast(Callable[[Tensor], Tensor], fn)
            # Choose step size to satisfy strong Wolfe conditions
            grad_fn = torch.func.grad(fn)
            phi0 = orig_loss
            grad_phi0 = gradk.dot(pk).item()
            alpha_k = strong_wolfe_line_search(fn, grad_fn, xk, pk, phi0, grad_phi0)
        elif line_search_fn == "prob_wolfe":
            assert fn is not None
            fn = cast(Callable[[Tensor, bool], Any], fn)
            var_f0, var_df0 = fn(xk, True)
            # Don't need function handle to return variances in line search
            alpha_k, _, _ = prob_line_search(
                lambda x: fn(x, False), xk, pk, orig_loss, gradk, var_f0, var_df0
            )
        else:
            # Use fixed step size
            alpha_k = lr / c

        self._add_param_vector(alpha_k * pk)
        closure()  # Recompute gradient after setting new param
        gradk_next = self._get_grad_vector()

        sk = alpha_k * pk
        yk = gradk_next - gradk + reg_term * sk
        s_hist[state["num_sy_pairs"] % m] = sk
        y_hist[state["num_sy_pairs"] % m] = yk
        state["num_sy_pairs"] += 1

        state["num_iters"] += 1
        return orig_loss
