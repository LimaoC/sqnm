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
from .sqn_base import SQNBase

logger = logging.getLogger(__name__)


class OLBFGS(SQNBase):
    def __init__(
        self,
        params,
        history_size: int = 20,
        grad_tol: float = 1e-4,
        line_search_fn: str | None = None,
        reg_term: float = 0.0,
        c: float = 0.1,
        alpha0: float = 0.1 * 100 / (100 + 2),
        tau: float = 10,
    ):
        """
        Online limited-memory BFGS (oL-BFGS)
        """
        if line_search_fn is not None and line_search_fn != "prob_wolfe":
            raise ValueError("LBFGS only supports probabilistic Wolfe line search")

        defaults = dict(
            history_size=history_size,
            grad_tol=grad_tol,
            line_search_fn=line_search_fn,
            reg_term=reg_term,
            c=c,
            alpha0=alpha0,
            tau=tau,
        )
        super().__init__(params, defaults)

        state = self.state[self._params[0]]
        # Used for probabilistic LS
        state["alpha_start"] = 1.0
        state["alpha_running_avg"] = state["alpha_start"]

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:  # type: ignore[override]
        """
        Two loop recursion for computing H_k * grad

        This differs from the standard two loop recursion in that H_k^(0) is computed
        from the (s, y) pairs from the (up to) history_size most recent iterates,
        instead of just the most recent (s, y) pair.
        """
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        line_search_fn = group["line_search_fn"]
        c = group["c"]
        k = state["num_iters"]
        sy_history = state["sy_history"]

        self._two_loop_recursion_check_curvature_pairs()

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
            alphas[-1] *= c  # Scale alpha_{k-1}
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
            fn: A pure function that computes the loss for a given input. The function
                should take a boolean parameter which, if True, also returns the
                gradient, loss variance, and gradient variance.
        """
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        grad_tol = group["grad_tol"]
        line_search_fn = group["line_search_fn"]
        reg_term = group["reg_term"]
        c = group["c"]
        alpha0 = group["alpha0"]
        tau = group["tau"]
        k = state["num_iters"]
        sy_history = state["sy_history"]
        alpha_start = state["alpha_start"]
        alpha_running_avg = state["alpha_running_avg"]

        ################################################################################

        if line_search_fn == "prob_wolfe" and fn is None:
            raise ValueError("fn parameter is needed for prob Wolfe line search")

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        orig_loss = closure()
        xk = self._get_param_vector()
        grad = self._get_grad_vector()

        # TODO: Replace this with a more robust criterion, stochastic gradient is noisy
        if grad.norm() < grad_tol:
            return orig_loss

        if k == 0:
            pk = -grad  # Gradient descent for first iteration
        else:
            # NOTE: Is it possible to check if pk is a descent direction here?
            # NOTE: Stochastic gradient isn't reliable
            pk = -self._two_loop_recursion(grad)

        if line_search_fn == "prob_wolfe":
            assert fn is not None

            f0, df0, var_f0, var_df0 = fn(xk, True)
            alpha_k, alpha_start, alpha_running_avg = prob_line_search(
                lambda x: fn(x, False),  # don't need to return vars in prob ls
                xk,
                pk,
                f0,
                df0,
                var_f0,
                var_df0,
                alpha_start,
                alpha_running_avg,
            )
            # Update alpha_start and alpha_running_avg for next iteration
            state["alpha_start"] = alpha_start
            state["alpha_running_avg"] = alpha_running_avg
        else:
            alpha_k = tau * alpha0 / (tau + k) / c

        xk_next = xk + alpha_k * pk
        self._set_param_vector(xk_next)
        closure()  # Recompute gradient after setting new param
        grad_next = self._get_grad_vector()
        sy_history[k % m] = (xk_next - xk, grad_next - grad + reg_term * (xk_next - xk))

        state["num_iters"] += 1
        return orig_loss
