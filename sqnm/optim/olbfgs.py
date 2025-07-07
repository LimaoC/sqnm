"""
Online limited-memory BFGS (oL-BFGS)

REF: Schraudolph, N. N., Yu, J., & Günter, S. (2007). A Stochastic Quasi-Newton Method
    for Online Convex Optimization.
"""

import logging
from typing import Callable

import torch
from torch import Tensor

from .sqn_base import SQNBase

logger = logging.getLogger(__name__)


class OLBFGS(SQNBase):
    def __init__(
        self,
        params,
        history_size: int = 20,
        grad_tol: float = 1e-4,
        l: float = 0,
        eps: float = 1e-10,
        eta0: float = 0.1 * 100 / (100 + 2),
        tau: float = 2 * 10**4,
        c: float = 0.1,
    ):
        """
        Online limited-memory BFGS (oL-BFGS)
        """
        defaults = dict(
            history_size=history_size,
            grad_tol=grad_tol,
            l=l,
            eps=eps,
            eta0=eta0,
            tau=tau,
            c=c,
        )
        super().__init__(params, defaults)

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:
        """
        Two loop recursion for computing H_k * grad

        This differs from the standard two loop recursion in that H_k^(0) is computed
        from the (s, y) pairs from the (up to) history_size most recent iterates,
        instead of just the most recent (s, y) pair.
        """
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
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
        r = const / min(k, m) * q
        for i in history_idxs:
            s_prev, y_prev = sy_history[i % m]
            beta = y_prev.dot(r) / s_prev.dot(y_prev)
            r += (alphas[i - (k - m)] - beta) * s_prev
        return r

    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> float:  # type: ignore[override]
        """
        Perform a single oL-BFGS iteration.

        Parameters:
            closure: A closure that re-evaluates the model and returns the loss.
        """
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        grad_tol = group["grad_tol"]
        l = group["l"]
        eps = group["eps"]
        eta0 = group["eta0"]
        tau = group["tau"]
        c = group["c"]
        k = state["num_iters"]
        sy_history = state["sy_history"]

        ################################################################################

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        orig_loss = closure()  # Populate gradients
        xk = self._get_param_vector()
        grad = self._get_grad_vector()
        # TODO: Replace this with a more robust criterion, stochastic gradient is noisy
        if grad.norm() < grad_tol:
            return orig_loss

        if k == 0:
            pk = -grad * eps
        else:
            # NOTE: Is it possible to check if pk is a descent direction here?
            # NOTE: Stochastic gradient isn't reliable
            pk = -self._two_loop_recursion(grad)

        # TODO: Replace this with a more robust step size
        alpha_k = tau * eta0 / (tau + k)
        xk_next = xk + (alpha_k / c) * pk
        self._set_param_vector(xk_next)
        closure()  # Populate gradient of xk_next
        grad_next = self._get_grad_vector()
        sy_history[k % m] = (xk_next - xk, grad_next - grad + l * (xk_next - xk))

        state["num_iters"] += 1
        return orig_loss
