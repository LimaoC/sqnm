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
            eps=eps,
            eta0=eta0,
            tau=tau,
            c=c,
        )
        super().__init__(params, defaults)

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:
        """
        Two loop recursion for computing H_k * grad

        H_k^0 is computed from the (s, y) pairs from the (up to) history_size most
        recent iterates, as opposed to just using the most recent (s, y) pair in the
        standard L-BFGS two loop recursion.
        """
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        k = state["num_iters"]
        sy_history = state["sy_history"]
        s_prev, y_prev = sy_history[(k - 1) % m]

        q = grad.clone()
        alphas = torch.zeros(m)
        const = 0
        for i in range(k - 1, max(k - m, 0) - 1, -1):
            s_prev, y_prev = sy_history[i % m]
            alphas[i - (k - m)] = s_prev.dot(q) / s_prev.dot(y_prev)
            q -= alphas[i - (k - m)] * y_prev

            const += s_prev.dot(y_prev) / y_prev.dot(y_prev)
        r = const / min(k, m) * q
        for i in range(max(k - m, 0), k):
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
        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        grad_tol = group["grad_tol"]
        eps = group["eps"]
        eta0 = group["eta0"]
        tau = group["tau"]
        c = group["c"]
        k = state["num_iters"]
        sy_history = state["sy_history"]

        orig_loss = closure()  # Populate gradients
        xk = self._get_param_vector()
        grad = self._get_grad_vector()
        # TODO: replace this with a more robust criterion
        if grad.norm() < grad_tol:
            return orig_loss

        if k == 0:
            pk = -grad * eps
        else:
            pk = -self._two_loop_recursion(grad)
            if grad.dot(pk) >= 0:
                logger.warning("p_k may not be a descent direction.")

        alpha_k = tau * eta0 / (tau + k)
        xk_next = xk + (alpha_k / c) * pk
        self._set_param_vector(xk_next)
        closure()
        grad_next = self._get_grad_vector()
        sy_history[k % m] = (xk_next - xk, grad_next - grad)

        state["num_iters"] += 1
        return orig_loss
