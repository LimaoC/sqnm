"""
Hessian-Vector Stochastic Quasi-Newton (SQN-Hv)

REF: Byrd, R. H., Hansen, S. L., Nocedal, J., & Singer, Y. (2015). A Stochastic Quasi-Newton Method for Large-Scale Optimization.
"""

import logging
from typing import Callable

import torch
from torch import Tensor
from torch.autograd.functional import hvp

from .sqn_base import SQNBase

logger = logging.getLogger(__name__)


class SQNHv(SQNBase):
    def __init__(
        self,
        params,
        history_size: int = 20,
        grad_tol: float = 1e-4,
        skip: int = 10,
        beta: float = 1e-1,
    ):
        """
        Hessian-Vector Stochastic Quasi-Newton (SQN-Hv)
        """
        defaults = dict(
            history_size=history_size,
            grad_tol=grad_tol,
            skip=skip,
            beta=beta,
        )
        super().__init__(params, defaults)

        state = self.state[self._params[0]]
        state["num_iters"] = 1  # Algorithm in paper starts from k = 1
        state["num_curvature_pairs"] = -1
        state["xt_avgs"] = [torch.zeros_like(self._get_param_vector())]

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:
        """
        Two loop recursion for computing H_k * grad

        REF: Algorithm 7.4 in Numerical Optimization by Nocedal and Wright
        """
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        t = state["num_curvature_pairs"] - 1
        sy_history = state["sy_history"]

        self._two_loop_recursion_check_curvature_pairs()

        q = grad.clone()
        alphas = torch.zeros(m)
        s_prev, y_prev = sy_history[t % m]
        for i in range(t - 1, max(t - m, 0) - 1, -1):
            s_prev, y_prev = sy_history[i % m]
            alphas[i - (t - m)] = s_prev.dot(q) / s_prev.dot(y_prev)
            q -= alphas[i - (t - m)] * y_prev
        r = (s_prev.dot(y_prev) / y_prev.dot(y_prev)) * q
        for i in range(max(t - m, 0), t):
            s_prev, y_prev = sy_history[i % m]
            beta = y_prev.dot(r) / s_prev.dot(y_prev)
            r += (alphas[i - (t - m)] - beta) * s_prev
        return r

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], float],
        curv_f: Callable[[Tensor], Tensor] | None = None,
    ) -> float:
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        grad_tol = group["grad_tol"]
        skip = group["skip"]  # L
        beta = group["beta"]
        k = state["num_iters"]
        sy_history = state["sy_history"]
        xt_avgs = state["xt_avgs"]

        ################################################################################

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        if k % skip != 0 and curv_f is not None:
            logger.warning(f"Got curv_f but didn't expect it on iteration {k}")
        if k % skip == 0 and curv_f is None:
            raise TypeError(f"Expected curv_f but didn't get it on iteration {k}")

        orig_loss = closure()  # Populate gradients
        xk = self._get_param_vector()
        grad = self._get_grad_vector()
        # TODO: Replace this with a more robust criterion, stochastic gradient is noisy
        if grad.norm() < grad_tol:
            return orig_loss

        # Accumulate average over L iterations
        xt_avgs[-1] += xk

        if k <= 2 * skip:
            # Stochastic gradient descent for first 2L iterations
            pk = -grad
        else:
            # NOTE: Is it possible to check if pk is a descent direction here?
            # NOTE: Stochastic gradient isn't reliable
            pk = -self._two_loop_recursion(grad)

        # TODO: Replace this with a more robust step size
        alpha_k = beta / k
        xk_next = xk + alpha_k * pk
        self._set_param_vector(xk_next)

        if k % skip == 0:
            # Compute curvature pairs every L iterations
            state["num_curvature_pairs"] += 1
            t = state["num_curvature_pairs"]
            curvature_f = torch.enable_grad()(curv_f)
            xt_avgs[-1] /= skip
            if t > 0:
                st = xt_avgs[-1] - xt_avgs[-2]
                yt = hvp(curvature_f, self._get_param_vector(), v=st, strict=True)[1]
                # print(st, yt)
                sy_history[(t - 1) % m] = (st, yt)
            xt_avgs.append(torch.zeros_like(self._get_param_vector()))

        state["num_iters"] += 1
        return orig_loss
