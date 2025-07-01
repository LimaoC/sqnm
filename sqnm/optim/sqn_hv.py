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
        state["num_curvature_iters"] = -1
        state["xt_avgs"] = [torch.zeros_like(self._get_param_vector())]

    def _two_loop_recursion_check_curvature_pairs(self):
        """
        Check that we're accessing the correct (s, y) pairs when computing the two loop
        recursion - they should all be non-zero vectors
        """
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        # The paper's t index is off by 1 compared to the convention
        # They define s_t = x_t - x_{t-1} instead of s_{t-1} = x_t - x_{t-1}
        # Account for this here
        t = state["num_curvature_iters"] - 1
        sy_history = state["sy_history"]

        for i in range(max(t - m, 0), t):
            s_prev, y_prev = sy_history[i % m]
            if torch.all(s_prev == 0) or torch.all(y_prev == 0):
                logger.warning(
                    f"Found a (s, y) pair at index {i} that is zero - "
                    "this is likely an error"
                )

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:
        """
        Two loop recursion for computing H_k * grad

        This differs from the standard two loop recursion in that the (s, y) pairs are
        indexed by t not k (as the curvature pair computations are decoupled from the
        stochastic gradient computations).
        """
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        t = state["num_curvature_iters"] - 1
        sy_history = state["sy_history"]

        self._two_loop_recursion_check_curvature_pairs()

        q = grad.clone()
        alphas = torch.zeros(m)
        s_prev, y_prev = sy_history[t % m]
        history_idxs = range(max(t - m, 0), t)
        for i in reversed(history_idxs):
            s_prev, y_prev = sy_history[i % m]
            alphas[i - (t - m)] = s_prev.dot(q) / s_prev.dot(y_prev)
            q -= alphas[i - (t - m)] * y_prev
        r = (s_prev.dot(y_prev) / y_prev.dot(y_prev)) * q
        for i in history_idxs:
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
            state["num_curvature_iters"] += 1
            t = state["num_curvature_iters"]
            curv_f = torch.enable_grad()(curv_f)
            xt_avgs[-1] /= skip
            if t > 0:
                st = xt_avgs[-1] - xt_avgs[-2]
                # Compute subsampled Hessian vector product on a different, larger
                # sample given by curv_f
                yt = hvp(curv_f, self._get_param_vector(), v=st, strict=True)[1]
                sy_history[(t - 1) % m] = (st, yt)
            xt_avgs.append(torch.zeros_like(self._get_param_vector()))

        state["num_iters"] += 1
        return orig_loss
