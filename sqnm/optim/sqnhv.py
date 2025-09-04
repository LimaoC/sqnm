"""
Hessian-Vector Stochastic Quasi-Newton (SQN-Hv)

REF: Byrd, R. H., Hansen, S. L., Nocedal, J., & Singer, Y. (2015). A Stochastic
    Quasi-Newton Method for Large-Scale Optimization.
"""

import logging
from typing import Any, Callable, cast

import torch
from torch import Tensor
from torch.autograd.functional import hvp

from sqnm.line_search import prob_line_search, strong_wolfe_line_search
from sqnm.optim.sqn_base import SQNBase

logger = logging.getLogger(__name__)


class SQNHv(SQNBase):
    LINE_SEARCH_FNS = ["strong_wolfe", "prob_wolfe"]

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        line_search_fn: str | None = None,
        history_size: int = 20,
        skip: int = 10,
    ):
        """
        Hessian-Vector Stochastic Quasi-Newton (SQN-Hv)

        Parameters:
            params: iterable of parameters to optimize
            lr: learning rate, ignored if line_search_fn is not None
            line_search_fn: line search function to use, either None for fixed step
                size, or one of OLBFGS.LINE_SEARCH_FNS
            history_size: history size, usually 2 <= m <= 30
            skip: number of iterations between curvature estimates
        """
        if line_search_fn is not None and line_search_fn not in self.LINE_SEARCH_FNS:
            raise ValueError(f"SQN-Hv only supports one of: {self.LINE_SEARCH_FNS}")

        defaults = dict(
            lr=lr,
            history_size=history_size,
            line_search_fn=line_search_fn,
            skip=skip,
        )
        super().__init__(params, defaults)

        state = self.state[self._params[0]]
        state["num_iters"] = 1  # Algorithm in paper starts from k = 1
        state["xt"] = [
            torch.zeros_like(self._get_param_vector()),
            torch.zeros_like(self._get_param_vector()),
        ]
        # Used for probabilistic LS
        state["alpha_start"] = 1.0
        state["alpha_running_avg"] = state["alpha_start"]

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:
        """
        Two loop recursion for computing H_k * grad

        This differs from the standard two loop recursion in that the (s, y) pairs are
        indexed by t not k (as the curvature pair computations are decoupled from the
        stochastic gradient computations).
        """
        group = self.param_groups[0]
        m = group["history_size"]

        state = self.state[self._params[0]]
        # The paper's t index is off by 1 compared to the convention, account for this
        # They define s_t = x_t - x_{t-1} instead of s_{t-1} = x_t - x_{t-1}
        t = state["num_sy_pairs"] - 1
        h = min(t, m)  # Number of curvature pairs to use
        idxs = torch.arange(t - h, t) % m
        s = state["s_hist"][idxs]  # [h, d]
        y = state["y_hist"][idxs]  # [h, d]

        q = grad.clone()
        sy = torch.sum(s * y, dim=1)  # [h], precompute s.dot(y) for each pair
        alphas = torch.zeros(h, device=grad.device)
        for i in reversed(range(h)):
            alphas[i] = s[i].dot(q) / sy[i]
            q -= alphas[i] * y[i]
        r = (sy[0] / (y[0].dot(y[0]))) * q
        for i in range(h):
            beta = y[i].dot(r) / sy[i]
            r += (alphas[i] - beta) * s[i]
        return r

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float],
        fn: Callable[[Tensor], Tensor] | Callable[[Tensor, bool], Any] | None = None,
        curvature_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> float:
        """
        Perform a single SQN-Hv iteration.

        Parameters:
            closure: A closure that re-evaluates the model and returns the loss.
            fn: A pure function that computes the loss for a given input. Required if
                line_search_fn == "prob_wolfe". The function should take a boolean
                parameter which, if True, also returns the gradient, loss variance, and
                gradient variance.
            curvature_fn: A pure function that computes the loss for a given input. The
                function should be provided every `skip` iterations.
        """
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        m = group["history_size"]
        lr = group["lr"]
        line_search_fn = group["line_search_fn"]
        skip = group["skip"]  # L

        state = self.state[self._params[0]]
        k = state["num_iters"]
        s_hist = state["s_hist"]
        y_hist = state["y_hist"]
        # Note index t for curvature pairs, which are decoupled from gradient estimates
        xt = state["xt"]
        alpha_start = state["alpha_start"]
        alpha_running_avg = state["alpha_running_avg"]

        if line_search_fn is not None and fn is None:
            raise ValueError("fn parameter is needed for line search")

        if k % skip != 0 and curvature_fn is not None:
            logger.warning(f"Got curvature_fn but didn't expect it on iteration {k}")
        if k % skip == 0 and curvature_fn is None:
            raise TypeError(f"Expected curvature_fn but didn't get it on iteration {k}")

        ################################################################################

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        orig_loss = closure()  # Populate gradients
        xk = self._get_param_vector()
        gradk = self._get_grad_vector()

        # NOTE: Termination criterion?

        # Accumulate average over L iterations
        xt[1] += xk

        if k <= 2 * skip:
            # Stochastic gradient descent for first 2L iterations
            pk = -gradk
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
            # Don't need function handle to return vars in line search
            if k <= 2 * skip:
                # Propagate step sizes in probabilistic ls - we don't have curvature
                # information yet
                alpha_k, alpha_start, alpha_running_avg = prob_line_search(
                    lambda x: fn(x, False),
                    xk,
                    pk,
                    orig_loss,
                    gradk,
                    var_f0,
                    var_df0,
                    a_running_avg=alpha_running_avg,
                    a0=alpha_start,
                )
                state["alpha_start"] = alpha_start
                state["running_avg"] = alpha_running_avg
            else:
                alpha_k, _, _ = prob_line_search(
                    lambda x: fn(x, False), xk, pk, orig_loss, gradk, var_f0, var_df0
                )
        else:
            # Use fixed step size
            alpha_k = lr

        self._set_param_vector(alpha_k * pk)

        if k % skip == 0:
            # Compute curvature pairs every L iterations
            t = state["num_sy_pairs"] - 1
            xt[1] /= skip
            if t >= 0:
                st = xt[1] - xt[0]
                # Compute subsampled Hessian vector product on a different, larger
                # sample given by curvature_fn
                _, yt = hvp(curvature_fn, xt[1], v=st, strict=True)
                s_hist[t % m] = st
                y_hist[t % m] = yt
            xt[0], xt[1] = xt[1], torch.zeros_like(xt[1])
            state["num_sy_pairs"] += 1

        state["num_iters"] += 1
        return orig_loss
