"""
Self-correcting BFGS (SC-BFGS)

Curtis, F. E. (2016). A Self-Correcting Variable-Metric Algorithm for Stochastic
    Optimization.
"""

import logging
from typing import Any, Callable, cast

import numpy as np
import torch
from torch import Tensor

from sqnm.line_search import prob_line_search, strong_wolfe_line_search
from sqnm.optim.sqn_base import SQNBase

logger = logging.getLogger(__name__)


class SCBFGS(SQNBase):
    LINE_SEARCH_FNS = ["strong_wolfe", "prob_wolfe"]

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        line_search_fn: str | None = None,
        history_size: int = 20,
        stable: bool = True,
        eta1: float = 1 / 4,
        eta2: float = 4,
        rho: float = 1 / 8,
        tau: float = 8,
    ):
        if line_search_fn is not None and line_search_fn not in self.LINE_SEARCH_FNS:
            raise ValueError(f"SC-BFGS only supports one of: {self.LINE_SEARCH_FNS}")
        if eta1 <= 0 or eta1 >= 1:
            raise ValueError("eta1 should be in the range (0, 1)")
        if eta2 < 1:
            raise ValueError("eta2 should be in the range [1, inf)")

        defaults = dict(
            lr=lr,
            line_search_fn=line_search_fn,
            history_size=history_size,
            stable=stable,
            eta1=eta1,
            eta2=eta2,
            rho=rho,
            tau=tau,
        )
        super().__init__(params, defaults)

        state = self.state[self._params[0]]
        state["num_proxy_funcs"] = 0

    def _compute_beta(self, sk, yk) -> float:
        group = self.param_groups[0]
        eta1 = group["eta1"]
        eta2 = group["eta2"]

        sksk = sk.dot(sk)
        skyk = sk.dot(yk)
        ykyk = yk.dot(yk)

        beta_lower = 0
        if skyk < eta1 * sksk:
            beta_lower = (eta1 * sksk - skyk) / (sksk - skyk)
        yy = beta_lower * sk + (1 - beta_lower) * yk
        if beta_lower > 0 and yy.dot(yy) > eta2 * sk.dot(yy):
            beta_lower = 1
        beta_upper = 0
        if ykyk > eta2 * skyk:
            # ax^2 + bx + c
            a = (sksk - 2 * skyk + ykyk).item()
            b = (2 * skyk - 2 * ykyk - eta2 * sksk + eta2 * skyk).item()
            c = (ykyk - eta2 * skyk).item()
            beta_upper = np.min(np.roots([a, b, c]))

        return max(beta_lower, beta_upper)

    def _check_self_correcting_conditions(
        self, xk: Tensor, pk: Tensor, proxy_fn: Callable[[Tensor], Tensor]
    ) -> bool:
        group = self.param_groups[0]
        rho = group["rho"]
        tau = group["tau"]

        proxy_grad_fn = torch.func.grad(proxy_fn)
        proxy_gradk = proxy_grad_fn(xk)
        norm_sq = proxy_gradk.dot(proxy_gradk)
        return rho * norm_sq <= -proxy_gradk.dot(pk) and pk.dot(pk) <= tau * norm_sq

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float],
        proxy_fn: Callable[[Tensor], Tensor],
        fn: Callable[[Tensor], Tensor] | Callable[[Tensor, bool], Any] | None = None,
    ):
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        lr = group["lr"]
        line_search_fn = group["line_search_fn"]
        m = group["history_size"]
        stable = group["stable"]

        state = self.state[self._params[0]]
        k = state["num_iters"]
        # sy_history = state["sy_history"]
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
            # Store curvature pairs from previous iteration
            # Store (sk, yk) from previous iteration
            # sk computed already - need yk using this iteration's stochastic gradient
            sk = s_hist[state["num_sy_pairs"] % m]
            vk = gradk - state["gradk_prev"]
            yk_tmp = state["alpha_k_prev"] * vk
            betak = self._compute_beta(sk, yk_tmp)
            yk = betak * sk + (1 - betak) * yk_tmp
            # Keep reference to the y we've replaced, in case we need to discard yk
            y_tmp = y_hist[state["num_sy_pairs"] % m]
            # Set new curvature pair (sk, yk)
            y_hist[state["num_sy_pairs"] % m] = yk
            state["num_sy_pairs"] += 1

            # NOTE: Can't reliably check if pk is a descent direction here
            pk = -self._two_loop_recursion(gradk)

            if stable:
                # Check self-correcting assumptions using proxy batch
                sc_cond = self._check_self_correcting_conditions(xk, pk, proxy_fn)
                state["num_proxy_funcs"] += 1
                if sc_cond:
                    # Assumptions satisfied, reset counter and use this curvature pair
                    state["num_proxy_funcs"] = 0
                else:
                    # Not satisfied, discard this batch (undo curvature pair)
                    state["num_sy_pairs"] -= 1
                    y_hist[state["num_sy_pairs"] % m] = y_tmp
                    # If we haven't reached the max number of evals, try again with a
                    # different batch, else continue
                    if state["num_proxy_funcs"] < 2:
                        return orig_loss

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
            alpha_k = lr

        sk = alpha_k * pk
        self._add_param_vector(sk)

        # Can only store sk now, need next batch to compute yk
        s_hist[state["num_sy_pairs"] % m] = sk
        state["alpha_k_prev"] = alpha_k
        state["gradk_prev"] = gradk

        state["num_iters"] += 1
        return orig_loss
