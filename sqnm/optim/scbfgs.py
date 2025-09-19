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
        batch_size: int,
        lr: float = 1e-3,
        line_search_fn: str | None = None,
        history_size: int = 20,
        stable: bool = True,
        eta1: float = 1 / 4,
        eta2: float = 4,
        rho: float = 1 / 8,
        tau: float = 8,
        weight_decay: float = 0.0,
    ):
        """
        Self-Correcting BFGS (SC-BFGS)

        Parameters:
            params: iterable of parameters to optimize
            batch_size: used to keep track of number of function evaluations
            lr: learning rate, ignored if line_search_fn is not None
            line_search_fn: line search function to use, either None for fixed step
                size, or one of SCBFGS.LINE_SEARCH_FNS
            history_size: history size, usually 2 <= m <= 30
            stable: if True, the current update is skipped if self-correcting conditions
                aren't met for an independent proxy batch (skips up to two batches)
            eta1: lower-bound for s.dot(y)/||s||^2 in self-correcting theorem
            eta2: upper-bound for ||y||^2/s.dot(y) in self-correcting theorem
            rho: scaling constant in self-correcting assumption
            tau: shift constant in self-correcting assumption
            weight_decay: l2 regularisation term
        """
        if line_search_fn is not None and line_search_fn not in self.LINE_SEARCH_FNS:
            raise ValueError(f"SC-BFGS only supports one of: {self.LINE_SEARCH_FNS}")
        if eta1 <= 0 or eta1 >= 1:
            raise ValueError("eta1 should be in the range (0, 1)")
        if eta2 < 1:
            raise ValueError("eta2 should be in the range [1, inf)")

        defaults = dict(
            batch_size=batch_size,
            lr=lr,
            line_search_fn=line_search_fn,
            history_size=history_size,
            stable=stable,
            eta1=eta1,
            eta2=eta2,
            rho=rho,
            tau=tau,
            weight_decay=weight_decay,
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
        closure: Callable[[], Tensor],
        proxy_fn: Callable[[Tensor], Tensor] | None = None,
        fn: Callable[[Tensor], Tensor] | Callable[[Tensor, bool], Any] | None = None,
    ):
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        batch_size = group["batch_size"]
        lr = group["lr"]
        line_search_fn = group["line_search_fn"]
        m = group["history_size"]
        stable = group["stable"]
        weight_decay = group["weight_decay"]

        state = self.state[self._params[0]]
        k: int = state["num_iters"]
        s_hist: Tensor = state["s_hist"]
        y_hist: Tensor = state["y_hist"]
        alphas: list[float] = state["alphas"]
        sdotys: list[Tensor] = state["sdotys"]

        if line_search_fn is not None and fn is None:
            raise ValueError("fn parameter is needed for line search")
        if stable and proxy_fn is None:
            raise ValueError("proxy_fn parameter is needed for stable SC-BFGS")

        ################################################################################

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        orig_loss = closure()  # Populate gradients
        loss = float(orig_loss)
        state["func_evals"] += 2 * batch_size  # Forward + backward pass

        xk = self._get_param_vector()
        gradk = self._get_grad_vector()
        if weight_decay != 0:
            gradk.add_(xk, alpha=weight_decay)

        # NOTE: Termination criterion?

        if k == 0:
            pk = -gradk  # Gradient descent for first iteration
        else:
            # Compute (sk, yk) from previous iteration
            # sk computed already - need yk using this iteration's stochastic gradient
            sk = s_hist[state["num_sy_pairs"] % m]
            vk = gradk - state["gradk_prev"]
            yk_tmp = state["alpha_k_prev"] * vk
            betak = self._compute_beta(sk, yk_tmp)
            yk = betak * sk + (1 - betak) * yk_tmp
            if stable:
                # Keep reference to the y we've replaced, in case we need to discard yk
                y_tmp = y_hist[state["num_sy_pairs"] % m]
            # Set new curvature pair (sk, yk)
            alphas.append(state["alpha_k_prev"])
            sdotys.append(sk.dot(yk))
            y_hist[state["num_sy_pairs"] % m] = yk
            state["num_sy_pairs"] += 1

            # NOTE: Can't reliably check if pk is a descent direction here
            pk = -self._two_loop_recursion(gradk)

            if stable:
                assert proxy_fn is not None
                # Check self-correcting assumptions using proxy batch
                sc_cond = self._check_self_correcting_conditions(xk, pk, proxy_fn)
                state["func_evals"] += 2 * batch_size
                state["num_proxy_funcs"] += 1
                if sc_cond:
                    # Assumptions satisfied, reset counter and use this curvature pair
                    state["num_proxy_funcs"] = 0
                else:
                    # Not satisfied, discard this batch (undo curvature pair)
                    alphas.pop()
                    sdotys.pop()
                    y_hist[state["num_sy_pairs"] % m] = y_tmp
                    state["num_sy_pairs"] -= 1
                    # If we haven't reached the max number of evals, try again with a
                    # different batch, else continue
                    if state["num_proxy_funcs"] < 2:
                        return orig_loss

        if line_search_fn == "strong_wolfe":
            fn = cast(Callable[[Tensor], Tensor], fn)
            grad_and_val_fn = torch.func.grad_and_value(fn)
            alpha_k, ls_func_evals = strong_wolfe_line_search(
                grad_and_val_fn, xk, pk, loss, gradk
            )
            state["func_evals"] += ls_func_evals * batch_size
        elif line_search_fn == "prob_wolfe":
            fn = cast(Callable[[Tensor, bool], Any], fn)
            grad_and_val_fn = torch.func.grad_and_value(lambda x: fn(x, False))
            var_f0, var_df0 = fn(xk, True)
            # Don't need function handle to return variances in line search
            alpha_k, _, _, ls_func_evals = prob_line_search(
                grad_and_val_fn, xk, pk, loss, gradk, var_f0, var_df0
            )
            state["func_evals"] += ls_func_evals * batch_size
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
