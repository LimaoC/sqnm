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
    EXCEPTION_STRATS = ["skip_update"]

    def __init__(
        self,
        params,
        batch_size: int,
        curvature_batch_size: int,
        lr: float = 1e-3,
        line_search_fn: str | None = None,
        history_size: int = 20,
        skip: int = 10,
        weight_decay: float = 0.0,
    ):
        """
        Hessian-Vector Stochastic Quasi-Newton (SQN-Hv)

        Parameters:
            params: iterable of parameters to optimize
            batch_size: used to keep track of number of function evaluations
            curvature_batch_size: used to keep track of number of function evaluations
            lr: learning rate, ignored if line_search_fn is not None
            line_search_fn: line search function to use, either None for fixed step
                size, or one of SQNHv.LINE_SEARCH_FNS
            history_size: history size, usually 2 <= m <= 30
            skip: number of iterations between curvature estimates
            weight_decay: l2 regularisation term
        """
        if line_search_fn is not None and line_search_fn not in self.LINE_SEARCH_FNS:
            raise ValueError(f"SQN-Hv only supports one of: {self.LINE_SEARCH_FNS}")

        defaults = dict(
            lr=lr,
            history_size=history_size,
            line_search_fn=line_search_fn,
            skip=skip,
            weight_decay=weight_decay,
            batch_size=batch_size,
            curvature_batch_size=curvature_batch_size,
        )
        super().__init__(params, defaults)

        state = self.state[self._params[0]]
        state["num_iters"] = 1  # Algorithm in paper starts from k = 1
        state["num_sy_pairs"] = -1
        state["xt"] = torch.zeros_like(self._get_param_vector())
        # Used for probabilistic LS
        if line_search_fn == "prob_wolfe":
            state["alpha_start"] = 1.0
            state["alpha_running_avg"] = state["alpha_start"]

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], Tensor],
        fn: Callable[[Tensor], Tensor] | Callable[[Tensor, bool], Any] | None = None,
        curvature_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        """
        Perform a single SQN-Hv iteration.

        Parameters:
            closure: A closure that re-evaluates the model and returns the loss.
            fn: A pure function that computes the loss for a given input. Required if
                using line search. For linesearch_fn == "prob_wolfe", the function
                should also take a boolean parameter which, if True, also returns the
                gradient, loss variance, and gradient variance.
            curvature_fn: A pure function that computes the loss for a given input. The
                function should be provided every `skip` iterations.
        """
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        lr = group["lr"]
        line_search_fn = group["line_search_fn"]
        m = group["history_size"]
        skip = group["skip"]  # L
        weight_decay = group["weight_decay"]
        batch_size = group["batch_size"]
        curvature_batch_size = group["curvature_batch_size"]

        state = self.state[self._params[0]]
        k: int = state["num_iters"]
        s_hist: Tensor = state["s_hist"]
        y_hist: Tensor = state["y_hist"]
        alphas: list[float] = state["alphas"]
        sdotys: list[Tensor] = state["sdotys"]
        # Note index t for curvature pairs, which are decoupled from gradient estimates
        xt: Tensor = state["xt"]

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
        loss = float(orig_loss)
        state["func_evals"] += 2 * batch_size  # Forward + backward pass

        xk = self._get_param_vector()
        gradk = self._get_grad_vector()
        if weight_decay != 0:
            gradk.add_(xk, alpha=weight_decay)

        # NOTE: Termination criterion?

        if k <= 2 * skip:
            # Stochastic gradient descent for first 2L iterations
            pk = -gradk
        else:
            # NOTE: Can't reliably check if pk is a descent direction here
            pk = -self._two_loop_recursion(gradk)

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
            # Don't need function handle to return vars in line search
            if k <= 2 * skip:
                # Propagate step sizes in probabilistic ls - we don't have curvature
                # information yet
                alpha_k, alpha_start, alpha_running_avg, ls_func_evals = (
                    prob_line_search(
                        grad_and_val_fn,
                        xk,
                        pk,
                        loss,
                        gradk,
                        var_f0,
                        var_df0,
                        a_running_avg=state["alpha_running_avg"],
                        a0=state["alpha_start"],
                    )
                )
                state["alpha_start"] = alpha_start
                state["alpha_running_avg"] = alpha_running_avg
            else:
                alpha_k, _, _, ls_func_evals = prob_line_search(
                    grad_and_val_fn, xk, pk, loss, gradk, var_f0, var_df0
                )
            state["func_evals"] += ls_func_evals * batch_size
        else:
            # Use fixed step size
            alpha_k = lr

        sk = alpha_k * pk
        self._add_param_vector(sk)

        # Accumulate average over L iterations
        xt.add_(xk + sk)

        if k % skip == 0:
            # Compute curvature pairs every L iterations
            t = state["num_sy_pairs"]
            state["num_sy_pairs"] += 1
            xt.div_(skip)
            if t >= 0:
                st = sk
                # Compute subsampled Hessian vector product on a different sample given
                # by curvature_fn
                _, yt = hvp(curvature_fn, xt, v=st, create_graph=False, strict=True)
                state["func_evals"] += 4 * curvature_batch_size
                sdotys.append(st.dot(yt))
                s_hist[t % m] = st
                y_hist[t % m] = yt
            xt.zero_()

        # Update state
        alphas.append(alpha_k)
        state["num_iters"] += 1

        return orig_loss
