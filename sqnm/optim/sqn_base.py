"""
Base for stochastic quasi-Newton (SQN) optimizers
"""

import logging
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

logger = logging.getLogger(__name__)


class SQNBase(Optimizer):
    """Base for stochastic quasi-Newton (SQN) optimizers"""

    def __init__(self, params: ParamsT, defaults: dict[str, Any]):
        if not defaults.get("history_size"):
            raise ValueError("L-BFGS type optimizers must have a history size")
        if defaults["history_size"] < 1:
            raise ValueError("History size must be positive")

        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "L-BFGS type optimizers don't support per-parameter options"
            )

        self._params: list[torch.nn.Parameter] = self.param_groups[0]["params"]

        # Store LBFGS state in first param
        state = self.state[self._params[0]]
        state["num_iters"] = 0
        # Store the m most recent (s, y) pairs
        # s is the iterate difference, y is the gradient difference
        d = self._get_param_vector().shape[0]
        m = defaults["history_size"]
        state["sy_history"] = [(torch.zeros(d), torch.zeros(d)) for _ in range(m)]

    def _get_grad_vector(self) -> Tensor:
        """Concatenates gradients from all parameters into a 1D tensor"""
        grads = []
        for param in self._params:
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        return torch.cat(grads)

    def _get_param_vector(self) -> Tensor:
        """Concatenates all parameters into a 1D tensor"""
        return torch.cat([p.data.view(-1) for p in self._params])

    def _set_param_vector(self, vec: Tensor):
        """Set model parameters to the given tensor"""
        offset = 0
        for param in self._params:
            numel = param.numel()
            param.data.copy_(vec[offset : offset + numel].view_as(param))
            offset += numel

    def _two_loop_recursion_check_curvature_pairs(self):
        """
        Check that we're accessing the correct (s, y) pairs when computing the two loop
        recursion - they should all be non-zero vectors
        """
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        k = state["num_iters"]
        sy_history = state["sy_history"]

        for i in range(max(k - m, 0), k):
            s_prev, y_prev = sy_history[i % m]
            if torch.all(s_prev == 0) or torch.all(y_prev == 0):
                logger.warning(
                    "Found a (s, y) pair that is all zero - this is likely an error"
                )

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:
        """
        Two loop recursion for computing H_k * grad

        This may be overridden if a particular optimiser uses a different algorithm,
        e.g. a different starting matrix H_k^(0).

        REF: Algorithm 7.4 in Numerical Optimization by Nocedal and Wright
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
        for i in reversed(history_idxs):
            s_prev, y_prev = sy_history[i % m]
            alphas[i - (k - m)] = s_prev.dot(q) / s_prev.dot(y_prev)
            q -= alphas[i - (k - m)] * y_prev
        r = (s_prev.dot(y_prev) / y_prev.dot(y_prev)) * q
        for i in history_idxs:
            s_prev, y_prev = sy_history[i % m]
            beta = y_prev.dot(r) / s_prev.dot(y_prev)
            r += (alphas[i - (k - m)] - beta) * s_prev
        return r
