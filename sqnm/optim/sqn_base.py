"""
Base for stochastic quasi-Newton (SQN) optimizers
"""

import logging
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from sqnm.utils.param import grad_vec

logger = logging.getLogger(__name__)


class SQNBase(Optimizer):
    """Base for stochastic quasi-Newton (SQN) optimizers"""

    def __init__(self, params: ParamsT, defaults: dict[str, Any]):
        if "lr" in defaults and defaults["lr"] <= 0:
            raise ValueError("Learning rate must be positive")
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
        with torch.no_grad():
            self._flat_params = parameters_to_vector(self._params).detach()

        # Store LBFGS state in first param
        state = self.state[self._params[0]]
        state["num_iters"] = 0
        # Store the m most recent (s, y) pairs
        # s is the iterate difference, y is the gradient difference
        m = defaults["history_size"]
        param = self._get_param_vector()
        state["s_hist"] = torch.zeros(m, param.numel(), device=param.device)
        state["y_hist"] = torch.zeros(m, param.numel(), device=param.device)
        state["num_sy_pairs"] = 0  # Also the next index to insert into

    def _get_grad_vector(self) -> Tensor:
        """Concatenates gradients from all parameters into a 1D tensor"""
        return grad_vec(self._params)

    def _get_param_vector(self) -> Tensor:
        """Returns parameter vector as a flat (1D) tensor"""
        return self._flat_params

    def _add_param_vector(self, vec: torch.Tensor):
        """Add given flat (1D) tensor to model parameters"""
        with torch.no_grad():
            self._flat_params.add_(vec)
            torch.nn.utils.vector_to_parameters(self._flat_params, self._params)

    def _set_param_vector(self, vec: Tensor):
        """Set model parameters to the given flat (1D) tensor"""
        with torch.no_grad():
            self._flat_params.copy_(vec)
            torch.nn.utils.vector_to_parameters(self._flat_params, self._params)

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:
        """
        Two loop recursion for computing H_k * grad

        This may be overridden if a particular optimiser uses a different algorithm,
        e.g. a different starting matrix H_k^(0).

        REF: Algorithm 7.4 in Numerical Optimization by Nocedal and Wright
        """
        group = self.param_groups[0]
        m = group["history_size"]

        state = self.state[self._params[0]]
        k = state["num_sy_pairs"]
        h = min(k, m)  # Number of curvature pairs to use
        idxs = torch.arange(k - h, k) % m
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
