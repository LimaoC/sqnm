from typing import Callable

import torch
from torch import Tensor
from torch.optim import SGD

from sqnm.line_search import armijo_line_search
from sqnm.utils.param import grad_vec


class SGDWithLS(SGD):
    def __init__(self, batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state = self.state[self.param_groups[0]["params"][0]]
        group = self.param_groups[0]
        group["batch_size"] = batch_size
        state["func_evals"] = 0

    def _get_grad_vector(self) -> Tensor:
        """Concatenates gradients from all parameters into a 1D tensor"""
        params = self.param_groups[0]["params"]  # Assume 1 param group
        return grad_vec(params)

    def _get_param_vector(self) -> Tensor:
        """Concatenates all parameters into a 1D tensor"""
        params = self.param_groups[0]["params"]  # Assume 1 param group
        return torch.cat([p.data.view(-1) for p in params])

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float],
        fn: Callable[[Tensor], Tensor],
    ) -> float:
        """
        Perform a single SGD iteration.

        Parameters:
            closure: A closure that re-evaluates the model and returns the loss.
            fn: A pure function that computes the loss for a given input.
        """
        state = self.state[self.param_groups[0]["params"][0]]
        group = self.param_groups[0]
        batch_size = group["batch_size"]

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        orig_loss = closure()  # Populate gradients
        loss = float(orig_loss)
        state["func_evals"] += 2 * batch_size  # Forward + backward pass

        xk = self._get_param_vector()
        gradk = self._get_grad_vector()
        pk = -gradk

        # Choose step size to satisfy Armijo conditions
        alpha_k, ls_func_evals = armijo_line_search(fn, xk, pk, loss, gradk)
        state["func_evals"] += ls_func_evals * batch_size

        # Assume 1 param group
        self.param_groups[0]["lr"] = alpha_k

        # Don't pass closure() to step - we've already populated gradients
        super().step()
        return orig_loss
