import logging
from typing import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from ..utils.line_search import strong_wolfe

logger = logging.getLogger(__name__)


class LBFGS(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1,
        history_size: int = 20,
        grad_tol: float = 1e-4,
        line_search_fn: str | None = None,
    ):
        """
        Limited-memory BFGS Algorithm

        Parameters:
            lr: learning rate, only used if line_search_fn = None
            history_size: history size, usually 2 <= m <= 30
            grad_tol: termination tolerance for gradient norm
            line_search_fn: line search function to use, either None for fixed step
                size, or "strong_wolfe" for strong Wolfe line search

        REF: Algorithm 7.5 in Numerical Optimization by Nocedal and Wright
        """
        if lr <= 0:
            raise ValueError("LBFGS learning rate must be positive")
        if history_size < 1:
            raise ValueError("LBFGS history size must be positive")
        if line_search_fn is not None and line_search_fn != "strong_wolfe":
            raise ValueError("LBFGS only supports strong Wolfe line search")

        defaults = dict(
            lr=lr,
            history_size=history_size,
            grad_tol=grad_tol,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options")

        self._params: list[torch.nn.Parameter] = self.param_groups[0]["params"]

        d = self._get_param_vector().shape[0]
        m = history_size
        # Store LBFGS state in first param
        state = self.state[self._params[0]]
        state["num_iters"] = 0
        # Store the m previous (s, y) pairs
        # s is the iterate difference, y is the gradient difference
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

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:
        """REF: Algorithm 7.4 in Numerical Optimization by Nocedal and Wright"""
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        m = group["history_size"]
        k = state["num_iters"]
        sy_history = state["sy_history"]

        s_prev, y_prev = sy_history[k % m]
        H0 = s_prev.dot(y_prev) / y_prev.dot(y_prev) * torch.eye(grad.shape[0])

        if k <= m:
            return torch.matmul(H0, grad)
        q = grad.clone()
        alphas = torch.zeros(m)
        for i in range(k - 1, k - m - 1, -1):
            s_prev, y_prev = sy_history[i % m]
            alphas[i - (k - m)] = s_prev.dot(q) / s_prev.dot(y_prev)
            q -= alphas[i - (k - m)] * y_prev
        r = torch.matmul(H0, q)
        for i in range(k - m, k):
            s_prev, y_prev = sy_history[i % m]
            beta = y_prev.dot(r) / s_prev.dot(y_prev)
            r += (alphas[i - (k - m)] - beta) * s_prev
        return r

    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> float:  # type: ignore[override]
        """
        Perform a single L-BFGS iteration.

        Parameters:
            closure: A closure that re-evaluates the model and returns the loss.
        """
        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        state = self.state[self._params[0]]
        lr = group["lr"]
        m = group["history_size"]
        grad_tol = group["grad_tol"]
        line_search_fn = group["line_search_fn"]
        k = state["num_iters"]
        sy_history = state["sy_history"]

        orig_loss = closure()  # Populate gradients
        x_k = self._get_param_vector()
        grad = self._get_grad_vector()
        if grad.norm() < grad_tol:
            return orig_loss

        if k == 0:
            p_k = -grad  # Gradient descent for first iteration
        else:
            p_k = -self._two_loop_recursion(grad)
            if grad.dot(p_k) >= 0:
                logger.warning("p_k is not a descent direction.")

        def f(x: Tensor) -> float:
            """Objective function - also sets param vector to x"""
            self._set_param_vector(x)
            loss = closure()
            return loss

        def grad_f(x: Tensor) -> Tensor:
            """Gradient function - also sets param vector to x"""
            self._set_param_vector(x)
            closure()  # Recompute gradient after setting new param
            return self._get_grad_vector()

        if line_search_fn is not None:
            # Choose step size to satisfy strong Wolfe conditions
            alpha_k = strong_wolfe(f, grad_f, x_k, p_k)
            x_k_next = x_k + alpha_k * p_k
        else:
            # Use fixed step size
            x_k_next = x_k + lr * p_k
        # Compute and store next iterates
        grad_next = grad_f(x_k_next)
        sy_history[(k + 1) % m] = (x_k_next - x_k, grad_next - grad)

        state["num_iters"] += 1
        return orig_loss
