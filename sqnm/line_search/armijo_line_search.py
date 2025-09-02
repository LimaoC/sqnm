import logging
from typing import Callable

from torch import Tensor

logger = logging.getLogger(__name__)


def armijo_line_search(
    fn: Callable[[Tensor], Tensor],
    grad_fn: Callable[[Tensor], Tensor],
    xk: Tensor,
    pk: Tensor,
    a0: float = 1,
    c: float = 1e-4,
) -> float:
    """
    Finds an optimal step size that the Armijo/sufficient decrease condition using a
    backtracking strategy.

    Parameters:
        fn: objective function, assumed to be bounded below along the direction p_k
        grad_fn: gradient of objective function
        xk: current iterate
        pk: direction, assumed to be a descent direction
        a0: initial step size
        c: parameter for Armijo/sufficient decrease condition

    REF: Algorithm 3.1 in Numerical Optimization by Nocedal and Wright
    """

    f_xk = fn(xk)
    grad_f_xk = grad_fn(xk)
    grad_f_xk_dot_pk = grad_f_xk.dot(pk)
    a_curr = a0

    while fn(xk + a_curr * pk) > f_xk + c * a_curr * grad_f_xk_dot_pk:
        a_curr /= 2
    return a_curr
