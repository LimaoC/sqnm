import logging
from typing import Callable

from torch import Tensor

logger = logging.getLogger(__name__)


def armijo_line_search(
    fn: Callable[[Tensor], Tensor],
    xk: Tensor,
    pk: Tensor,
    f_xk: float,
    grad_f_xk: Tensor,
    a0: float = 1,
    c: float = 1e-4,
) -> tuple[float, int]:
    """
    Finds an optimal step size that the Armijo/sufficient decrease condition using a
    backtracking strategy, alnog with the number of function evaluations used.

    Parameters:
        fn: objective function, assumed to be bounded below along the direction p_k
        xk: current iterate
        pk: direction, assumed to be a descent direction
        f_xk: initial function value at xk
        grad_f_xk: initial gradient value at xk
        a0: initial step size
        c: parameter for Armijo/sufficient decrease condition

    REF: Algorithm 3.1 in Numerical Optimization by Nocedal and Wright
    """
    ls_func_evals = 0

    grad_f_xk_dot_pk = grad_f_xk.dot(pk)
    a_curr = a0

    while fn(xk + a_curr * pk) > f_xk + c * a_curr * grad_f_xk_dot_pk:
        ls_func_evals += 1
        a_curr /= 2
    ls_func_evals += 1
    return a_curr, ls_func_evals
