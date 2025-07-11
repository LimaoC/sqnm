import logging
from functools import cache
from typing import Callable

import numpy as np
from torch import Tensor

logger = logging.getLogger(__name__)


def strong_wolfe_line_search(
    f: Callable[[Tensor], float],
    grad_f: Callable[[Tensor], Tensor],
    xk: Tensor,
    pk: Tensor,
    a0: float = 1,
    a_max: float = 100,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iters: int = 200,
    zoom_max_iters: int = 20,
) -> float:
    """
    Finds an optimal step size that satisfies strong Wolfe conditions.

    Parameters:
        f: objective function, assumed to be bounded below along the direction p_k
        grad_f: gradient of objective function
        xk: current iterate
        pk: direction, assumed to be a descent direction
        a0: initial step size (1 should always be used as the initial step size for
            Newton and quasi-Newton methods)
        a_max: maximum step size
        c1: parameter for Armijo/sufficient decrease condition
        c2: parameter for curvature condition
        max_iters: max number of line search iterations to compute
        zoom_max_iters: max number of zoom() iterations to compute

    REF: Algorithm 3.5 in Numerical Optimization by Nocedal and Wright
    """

    @cache
    def phi(a_k: float) -> float:
        return f(xk + a_k * pk)

    @cache
    def grad_phi(a_k: float) -> float:
        return grad_f(xk + a_k * pk).dot(pk).item()

    def zoom(a_lo: float, a_hi: float) -> float:
        """REF: Algorithm 3.6 in Numerical Optimization by Nocedal and Wright"""

        z_iters = 0
        while z_iters < zoom_max_iters:
            z_iters += 1
            # Interpolate to find a trial step size a_j in [a_lo, a_hi]
            a_j = _cubic_interp(
                a_lo,
                a_hi,
                phi(a_lo),
                phi(a_hi),
                grad_phi(a_lo),
                grad_phi(a_hi),
            )
            # a_j should be in [a_lo, a_hi]... fallback to the midpoint if not
            if not _inside(a_j, a_lo, a_hi):
                a_j = (a_lo + a_hi) / 2

            # Armijo/sufficient decrease condition
            if (phi(a_j) > phi(0) + c1 * a_j * grad_phi(0)) or phi(a_j) >= phi(a_lo):
                a_hi = a_j
            else:
                # Curvature condition
                if np.abs(grad_phi(a_j)) <= -c2 * grad_phi(0):
                    break
                if grad_phi(a_j) * (a_hi - a_lo) >= 0:
                    a_hi = a_lo
                a_lo = a_j

        if z_iters == zoom_max_iters:
            logger.warning(
                "zoom() returning without satisfying strong Wolfe conditions, "
                "defaulting to alpha0 step size."
            )
            return a0
        return a_j

    a_prev = 0.0
    a_curr = a0
    a_star = a_curr  # Fallback, if something goes wrong
    phi_prev = phi(0)

    iters = 1
    while iters < max_iters:
        # Armijo/sufficient decrease condition
        armijo_cond = phi(a_curr) > phi(0) + c1 * a_curr * grad_phi(0)
        if armijo_cond or (phi(a_curr) >= phi_prev and iters > 1):
            a_star = zoom(a_prev, a_curr)
            break

        # Curvature condition
        if np.abs(grad_phi(a_curr)) <= -c2 * grad_phi(0):
            a_star = a_curr
            break

        if grad_phi(a_curr) >= 0:
            a_star = zoom(a_curr, a_prev)
            break

        a_prev, a_curr = a_curr, min(a_curr * 2, a_max)
        phi_prev = phi(a_prev)
        iters += 1
    return a_star


def _cubic_interp(
    x1: float, x2: float, f1: float, f2: float, grad_f1: float, grad_f2: float
) -> float:
    """
    Find the minimizer of the Hermite-cubic polynomial interpolating a function
    of one variable, at the two points x1 and x2, using the function values f(x_1) = f1
    and f(x_2) = f2 and derivatives grad_f(x_1) = grad_f1 and grad_f(x_2) = grad_f2.

    REF: Equation 3.59 in Numerical Optimization, Nocedal and Wright
    """
    d1 = grad_f1 + grad_f2 - 3 * (f1 - f2) / (x1 - x2)
    d2 = np.sign(x2 - x1) * np.sqrt(d1**2 - grad_f1 * grad_f2)
    xmin = x2 - (x2 - x1) * (grad_f2 + d2 - d1) / (grad_f2 - grad_f1 + 2 * d2)
    return xmin


def _inside(x: float, a: float, b: float) -> bool:
    """Returns whether x is in (a, b)"""
    if not np.isreal(x):
        return False

    a, b = min(a, b), max(a, b)
    return a <= x <= b
