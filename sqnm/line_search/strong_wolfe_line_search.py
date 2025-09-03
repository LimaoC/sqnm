import logging
from typing import Callable

import numpy as np
from torch import Tensor

logger = logging.getLogger(__name__)


def strong_wolfe_line_search(
    fn: Callable[[Tensor], Tensor],
    grad_fn: Callable[[Tensor], Tensor],
    xk: Tensor,
    pk: Tensor,
    phi0: float,
    grad_phi0: float,
    a0: float = 1,
    a_max: float = 100,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iters: int = 10,
    zoom_max_iters: int = 10,
) -> float:
    """
    Finds an optimal step size that satisfies strong Wolfe conditions.

    Parameters:
        fn: objective function, assumed to be bounded below along the direction p_k
        grad_fn: gradient of objective function
        xk: current iterate
        pk: direction, assumed to be a descent direction
        phi0: initial phi value, phi(0) = fn(xk)
        grad_phi0: initial grad_phi value, grad_phi(0) = grad_fn(xk).dot(pk)
        a0: initial step size (1 should always be used as the initial step size for
            Newton and quasi-Newton methods)
        a_max: maximum step size
        c1: parameter for Armijo/sufficient decrease condition
        c2: parameter for curvature condition
        max_iters: max number of line search iterations to compute
        zoom_max_iters: max number of zoom() iterations to compute

    REF: Algorithm 3.5 in Numerical Optimization by Nocedal and Wright
    """

    def phi(a_k: float) -> float:
        return fn(xk + a_k * pk).item()

    def grad_phi(a_k: float) -> float:
        return grad_fn(xk + a_k * pk).dot(pk).item()

    def zoom(
        a_lo: float,
        a_hi: float,
        phi_lo: float,
        phi_hi: float,
        grad_phi_lo: float,
        grad_phi_hi: float,
    ) -> float:
        """REF: Algorithm 3.6 in Numerical Optimization by Nocedal and Wright"""

        # Maintain three conditions in each iteration:
        # (a) The interval (a_lo, a_hi) contains step lengths that satisfy strong Wolfe
        # (b) a_lo gives the smallest function value among all step lengths generated so
        #     far that satisfy the sufficient decrease condition
        # (c) a_hi is chosen s.t. grad_phi(a_lo) * (a_hi - a_lo) < 0

        z_iters = 0
        while z_iters < zoom_max_iters:
            z_iters += 1
            if a_lo == a_hi:
                # Failsafe, break here
                a_j = a_lo
                break
            # Interpolate to find a trial step size a_j in (a_lo, a_hi)
            a_j = _cubic_interp(
                a_lo,
                a_hi,
                phi_lo,
                phi_hi,
                grad_phi_lo,
                grad_phi_hi,
            )
            # a_j should be in [a_lo, a_hi]... fallback to the midpoint if not
            if not _inside(a_j, a_lo, a_hi):
                a_j = (a_lo + a_hi) / 2

            phi_j = phi(a_j)
            # Armijo/sufficient decrease condition
            armijo_cond = phi_j <= phi0 + c1 * a_j * grad_phi0
            if not armijo_cond or phi_j >= phi_lo:
                # Narrow search to (a_lo, a_j) - unless a_lo == a_j
                if a_lo == a_j:
                    break
                a_hi, phi_hi, grad_phi_hi = a_j, phi_j, grad_phi(a_j)
            else:
                # (Modified) curvature condition
                grad_phi_j = grad_phi(a_j)
                if np.abs(grad_phi_j) <= -c2 * grad_phi0:
                    # a_j satisfies strong Wolfe conditions, stop here
                    break
                # Maintain condition (c)
                if grad_phi_j * (a_hi - a_lo) >= 0:
                    a_hi, phi_hi, grad_phi_hi = a_lo, phi_lo, grad_phi_lo
                # Maintain condition (b)
                a_lo, phi_lo, grad_phi_lo = a_j, phi_j, grad_phi_j

        # NOTE: Returning here without satisfying strong Wolfe conditions
        return a_j

    a_prev = 0.0
    a_curr = a0
    a_star = a_curr  # Fallback, if something goes wrong
    phi_prev = phi0
    grad_phi_prev = grad_phi0

    iters = 1
    while iters <= max_iters:
        # Armijo/sufficient decrease condition
        phi_curr = phi(a_curr)
        armijo_cond = phi_curr <= phi0 + c1 * a_curr * grad_phi0
        if not armijo_cond or (phi_curr >= phi_prev and iters > 1):
            # (a_prev, a_curr) contains step lengths satisfying strong Wolfe conditions
            a_star = zoom(
                a_prev, a_curr, phi_prev, phi_curr, grad_phi_prev, grad_phi(a_curr)
            )
            break

        # (Modified) curvature condition
        grad_phi_curr = grad_phi(a_curr)
        if np.abs(grad_phi_curr) <= -c2 * grad_phi0:
            # a_curr satisfies strong Wolfe conditions, stop here
            a_star = a_curr
            break

        if grad_phi_curr >= 0:
            # (a_prev, a_curr) contains step lengths satisfying strong Wolfe conditions
            a_star = zoom(
                a_curr, a_prev, phi_prev, phi_curr, grad_phi_prev, grad_phi_curr
            )
            break

        # Extrapolate to find next trial value (doubling strategy)
        a_prev, a_curr = a_curr, min(a_curr * 2, a_max)
        phi_prev, grad_phi_prev = phi_curr, grad_phi_curr
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
