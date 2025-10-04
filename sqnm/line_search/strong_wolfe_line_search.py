import logging
from typing import Callable

import numpy as np
from torch import Tensor

logger = logging.getLogger(__name__)


def strong_wolfe_line_search(
    fn: Callable[[Tensor], tuple[Tensor, Tensor]],
    xk: Tensor,
    pk: Tensor,
    f_xk: float,
    grad_f_xk: Tensor,
    a0: float = 1.0,
    a_max: float = 10,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iters: int = 10,
    zoom_max_iters: int = 10,
) -> tuple[float, int]:
    """
    Finds and returns an optimal step size that satisfies strong Wolfe conditions,
    along with the number of function evaluations used.

    Parameters:
        fn: function to compute a tuple of (gradient, objective) for a given input,
            objective function is assumed to be bounded below along the direction p_k
        xk: current iterate
        pk: direction, assumed to be a descent direction
        f_xk: initial function value at xk
        grad_f_xk: initial gradient value at xk
        a0: initial step size (1 should always be used as the initial step size for
            Newton and quasi-Newton methods)
        a_max: maximum step size
        c1: parameter for Armijo/sufficient decrease condition
        c2: parameter for curvature condition
        max_iters: max number of line search iterations to compute
        zoom_max_iters: max number of zoom() iterations to compute

    REF: Algorithm 3.5 in Numerical Optimization by Nocedal and Wright
    """

    func_evals = 0

    a_prev = 0.0
    a_curr = a0
    a_star = a_curr
    grad_phi0 = grad_f_xk.dot(pk).item()
    phi_prev, grad_phi_prev = f_xk, grad_phi0

    best_armijo_a = a_curr
    best_armijo_phi = float("inf")

    def phi(a_k: float) -> tuple[float, float]:
        grad, val = fn(xk + a_k * pk)
        return grad.dot(pk).item(), val.item()

    def armijo_ok(phi_a: float, a: float) -> bool:
        return phi_a <= f_xk + c1 * a * grad_phi0

    def curvature_ok(grad_phi_a: float) -> bool:
        return abs(grad_phi_a) <= -c2 * grad_phi0

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
        nonlocal func_evals

        z_iters = 0
        while z_iters < zoom_max_iters:
            z_iters += 1
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

            grad_phi_j, phi_j = phi(a_j)
            func_evals += 2
            if not armijo_ok(phi_j, a_j) or phi_j >= phi_lo:
                # Narrow search to (a_lo, a_j) - unless a_lo == a_j
                if abs(a_lo - a_j) < 1e-9:
                    break
                a_hi, phi_hi, grad_phi_hi = a_j, phi_j, grad_phi_j
            else:
                if curvature_ok(grad_phi_j):
                    # a_j satisfies strong Wolfe conditions, stop here
                    break
                # Maintain condition (c)
                if grad_phi_j * (a_hi - a_lo) >= 0:
                    a_hi, phi_hi, grad_phi_hi = a_lo, phi_lo, grad_phi_lo
                # Maintain condition (b)
                a_lo, phi_lo, grad_phi_lo = a_j, phi_j, grad_phi_j

        # NOTE: Returning here without satisfying strong Wolfe conditions
        return a_j

    iters = 1
    while iters <= max_iters:
        grad_phi_curr, phi_curr = phi(a_curr)
        func_evals += 2

        armijo = armijo_ok(phi_curr, a_curr)

        if armijo and phi_curr < best_armijo_phi:
            best_armijo_a, best_armijo_phi = a_curr, phi_curr

        if not armijo or (phi_curr >= phi_prev and iters > 1):
            # (a_prev, a_curr) contains step lengths satisfying strong Wolfe conditions
            a_star = zoom(
                a_prev, a_curr, phi_prev, phi_curr, grad_phi_prev, grad_phi_curr
            )
            break

        if curvature_ok(grad_phi_curr):
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

    # Fallback to best Armijo point if Wolfe conditions aren't satisfied
    if iters > max_iters:
        return best_armijo_a, func_evals
    return a_star, func_evals


def _cubic_interp(
    x1: float, x2: float, f1: float, f2: float, g1: float, g2: float
) -> float:
    """
    Find the minimizer of the Hermite-cubic polynomial interpolating a function
    of one variable, at the two points x1 and x2, using the function values f(x_1) = f1
    and f(x_2) = f2 and derivatives grad_f(x_1) = g2 and grad_f(x_2) = g2.

    REF: Equation 3.59 in Numerical Optimization, Nocedal and Wright
    """
    if x1 == x2:
        return x1
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        try:
            d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
            discriminant = d1**2 - g1 * g2
            if discriminant < 0:
                # Default to midpoint
                return 0.5 * (x1 + x2)
            d2 = np.sign(x2 - x1) * np.sqrt(discriminant)
            return x2 - (x2 - x1) * (g2 + d2 - d1) / (g2 - g1 + 2 * d2)
        except ArithmeticError:
            # Default to midpoint
            return 0.5 * (x1 + x2)


def _inside(x: float, a: float, b: float) -> bool:
    """Returns whether x is in (a, b)"""
    if not np.isfinite(x):
        return False

    a, b = min(a, b), max(a, b)
    return a <= x <= b
