import logging

import numpy as np

from ..types import ScalarFn, Vector, VectorFn

logger = logging.getLogger(__name__)


def strong_wolfe(
    f: ScalarFn,
    grad_f: VectorFn,
    x_k: Vector,
    p_k: Vector,
    alpha0: float = 1,
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
        x_k: current iterate
        p_k: direction, assumed to be a descent direction
        alpha0: initial step size (1 should always be used as the initial step size for
            Newton and quasi-Newton methods)
        c1: parameter for Armijo/sufficient decrease condition
        c2: parameter for curvature condition
        max_iters: max number of line search iterations to compute
        zoom_max_iters: max number of zoom() iterations to compute

    REF: Algorithm 3.5 in Numerical Optimization by Nocedal and Wright
    """

    def phi(alpha: float) -> float:
        return f(x_k + alpha * p_k)

    def grad_phi(alpha: float) -> float:
        return grad_f(x_k + alpha * p_k).T.dot(p_k)

    # Initial values
    phi0 = phi(0)  # Note that phi0 = f(xk)
    grad_phi0 = grad_phi(0)

    def zoom(a_lo: float, a_hi: float) -> float:
        """REF: Algorithm 3.6 in Numerical Optimization by Nocedal and Wright"""

        z_iters = 0
        while z_iters < zoom_max_iters:
            z_iters += 1
            # Interpolate to find a trial step size a_j in [a_lo, a_hi]
            a_j = cubic_interp(
                a_lo,
                a_hi,
                phi(a_lo),
                phi(a_hi),
                grad_phi(a_lo),
                grad_phi(a_hi),
            )
            # a_j should be in [a_lo, a_hi]... fallback to a_mid if not
            a_mid = (a_lo + a_hi) / 2
            if not inside(a_j, a_lo, a_mid):
                a_j = a_mid

            phi_a_j = phi(a_j)
            if (phi_a_j > phi0 + c1 * a_j * grad_phi0) or phi_a_j >= phi(a_lo):
                a_hi = a_j
            else:
                grad_phi_a_j = grad_phi(a_j)
                if np.abs(grad_phi_a_j) <= -c2 * grad_phi0:
                    break
                if grad_phi_a_j * (a_hi - a_lo) >= 0:
                    a_hi = a_lo
                a_lo = a_j

        if z_iters == zoom_max_iters:
            logger.warning(
                "zoom() returning without satisfying strong Wolfe conditions."
            )
        return a_j

    alpha_prev = 0.0
    alpha_curr = alpha0
    alpha_star = alpha_curr  # Fallback, if something goes wrong
    phi_prev = phi0

    iters = 1
    while iters < max_iters:
        phi_curr = phi(alpha_curr)
        if (phi_curr > phi0 + c1 * alpha_curr * grad_phi0) or (
            phi_curr >= phi_prev and iters > 1
        ):
            alpha_star = zoom(alpha_prev, alpha_curr)
            break

        grad_phi_curr = grad_phi(alpha_curr)
        if np.abs(grad_phi_curr) <= -c2 * grad_phi0:
            alpha_star = alpha_curr
            break
        if grad_phi_curr >= 0:
            alpha_star = zoom(alpha_curr, alpha_prev)
            break

        alpha_prev, alpha_curr = alpha_curr, alpha_curr * 2
        phi_prev = phi_curr
        iters += 1
    return alpha_star


def cubic_interp(
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


def inside(x: float, a: float, b: float) -> bool:
    """Returns whether x is in (a, b)"""
    if not np.isreal(x):
        return False

    a, b = min(a, b), max(a, b)
    return a <= x <= b
