import numpy as np


def strong_wolfe(f, grad_f, x_k, p_k, alpha0=1, c1=1e-4, c2=0.9, max_iters=200):
    """
    For Newton and quasi-Newton methods, alpha0=1 should always be used as the
    initial step size.

    We assume pk is a descent direction, and that f is bounded below along the direction
    p_k.

    REF: Algorithm 3.5 in Numerical Optimization by Nocedal and Wright
    """

    def phi(alpha):
        return f(x_k + alpha * p_k)

    def grad_phi(alpha):
        return grad_f(alpha).T.dot(p_k)

    # Initial values
    phi0 = phi(0)  # Note that phi0 = f(xk)
    grad_phi0 = grad_phi(0)

    def zoom(a_lo, a_hi):
        """REF: Algorithm 3.6 in Numerical Optimization by Nocedal and Wright"""
        while True:
            # Interpolate to find a trial step length alpha_j between alpha_lo and
            # a_hi
            a_j = cubic_interp(
                a_lo,
                a_hi,
                phi(a_lo),
                phi(a_hi),
                grad_phi(a_lo),
                grad_phi(a_hi),
            )

            phi_a_j = phi(a_j)
            if (phi_a_j > phi0 + c1 * a_j * grad_phi0) or phi_a_j >= phi(a_lo):
                a_hi = a_j
            else:
                grad_phi_a_j = grad_phi(a_j)
                if np.abs(grad_phi_a_j) <= -c2 * grad_phi0:
                    a_star = a_j
                    break
                if grad_phi_a_j * (a_hi - a_lo) >= 0:
                    a_hi = a_lo
                a_lo = a_j

        return a_star

    alpha_prev = 0
    alpha_curr = alpha0
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
        iters += 1

    return alpha_star


def cubic_interp(x1, x2, f1, f2, grad_f1, grad_f2):
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
