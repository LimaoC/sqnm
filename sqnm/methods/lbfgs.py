import numpy as np

from ..types import Matrix, ScalarFn, Vector, VectorFn
from ..utils.line_search import strong_wolfe


def l_bfgs(
    f: ScalarFn,
    grad_f: VectorFn,
    x0: Vector,
    m: int = 20,
    alpha0: float = 1,
    tau: float = 1e-4,
    max_iters: int = 1_000,
    callback=None,
) -> Vector:
    """
    Limited-memory BFGS Algorithm

    Parameters:
        f: objective function
        grad_f: gradient of objective function
        x_0: starting iterate
        m: history size for L-BFGS, usually 2 <= m <= 30
        alpha0: initial step size
        p_k: direction, assumed to be a descent direction
        alpha0: initial step size (1 should always be used as the initial step size for
            Newton and quasi-Newton methods)
        c1: parameter for Armijo/sufficient decrease condition
        c2: parameter for curvature condition
        max_iters: max number of line search iterations to compute
        callback: function to call at each iteration, takes the iteration number k,
            iterate x_k, function iterate f(x_k), and gradient norm ||grad_f(x_k)||

    REF: Algorithm 7.5 in Numerical Optimization by Nocedal and Wright
    """
    # Initial values
    k = 0
    d = x0.shape[0]
    x_k = np.copy(x0)
    grad_f_x_k = grad_f(x_k)

    # Store the m previous (s_k, y_k) = (x_{k+1} - x_k, grad f(x_{k+1}) - grad f(x_k))
    sy_iterates = np.array([(np.zeros(d), np.zeros(d)) for _ in range(m)])
    sy_iterates[0] = (x_k, grad_f_x_k)
    id = np.identity(d)

    def two_loop_recursion(H0: Matrix, grad_f_k: Vector) -> Vector:
        """REF: Algorithm 7.4 in Numerical Optimization by Nocedal and Wright"""
        if k <= m:
            return H0.dot(grad_f_k)
        q = np.copy(grad_f_k)
        alphas = np.zeros(m)
        for i in range(k - 1, k - m - 1, -1):
            s_prev, y_prev = sy_iterates[i % m]
            alphas[i - (k - m)] = s_prev.dot(q) / s_prev.dot(y_prev)
            q -= alphas[i - (k - m)] * y_prev
        r = H0.dot(q)
        for i in range(k - m, k):
            s_prev, y_prev = sy_iterates[i % m]
            beta = y_prev.dot(r) / s_prev.dot(y_prev)
            r += (alphas[i - (k - m)] - beta) * s_prev
        return r

    while np.linalg.norm(grad_f_x_k) > tau and k <= max_iters:
        if callback:
            callback(k, x_k, f(x_k), np.linalg.norm(grad_f_x_k))

        if k == 0:
            p_k = -grad_f_x_k
        else:
            s_prev, y_prev = sy_iterates[k % m]
            # Standard method for choosing H0_k from equation (7.20)
            H0_k = s_prev.dot(y_prev) / y_prev.dot(y_prev) * id
            p_k = -two_loop_recursion(H0_k, grad_f_x_k)

        # Choose step size to satisfy strong Wolfe conditions
        alpha_k = strong_wolfe(f, grad_f, x_k, p_k, alpha0)

        # Compute next iterates and store them
        x_k_next = x_k + alpha_k * p_k
        grad_f_x_k_next = grad_f(x_k_next)
        sy_iterates[(k + 1) % m] = (x_k_next - x_k, grad_f_x_k_next - grad_f_x_k)
        x_k, grad_f_x_k = x_k_next, grad_f_x_k_next
        k += 1
    if callback:
        callback(k, x_k, f(x_k), np.linalg.norm(grad_f_x_k))
    return x_k
