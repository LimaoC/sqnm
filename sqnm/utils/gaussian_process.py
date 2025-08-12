import logging
from functools import wraps

import torch
from torch import Tensor

from ..utils.matrix import block_tensor

OFFSET = 10.0

logger = logging.getLogger(__name__)


def cached_computation(method):
    key = method.__name__

    @wraps(method)
    def wrapper(self, x):
        if key not in self.func_cache:
            self.func_cache[key] = {}

        if x in self.func_cache[key]:
            return self.func_cache[key][x]

        result = method(self, x)
        self.func_cache[key][x] = result
        return result

    return wrapper


class ProbLSGaussianProcess:
    def __init__(self, sigma_f, sigma_df):
        self.sigma_f = sigma_f
        self.sigma_df = sigma_df

        self.N = 0
        self.ts = []
        self.ys = []
        self.dys = []

        self.K = None
        self.Kd = None
        self.dKd = None

        self.A = None
        self.G = None

        self.func_cache = {}

    def add(self, t, y, dy, var_f=0.0, var_df=0.0):
        self.N += 1
        self.ts.append(t)
        self.ts_vec = torch.as_tensor(self.ts).t()
        self.ys.append(y)
        self.dys.append(dy)

        self.K = _k(self.ts_vec.unsqueeze(1), self.ts_vec.unsqueeze(0))
        self.Kd = _kd(self.ts_vec.unsqueeze(1), self.ts_vec.unsqueeze(0))
        self.dKd = _dkd(self.ts_vec.unsqueeze(1), self.ts_vec.unsqueeze(0))

        noise_vector = torch.tensor([self.sigma_f**2] * (2 * self.N))
        noise_vector[self.N :] = self.sigma_df**2
        k_matrix = block_tensor(self.K, self.Kd, self.Kd.t(), self.dKd)
        self.G = torch.diag(noise_vector) + k_matrix
        # Store LU decomposition of G so we can compute Gx = b
        self.LU, self.LU_pivots = torch.linalg.lu_factor(self.G)

        resid = torch.as_tensor(self.ys + self.dys).unsqueeze(1)
        self.A = torch.linalg.solve(self.G, resid)

        # Reset cache every time the GP is updated
        for func in self.func_cache:
            self.func_cache[func].clear()

    @cached_computation
    def _solve_G(self, b):
        """Solve Gx = b"""
        return torch.linalg.lu_solve(self.LU, self.LU_pivots, b)

    @cached_computation
    def mu(self, t):
        """Posterior mean of GP at t"""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        return _concat(_k(t, T), _kd(t, T)) @ self.A

    @cached_computation
    def d1mu(self, t):
        """First derivative of posterior mean of GP at t"""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        return _concat(_dk(t, T), _dkd(t, T)) @ self.A

    @cached_computation
    def d2mu(self, t):
        """Second derivative of posterior mean of GP at t"""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        return _concat(_ddk(t, T), _ddkd(t, T)) @ self.A

    @cached_computation
    def d3mu(self, t):
        """Third derivative of posterior mean of GP at t"""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        return _concat(_dddk(t, T), torch.zeros(1, self.N)) @ self.A

    @cached_computation
    def V(self, t):
        """Posterior variance of function values at t"""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        k_tt = _k(t, t)
        k_vector = _concat(_k(t, T), _kd(t, T))
        return k_tt - k_vector @ self._solve_G(k_vector.t())

    @cached_computation
    def Vd(self, t):
        """Posterior variance of function values and derivatives at t"""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        kd_tt = _kd(t, t)
        k_vector_1 = _concat(_k(t, T), _kd(t, T))
        k_vector_2 = _concat(_dk(t, T), _dkd(t, T)).t()
        return kd_tt - k_vector_1 @ self._solve_G(k_vector_2)

    @cached_computation
    def dVd(self, t):
        """Posterior variance of derivatives at t"""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        dkd_tt = _dkd(t, t)
        k_vector = _concat(_dk(t, T), _dkd(t, T))
        return dkd_tt - k_vector @ self._solve_G(k_vector.t())

    @cached_computation
    def V0f(self, t):
        """Posterior covariances of function values at t = 0 and t"""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        zero = torch.zeros(1)
        k_0t = _k(zero, t)
        k_vector_1 = _concat(_k(zero, T), _kd(zero, T))
        k_vector_2 = _concat(_k(t, T), _kd(t, T)).t()
        return k_0t - k_vector_1 @ self._solve_G(k_vector_2)

    @cached_computation
    def Vd0f(self, t):
        """Posterior covariance of gradient and function value at t = 0 and t resp."""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        zero = torch.zeros(1)
        dk_0t = _dk(zero, t)
        k_vector_1 = _concat(_dk(zero, T), _dkd(zero, T))
        k_vector_2 = _concat(_k(t, T), _kd(t, T)).t()
        return dk_0t - k_vector_1 @ self._solve_G(k_vector_2)

    @cached_computation
    def V0df(self, t):
        """Posterior covariance of function value and gradient at t = 0 and t resp."""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        zero = torch.zeros(1)
        kd_0t = _kd(zero, t)
        k_vector_1 = _concat(_k(zero, T), _kd(zero, T))
        k_vector_2 = _concat(_dk(t, T), _dkd(t, T)).t()
        return kd_0t - k_vector_1 @ self._solve_G(k_vector_2)

    @cached_computation
    def Vd0df(self, t):
        """Same as _V0f() but for gradients"""
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        T = self.ts_vec
        zero = torch.zeros(1)
        dkd_0t = _dkd(zero, t)
        k_vector_1 = _concat(_dk(zero, T), _dkd(zero, T))
        k_vector_2 = _concat(_dk(t, T), _dkd(t, T)).t()
        return dkd_0t - k_vector_1 @ self._solve_G(k_vector_2)


def _concat(a: Tensor, b: Tensor):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return torch.cat((a, b), dim=1)


def _k(a, b):
    c = torch.minimum(a + OFFSET, b + OFFSET)
    return (c**3) / 3 + 0.5 * torch.abs(a - b) * (c**2)


def _kd(a, b):
    aa = a + OFFSET
    bb = b + OFFSET
    return ((a < b) * 0.5 * aa**2) + ((a >= b) * aa * bb - 0.5 * (bb**2))


def _dk(a, b):
    return _kd(b, a)


def _dkd(a, b):
    return torch.minimum(a + OFFSET, b + OFFSET)


def _ddk(a, b):
    return (a <= b) * (b - a)


def _ddkd(a, b):
    return (a <= b).float()


def _dddk(a, b):
    return -(a <= b).float()
