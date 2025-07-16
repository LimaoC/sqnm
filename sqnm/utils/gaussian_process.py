import logging
from functools import cache

import numpy as np
import torch
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal
from torch import Tensor

from ..utils.matrix import block_tensor

logger = logging.getLogger(__name__)


class ProbLSGaussianProcess:
    def __init__(self, sigma_f, sigma_df, offset=10.0):
        # Hyperparameters
        self.offset = offset

        self.sigma_f = sigma_f
        self.sigma_df = sigma_df

        self.N = 0
        self.ts = []
        self.ys = []
        self.dys = []
        self.var_fs = []
        self.var_dfs = []

        self.K = None
        self.Kd = None
        self.dKd = None

        self.A = None
        self.G = None

    def add(self, t, y, dy, var_f=0.0, var_df=0.0):
        self.N += 1
        self.ts.append(t)
        self.ys.append(y)
        self.dys.append(dy)
        self.var_fs.append(var_f)
        self.var_dfs.append(var_df)

    def update(self):
        self.K = torch.zeros((self.N, self.N))
        self.Kd = torch.zeros((self.N, self.N))
        self.dKd = torch.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(self.N):
                self.K[i, j] = self._k(self.ts[i], self.ts[j])
                self.Kd[i, j] = self._kd(self.ts[i], self.ts[j])
                self.dKd[i, j] = self._dkd(self.ts[i], self.ts[j])

        noise_vector = torch.tensor([self.sigma_f**2] * (2 * self.N))
        noise_vector[self.N :] = self.sigma_df**2
        k_matrix = block_tensor(self.K, self.Kd, self.Kd.t(), self.dKd)
        self.G = torch.diag(noise_vector) + k_matrix

        resid = torch.as_tensor(self.ys + self.dys).unsqueeze(1)
        self.A = torch.linalg.solve(self.G, resid)

    def _concat(self, a: Tensor, b: Tensor):
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        return torch.cat((a, b), dim=1)

    def mu(self, t):
        """Posterior mean of GP at t"""
        T = torch.as_tensor(self.ts)
        return self._concat(self._k(t, T.t()), self._kd(t, T.t())) @ self.A

    def d1mu(self, t):
        """First derivative of posterior mean of GP at t"""
        T = torch.as_tensor(self.ts)
        return self._concat(self._dk(t, T.t()), self._dkd(t, T.t())) @ self.A

    def d2mu(self, t):
        """Second derivative of posterior mean of GP at t"""
        T = torch.as_tensor(self.ts)
        return self._concat(self._ddk(t, T.t()), self._ddkd(t, T.t())) @ self.A

    def d3mu(self, t):
        """Third derivative of posterior mean of GP at t"""
        T = torch.as_tensor(self.ts)
        return self._concat(self._dddk(t, T.t()), torch.zeros(1, self.N)) @ self.A

    def V(self, t):
        """Posterior variance of function values at t"""
        T = torch.as_tensor(self.ts)
        k_tt = self._k(t, t)
        k_vector = self._concat(self._k(t, T.t()), self._kd(t, T.t()))
        return k_tt - k_vector @ torch.linalg.solve(self.G, k_vector.t())

    def Vd(self, t):
        """Posterior variance of function values and derivatives at t"""
        T = torch.as_tensor(self.ts)
        kd_tt = self._kd(t, t)
        k_vector_1 = self._concat(self._k(t, T.t()), self._kd(t, T.t()))
        k_vector_2 = self._concat(self._dk(t, T.t()), self._dkd(t, T.t())).t()
        return kd_tt - k_vector_1 @ torch.linalg.solve(self.G, k_vector_2)

    def dVd(self, t):
        """Posterior variance of derivatives at t"""
        T = torch.as_tensor(self.ts)
        dkd_tt = self._dkd(t, t)
        k_vector = self._concat(self._dk(t, T.t()), self._dkd(t, T.t()))
        return dkd_tt - k_vector @ torch.linalg.solve(self.G, k_vector.t())

    def V0f(self, t):
        """Posterior covariances of function values at t = 0 and t"""
        T = torch.as_tensor(self.ts)
        k_0t = self._k(0, t)
        k_vector_1 = self._concat(self._k(0, T.t()), self._kd(0, T.t()))
        k_vector_2 = self._concat(self._k(t, T.t()), self._kd(t, T.t())).t()
        return k_0t - k_vector_1 @ torch.linalg.solve(self.G, k_vector_2)

    def Vd0f(self, t):
        """Posterior covariance of gradient and function value at t = 0 and t resp."""
        T = torch.as_tensor(self.ts)
        dk_0t = self._dk(0, t)
        k_vector_1 = self._concat(self._dk(0, T.t()), self._dkd(0, T.t()))
        k_vector_2 = self._concat(self._k(t, T.t()), self._kd(t, T.t())).t()
        return dk_0t - k_vector_1 @ torch.linalg.solve(self.G, k_vector_2)

    def V0df(self, t):
        """Posterior covariance of function value and gradient at t = 0 and t resp."""
        T = torch.as_tensor(self.ts)
        kd_0t = self._kd(0, t)
        k_vector_1 = self._concat(self._k(0, T.t()), self._kd(0, T.t()))
        k_vector_2 = self._concat(self._dk(t, T.t()), self._dkd(t, T.t())).t()
        return kd_0t - k_vector_1 @ torch.linalg.solve(self.G, k_vector_2)

    def Vd0df(self, t):
        """Same as _V0f() but for gradients"""
        T = torch.as_tensor(self.ts)
        dkd_0t = self._dkd(0, t)
        k_vector_1 = self._concat(self._dk(0, T.t()), self._dkd(0, T.t()))
        k_vector_2 = self._concat(self._dk(t, T.t()), self._dkd(t, T.t())).t()
        return dkd_0t - k_vector_1 @ torch.linalg.solve(self.G, k_vector_2)

    @cache
    def _k(self, a, b):
        a = torch.as_tensor(a)
        b = torch.as_tensor(b)
        a = a.to(b)
        c = torch.minimum(a + self.offset, b + self.offset)
        return (c**3) / 3 + 0.5 * torch.abs(a - b) * (c**2)

    @cache
    def _kd(self, a, b):
        a = torch.as_tensor(a)
        b = torch.as_tensor(b)
        a = a.to(b)
        return ((a < b) * 0.5 * (a + self.offset) ** 2) + (
            (a >= b) * (a + self.offset) * (b + self.offset)
            - 0.5 * ((b + self.offset) ** 2)
        )

    @cache
    def _dk(self, a, b):
        return self._kd(b, a)

    @cache
    def _dkd(self, a, b):
        a = torch.as_tensor(a)
        b = torch.as_tensor(b)
        a = a.to(b)
        return torch.minimum(a + self.offset, b + self.offset)

    @cache
    def _ddk(self, a, b):
        a = torch.as_tensor(a)
        b = torch.as_tensor(b)
        a = a.to(b)
        return (a <= b) * (b - a)

    @cache
    def _ddkd(self, a, b):
        a = torch.as_tensor(a)
        b = torch.as_tensor(b)
        a = a.to(b)
        return (a <= b).float()

    @cache
    def _dddk(self, a, b):
        a = torch.as_tensor(a)
        b = torch.as_tensor(b)
        a = a.to(b)
        return -(a <= b).float()
