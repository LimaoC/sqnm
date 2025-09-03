"""
Probabilistic Line Search

REF: Mahsereci, M., & Hennig, P. (2016). Probabilistic Line Searches for Stochastic Optimization.

Maren Mahsereci and Philipp Hennig copyright statement:
***
Copyright (c) 2015 (post NIPS 2015 release 4.0), Maren Mahsereci, Philipp Hennig
mmahsereci@tue.mpg.de, phennig@tue.mpg.de
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import logging
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from ..utils.bvn import bvn_prob, std_norm_cdf, std_norm_pdf
from ..utils.gaussian_process import ProbLSGaussianProcess

logger = logging.getLogger(__name__)


def prob_line_search(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,  # [d]
    dir: Tensor,  # [d]
    f0: float,
    df0: Tensor,  # [d]
    var_f0: float,  # float
    var_df0: Tensor,  # [d]
    a_running_avg: float | None = None,
    a0: float = 1.0,  # QN methods should try a step size of 1.0 first
    L: int = 10,  # max number of function evaluations
    wolfe_threshold: float = 0.3,
) -> tuple[float, float, float]:
    grad_fn = torch.func.grad(fn)

    # Scaling and noise level of GP
    beta = torch.abs(dir @ df0).item()  # NOTE: df0 not var_df0 as in pseudocode
    sigma_f = (var_f0 ** (1 / 2)) / (a0 * beta)
    sigma_df = (torch.sqrt((dir**2) @ var_df0) / beta).item()

    gp = ProbLSGaussianProcess(sigma_f, sigma_df)
    tt_ext = 1.0  # Scaled step size for extrapolation
    tt = 1.0  # Scaled step size of first function evaluation (actual is tt * a0)

    gp.add(0.0, 0.0, (df0 @ dir) / beta)

    while gp.N < L - 1:
        # Evaluate objective at tt, check if Wolfe prob. is above threshold
        y, dy = _evaluate_objective(fn, grad_fn, f0, tt, x0, a0, dir, beta)
        gp.add(tt, y, dy)
        if _prob_wolfe(tt, gp) > wolfe_threshold:
            return _rescale_output(a0, tt, a_running_avg)

        # Find suitable candidates for next evaluation
        ms = [gp.mu(t) for t in gp.ts]
        dms = [gp.d1mu(t) for t in gp.ts]

        min_m_idx = np.argmin(ms)
        m_min, tt_min, dm_min = ms[min_m_idx], gp.ts[min_m_idx], dms[min_m_idx]

        # Check for nearly deterministic (small variance) step size with near zero grad
        if torch.abs(dm_min) < 1e-5 and gp.Vd(tt_min) < 1e-4:
            # Stop here - though Wolfe conditions may not be guaranteed
            y = gp.ys[min_m_idx]
            dy = gp.dys[min_m_idx]
            return _rescale_output(a0, tt, a_running_avg)

        # Sort both gp.ts and ms, according to sort order for gp.ts
        ts_sorted, ms_sorted = zip(*sorted(zip(gp.ts, ms)))
        ts_cand = []
        ms_cand = []
        ss_cand = []

        ts_wolfe = []
        ms_wolfe = []

        # Find local minima in all N - 1 cells
        for n in range(gp.N - 1):
            tt_rep = ts_sorted[n] + 1e-6 * (ts_sorted[n + 1] - ts_sorted[n])
            tt_cub_min = _cubic_minimum(tt_rep, gp)

            # If cubic minimum lies in [T_n, T_{n+1}], it is a candidate
            # Otherwise, check if we are going uphill, and break early if so
            if ts_sorted[n] < tt_cub_min < ts_sorted[n + 1]:
                if (
                    not (torch.isnan(tt_cub_min) or torch.isinf(tt_cub_min))
                    and tt_cub_min > 0
                ):
                    ts_cand.append(tt_cub_min.item())
                    ms_cand.append(gp.mu(tt_cub_min))
                    ss_cand.append(torch.sqrt(gp.V(tt_cub_min)))  # NOTE: Take sqrt here
            elif n == 0 and gp.d1mu(0) > 0:
                tt = 0.01 * (ts_sorted[n] + ts_sorted[n + 1])
                return _rescale_output(a0, tt, a_running_avg)

            # Check if an old step size is now acceptable
            if n > 0 and _prob_wolfe(ts_sorted[n], gp) > wolfe_threshold:
                ts_wolfe.append(ts_sorted[n])
                ms_wolfe.append(ms_sorted[n])

        # If any points are acceptable, return the step size with the lowest GP mean,
        # with preference for the current step size tt
        if len(ts_wolfe) > 0:
            if tt in ts_wolfe:
                return _rescale_output(a0, tt, a_running_avg)
            tt = ts_wolfe[np.argmin(ms_wolfe)]
            return _rescale_output(a0, tt, a_running_avg)

        tt_next = max(gp.ts) + tt_ext
        ts_cand.append(tt_next)
        ms_cand.append(gp.mu(tt_next))
        ss_cand.append(torch.sqrt(gp.V(tt_next)))

        # m_min: Collected (not candidate) step size with lowest GP mean
        ei_cand = _expected_improvement(ms_cand, ss_cand, m_min)
        pw_cand = torch.tensor([_prob_wolfe(t, gp) for t in ts_cand])
        t_best_cand = ts_cand[torch.argmax(ei_cand * pw_cand)]

        # Extend extrapolation step if necessary
        if t_best_cand == tt_next:
            tt_ext *= 2
        tt = t_best_cand

    # Reached budget without finding acceptable point, check final point for acceptance
    y, dy = _evaluate_objective(fn, grad_fn, f0, tt, x0, a0, dir, beta)
    gp.add(tt, y, dy)
    if _prob_wolfe(tt, gp) > wolfe_threshold:
        return _rescale_output(a0, tt, a_running_avg)

    # Otherwise - return point with lowest GP mean
    ms = [gp.mu(t) for t in gp.ts[1:]]
    tt = gp.ts[np.argmin(ms) + 1]
    return _rescale_output(a0, tt, a_running_avg)


def _rescale_output(a0, tt, a_running_avg) -> tuple[float, float, float]:
    a_accepted = tt * a0
    if a_running_avg is None:
        return float(a_accepted), -1, -1

    a_ext = 1.3
    theta_reset = 100
    gamma = 0.95

    a_running_avg = gamma * a_running_avg + (1 - gamma) * a_accepted
    a_next = a_accepted * a_ext

    # Reset scaling if new GP scaling is drastically different from previous scalings
    if (a_next < a_running_avg / theta_reset) or (a_next > a_running_avg * theta_reset):
        a_next = a_running_avg

    return float(a_accepted), float(a_next), float(a_running_avg)


def _evaluate_objective(fn, grad_fn, f0, tt, x0, a0, dir, beta):
    """Evaluates and re-scales function and gradient value"""
    x = x0 + tt * a0 * dir
    y = (fn(x) - f0) / (a0 * beta)
    dy = (grad_fn(x) @ dir) / beta
    return y, dy


def _cubic_minimum(t, gp: ProbLSGaussianProcess):
    d1m_t = gp.d1mu(t)
    d2m_t = gp.d2mu(t)
    d3m_t = gp.d3mu(t)
    a = 0.5 * d3m_t
    b = d2m_t - t * d3m_t
    c = d1m_t - d2m_t * t + 0.5 * d3m_t * (t**2)

    if torch.abs(d3m_t) < 1e-9:
        return -(d1m_t - t * d2m_t) / d2m_t

    det = b**2 - 4 * a * c
    if det < 0:
        return float("inf")

    lr = (-b - torch.sign(a) * torch.sqrt(det)) / (2 * a)
    rr = (-b + torch.sign(a) * torch.sqrt(det)) / (2 * a)

    dt_l = lr - t
    dt_r = rr - t
    cv_l = d1m_t * dt_l + 0.5 * d2m_t * (dt_l**2) + (d3m_t * (dt_l**3)) / 6
    cv_r = d1m_t * dt_r + 0.5 * d2m_t * (dt_r**2) + (d3m_t * (dt_r**3)) / 6

    if cv_l < cv_r:
        return lr
    return rr


def _prob_wolfe(t, gp: ProbLSGaussianProcess, c1=0.05, c2=0.5):
    # Mean and covariance values at start position t = 0
    mu0 = gp.mu(0)
    dmu0 = gp.d1mu(0)
    V0 = gp.V(0)
    Vd0 = gp.Vd(0)
    dVd0 = gp.dVd(0)

    # Marginal mean and variance for Armijo condition
    mu_a = mu0 - gp.mu(t) + c1 * t * dmu0
    V_aa = (
        V0
        + ((c1 * t) ** 2) * dVd0
        + gp.V(t)
        + 2 * (c1 * t * (Vd0 - gp.Vd0f(t)) - gp.V0f(t))
    )

    # Marginal mean and variance for curvature condition
    mu_b = gp.d1mu(t) - c2 * dmu0
    V_bb = (c2**2) * dVd0 - 2 * c2 * gp.Vd0df(t) + gp.dVd(t)

    # Extremely small variances ==> very certain (deterministic evaluation)
    if V_aa <= 1e-9 and V_bb <= 1e-9:
        return float((mu_a >= 0) * (mu_b >= 0))

    # Zero or negative variances (maybe something went wrong?)
    if V_aa <= 0 or V_bb <= 0:
        return 0.0

    # Covariance between conditions
    V_ab = (
        -c2 * (Vd0 + c1 * t * dVd0)
        + c2 * gp.Vd0f(t)
        + gp.V0df(t)
        + c1 * t * gp.Vd0df(t)
        - gp.Vd(t)
    )

    # Noisy case (everything is alright)
    # Correlation
    rho = (V_ab / torch.sqrt(V_aa * V_bb)).item()

    # Lower and upper integral limits for Armijo condition
    low_a = float(-mu_a / torch.sqrt(V_aa))
    up_a = float("inf")

    # Lower and upper integral limits for curvature condition
    low_b = float(-mu_b / torch.sqrt(V_bb))
    b_bar = 2 * c2 * (torch.abs(dmu0) + 2 * torch.sqrt(dVd0))
    up_b = float((b_bar - mu_b) / torch.sqrt(V_bb))

    return bvn_prob(low_a, up_a, low_b, up_b, rho)


def _expected_improvement(m, s, eta):
    m = torch.as_tensor(m)
    s = torch.as_tensor(s)
    m = m.to(s)
    z = (eta - m) / s
    return (eta - m) * std_norm_cdf(z) + s * std_norm_pdf(z)
