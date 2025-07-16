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

import numpy as np
import torch
from torch import Tensor

from .bvn import bvn_prob, std_norm_cdf, std_norm_pdf
from .gaussian_process import ProbLSGaussianProcess

logger = logging.getLogger(__name__)


def prob_line_search(
    f,
    x0: Tensor,  # [d]
    dir: Tensor,  # [d]
    f0,  # float
    df0,  # [d]
    var_f0,  # float
    var_df0: Tensor,  # [d]
    a0,
    a_stats,
):
    L = 6  # Max number of f evaluations per line search
    wolfe_threshold = 0.3

    # Scaling and noise level of GP
    beta = torch.abs(dir.t() @ df0).item()  # NOTE: df0 not var_df0 as in pseudocode
    sigma_f = (torch.sqrt(var_f0) / (a0 * beta)).item()
    sigma_df = (torch.sqrt((dir**2).t() @ var_df0) / beta).item()

    gp = ProbLSGaussianProcess(sigma_f, sigma_df)
    t_ext = 1.0  # Scaled step size for extrapolation
    tt = 1.0  # Scaled position of first function evaluation

    gp.add(0.0, 0.0, (df0.t() @ dir) / beta)
    gp.update()

    while gp.N < L + 1:
        y, dy, var_f, var_df = _evaluate_objective(f, f0, tt, x0, a0, dir, beta)
        gp.add(tt, y, dy, var_f, var_df)
        gp.update()

        if _prob_wolfe(tt, df0, gp) > wolfe_threshold:
            return _rescale_output(
                x0, f0, a0, dir, tt, y, dy, var_f, var_df, beta, a_stats
            )

        ms = [gp.mu(t) for t in gp.ts]
        dms = [gp.d1mu(t) for t in gp.ts]

        min_m_idx = np.argmin(ms)
        t_min, dm_min = gp.ts[min_m_idx], dms[min_m_idx]

        if torch.abs(dm_min) < 1e-5 and gp.Vd(t_min) < 1e-4:  # nearly deterministic
            y = gp.ys[min_m_idx]
            dy = gp.dys[min_m_idx]
            var_f = gp.var_fs[min_m_idx]
            var_df = gp.var_dfs[min_m_idx]
            return _rescale_output(
                x0, f0, a0, dir, t_min, y, dy, var_f, var_df, beta, a_stats
            )

        ts_sorted = sorted(gp.ts)
        ts_cand = []
        ms_cand = []
        ss_cand = []
        ts_wolfe = []

        for n in range(gp.N - 1):
            t_rep = ts_sorted[n] + 1e-6 * (ts_sorted[n + 1] - ts_sorted[n])
            t_cub_min = _cubic_minimum(t_rep, gp)

            if ts_sorted[n] < t_cub_min < ts_sorted[n + 1]:
                if (
                    not (torch.isnan(t_cub_min) or torch.isinf(t_cub_min))
                    and t_cub_min > 0
                ):
                    ts_cand.append(t_cub_min)
                    ms_cand.append(gp.mu(t_cub_min))
                    ss_cand.append(gp.V(t_cub_min))
            else:
                if n == 0 and gp.d1mu(0) > 0:
                    t_cand = 0.01 * (ts_sorted[n] + ts_sorted[n + 1])
                    y, dy, var_f, var_df = _evaluate_objective(
                        f, f0, t_cand, x0, a0, dir, beta
                    )
                    return _rescale_output(
                        x0, f0, a0, dir, t_cand, y, dy, var_f, var_df, beta, a_stats
                    )

            if n > 0 and _prob_wolfe(ts_sorted[n], df0, gp) > wolfe_threshold:
                ts_wolfe.append(ts_sorted[n])

        if len(ts_wolfe) > 0:
            ms_wolfe = [gp.mu(t) for t in ts_wolfe]
            min_m_idx = np.argmin(ms_wolfe)
            t_min = gp.ts[min_m_idx]
            y, dy, var_f, var_df = _evaluate_objective(f, f0, t_min, x0, a0, dir, beta)
            return _rescale_output(
                x0, f0, a0, dir, t_min, y, dy, var_f, var_df, beta, a_stats
            )

        t_max = max(gp.ts)
        ts_cand.append(t_max + t_ext)
        ms_cand.append(gp.mu(t_max + t_ext))
        ss_cand.append(torch.sqrt(gp.V(t_max + t_ext)))

        ei_cand = _expected_improvement(ms_cand, ss_cand, t_min)
        pw_cand = torch.tensor([_prob_wolfe(t, df0, gp) for t in ts_cand])

        t_best_cand = ts_cand[torch.argmax(ei_cand * pw_cand)]
        if t_best_cand == tt + t_ext:
            t_ext *= 2

        tt = t_best_cand

    # Reached limit without finding acceptable point
    # Evaluate a final time, return point with lowest function value
    y, dy, var_f, var_df = _evaluate_objective(f, f0, tt, x0, a0, dir, beta)
    gp.add(tt, y, dy, var_f, var_df)
    gp.update()

    if _prob_wolfe(tt, df0, gp) > wolfe_threshold:
        return _rescale_output(x0, f0, a0, dir, tt, y, dy, var_f, var_df, beta, a_stats)

    ms = [gp.mu(t) for t in gp.ts]
    t_lowest = gp.ts[np.argmin(ms)]
    if t_lowest == tt:
        return _rescale_output(x0, f0, a0, dir, tt, y, dy, var_f, var_df, beta, a_stats)

    tt = t_lowest
    y, dy, var_f, var_df = _evaluate_objective(f, f0, tt, x0, a0, dir, beta)
    return _rescale_output(x0, f0, a0, dir, tt, y, dy, var_f, var_df, beta, a_stats)


def _rescale_output(x0, f0, a0, dir, tt, y, dy, var_f, var_df, beta, a_stats):
    a_ext = 1.3
    theta_reset = 100

    a_acc = tt * a0
    x_acc = x0 + a_acc * dir
    f_acc = y * (a0 * beta) + f0
    df_acc = dy
    var_f_acc = var_f
    var_df_acc = var_df
    gamma = 0.95
    a_stats = gamma * a_stats + (1 - gamma) * a_acc
    a_next = a_acc * a_ext

    if (a_next < a_stats / theta_reset) or (a_next > a_stats * theta_reset):
        a_next = a_stats

    return float(a_next), a_stats, x_acc, f_acc, df_acc, var_f_acc, var_df_acc


def _evaluate_objective(f, f0, tt, x0, a0, dir, beta):
    y, dy = f(x0 + tt * a0 * dir)
    var_f, var_df = 0, 0
    y = (y - f0) / (a0 * beta)
    dy = (dy.t() @ dir) / beta
    return y, dy, var_f, var_df


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


def _prob_wolfe(t, df0, gp: ProbLSGaussianProcess, c1=0.05, c2=0.5, strong=True):
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

    # Covariance between conditions
    V_ab = (
        -c2 * (Vd0 + c1 * t * dVd0)
        + c2 * gp.Vd0f(t)
        + gp.V0df(t)
        + c1 * t * gp.Vd0df(t)
        - gp.Vd(t)
    )

    # Extremely small variances ==> very certain (deterministic evaluation)
    if V_aa <= 1e-9 and V_bb <= 1e-9:
        return (mu_a >= 0) * (mu_b >= 0)

    # Zero or negative variances (maybe something went wrong?)
    if V_aa <= 0 or V_bb <= 0:
        return 0.0

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
    z = (eta - m) / s  # .view(-1)
    return (eta - m) * std_norm_cdf(z) + s * std_norm_pdf(z)
