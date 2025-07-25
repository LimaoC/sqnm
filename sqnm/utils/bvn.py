"""
Translated from a MATLAB implementation by Alan Genz which is based on the method
described by
    Drezner, Z and G.O. Wesolowsky, (1989),
    On the computation of the bivariate normal inegral,
    Journal of Statist. Comput. Simul. 35, pp. 101-107,
with major modifications for double precision, for |r| close to 1.

Alan Genz copyright statement:
***
Copyright (C) 2013, Alan Genz,  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided the following conditions are met:
  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in
     the documentation and/or other materials provided with the
     distribution.
  3. The contributor name(s) may not be used to endorse or promote
     products derived from this software without specific prior
     written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from scipy.special import erf  # type: ignore[import-untyped]


def bvn_prob(xl, xu, yl, yu, rho):
    """
    Compute the bivariate normal probability that xl < x < xu and yl < y < yu,
    with mean 0 and covariance matrix [[1, rho], [rho, 1]].
    """
    p = (
        bvn_prob_unbounded(xl, yl, rho)
        - bvn_prob_unbounded(xu, yl, rho)
        - bvn_prob_unbounded(xl, yu, rho)
        + bvn_prob_unbounded(xu, yu, rho)
    )
    return max(0.0, min(1.0, p))


def bvn_prob_unbounded(xl, yl, rho):
    """
    Compute the bivariate normal probability that x > xl and y > yl,
    with mean 0 and covariance matrix [[1, rho], [rho, 1]]
    """
    rho = max(-1.0, min(1.0, rho))
    if np.isposinf(xl) or np.isposinf(yl):
        return 0.0
    if np.isneginf(xl):
        return 1.0 if np.isneginf(yl) else std_norm_cdf(-yl)
    if np.isneginf(yl):
        return std_norm_cdf(-xl)
    if rho == 0.0:
        return std_norm_cdf(-xl) * std_norm_cdf(-yl)

    tp = 2.0 * np.pi
    h = xl
    k = yl
    hk = h * k
    bvn = 0.0

    if np.abs(rho) < 0.3:
        # Gauss Legendre points and weights, n = 6
        w = np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904])
        x = np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970])
    elif np.abs(rho) < 0.75:
        # Gauss Legendre points and weights, n = 12
        w = np.array(
            [
                0.04717533638651177,
                0.1069393259953183,
                0.1600783285433464,
                0.2031674267230659,
                0.2334925365383547,
                0.2491470458134029,
            ]
        )
        x = np.array(
            [
                0.9815606342467191,
                0.9041172563704750,
                0.7699026741943050,
                0.5873179542866171,
                0.3678314989981802,
                0.1252334085114692,
            ]
        )
    else:
        # Gauss Legendre points and weights, n = 20
        w = np.array(
            [
                0.01761400713915212,
                0.04060142980038694,
                0.06267204833410906,
                0.08327674157670475,
                0.1019301198172404,
                0.1181945319615184,
                0.1316886384491766,
                0.1420961093183821,
                0.1491729864726037,
                0.1527533871307259,
            ]
        )
        x = np.array(
            [
                0.9931285991850949,
                0.9639719272779138,
                0.9122344282513259,
                0.8391169718222188,
                0.7463319064601508,
                0.6360536807265150,
                0.5108670019508271,
                0.3737060887154196,
                0.2277858511416451,
                0.07652652113349733,
            ]
        )

    w = np.tile(w, 2)
    x = np.concatenate([1.0 - x, 1.0 + x])

    if np.abs(rho) < 0.925:
        hs = 0.5 * (h * h + k * k)
        asr = 0.5 * np.arcsin(rho)
        sn = np.sin(asr * x)
        bvn = np.dot(w, np.exp((sn * hk - hs) / (1.0 - sn**2)))
        bvn = bvn * asr / tp + std_norm_cdf(-h) * std_norm_cdf(-k)
    else:
        if rho < 0.0:
            k = -k
            hk = -hk
        if np.abs(rho) < 1.0:
            ass = 1.0 - rho**2
            a = np.sqrt(ass)
            bs = (h - k) ** 2
            asr = -0.5 * (bs / ass + hk)
            c = (4.0 - hk) / 8.0
            d = (12.0 - hk) / 80.0
            if asr > -100.0:
                bvn = (
                    a
                    * np.exp(asr)
                    * (1.0 - c * (bs - ass) * (1.0 - d * bs) / 3.0 + c * d * ass**2)
                )
            if hk > -100.0:
                b = np.sqrt(bs)
                sp = np.sqrt(tp) * std_norm_cdf(-b / a)
                bvn -= (
                    np.exp(-0.5 * hk) * sp * b * (1.0 - c * bs * (1.0 - d * bs) / 3.0)
                )
            a *= 0.5
            xs = (a * x) ** 2
            asr = -0.5 * (bs / xs + hk)
            idxs = (asr > -100.0).nonzero()
            xs = xs[idxs]
            sp = 1.0 + c * xs * (1.0 + 5.0 * d * xs)
            rs = np.sqrt(1.0 - xs)
            ep = np.exp(-0.5 * hk * xs / (1.0 + rs) ** 2) / rs
            bvn = (a * np.dot(np.exp(asr[idxs]) * (sp - ep), w[idxs]) - bvn) / tp
        if rho > 0:
            bvn += std_norm_cdf(-max(h, k))
        elif h >= k:
            bvn = -bvn
        else:
            if h < 0.0:
                L = std_norm_cdf(k) - std_norm_cdf(h)
            else:
                L = std_norm_cdf(-h) - std_norm_cdf(-k)
            bvn = L - bvn
    return max(0.0, min(1.0, bvn))


def std_norm_cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def std_norm_pdf(z):
    return np.exp(-0.5 * (z**2)) / np.sqrt(2.0 * np.pi)
