r"""
Two-half-cell profile–Rosenblatt SGS condensate cubature, mirroring Julia
``subgrid_layer_profile_quadrature.jl`` / ``integrate_over_sgs_linear_profile``.

**Cell rule:** profile–Rosen cubature on the **DN** mean-gradient path and separately on the **UP**
path, combined with **½–½** weights when both succeed (otherwise the surviving half gets full
weight). Matches the explicit two-transport branch in production Julia.

- **Frozen σ** in each half: the same ½( U⊛N_− + U⊛N_+ ) ``erf``-based
  u-marginal and Halley p-quantile as Julia.
- **Linear-in-z** segment law: the **same** ``_half_cell_segment_linvar_*_psi``
  reduction as production. The only ``ν ∈ {3/2, 1/2, −1/2}`` generalized tails
  are implemented with **O(1)** `erfc` / `erfcx` and the algebraic
  Ψ\_{−1/2} ↔ Ψ\_{1/2} and Ψ\_{3/2} recurrences (no k-loop, no mpmath).
- **Two-half cell:** :func:`profile_rosenblatt_cubature_two_halves_cell` implements that DN/UP split
  (``mean_gradient`` in :func:`two_slope_rosenblatt_params`).

**Environment:** this module needs SciPy (same stack as
``Variance_Stuff.ipynb`` in the project conda env, e.g. WeatherQuest), not
the system ``python3`` with no ``scipy``.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import (
    erf,
    erfc,
    erfcx,
    erfinv,
    gamma,
    gammaincc,
    roots_hermite,
)

_NUM_EPS = 1.0e-9


def _std_normal_cdf(y: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + erf(np.asarray(y) / np.sqrt(2.0)))


def _std_normal_pdf(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.exp(-0.5 * y * y) / np.sqrt(2.0 * np.pi)


def _normal_mills_integral(y: np.ndarray | float) -> np.ndarray | float:
    return y * _std_normal_cdf(y) + _std_normal_pdf(np.asarray(y, dtype=float))


def uniform_gaussian_convolution_cdf(x: float, L: float, s: float) -> float:
    s = max(float(s), _NUM_EPS)
    L = max(float(L), _NUM_EPS)
    u1 = (x + L * 0.5) / s
    u2 = (x - L * 0.5) / s
    return (s / L) * (float(_normal_mills_integral(u1)) - float(_normal_mills_integral(u2)))


def uniform_gaussian_convolution_pdf(x: float, L: float, s: float) -> float:
    s2 = s * np.sqrt(2.0)
    s = max(float(s), _NUM_EPS)
    L = max(float(L), _NUM_EPS)
    u1 = (x + L * 0.5) / s2
    u2 = (x - L * 0.5) / s2
    return (1.0 / (2.0 * L)) * (erf(u1) - erf(u2))


def uniform_gaussian_convolution_pdf_prime(x: float, L: float, s: float) -> float:
    s = max(float(s), _NUM_EPS)
    L = max(float(L), _NUM_EPS)
    u1 = (x + L * 0.5) / s
    u2 = (x - L * 0.5) / s
    c = 1.0 / (L * s * np.sqrt(2.0 * np.pi))
    return c * (np.exp(-0.5 * u1 * u1) - np.exp(-0.5 * u2 * u2))


def _ug_cdf_shifted(x: float, a: float, b: float, s: float) -> float:
    L = b - a
    mid = 0.5 * (a + b)
    return uniform_gaussian_convolution_cdf(x - mid, L, s)


def _ug_pdf_shifted(x: float, a: float, b: float, s: float) -> float:
    L = b - a
    mid = 0.5 * (a + b)
    return uniform_gaussian_convolution_pdf(x - mid, L, s)


def _ug_pdfp_shifted(x: float, a: float, b: float, s: float) -> float:
    L = b - a
    mid = 0.5 * (a + b)
    return uniform_gaussian_convolution_pdf_prime(x - mid, L, s)


def mixture_uniform_convolution_cdf(x: float, L_dn: float, s_dn: float, L_up: float, s_up: float) -> float:
    F_dn = _ug_cdf_shifted(x, -L_dn, 0.0, s_dn)
    F_up = _ug_cdf_shifted(x, 0.0, L_up, s_up)
    return 0.5 * (F_dn + F_up)


def mixture_uniform_convolution_pdf(x: float, L_dn: float, s_dn: float, L_up: float, s_up: float) -> float:
    f_dn = _ug_pdf_shifted(x, -L_dn, 0.0, s_dn)
    f_up = _ug_pdf_shifted(x, 0.0, L_up, s_up)
    return 0.5 * (f_dn + f_up)


def mixture_uniform_convolution_pdf_prime(
    x: float, L_dn: float, s_dn: float, L_up: float, s_up: float
) -> float:
    fp_dn = _ug_pdfp_shifted(x, -L_dn, 0.0, s_dn)
    fp_up = _ug_pdfp_shifted(x, 0.0, L_up, s_up)
    return 0.5 * (fp_dn + fp_up)


def mixture_uniform_mean_var(
    L_dn: float, s_dn: float, L_up: float, s_up: float
) -> tuple[float, float]:
    """Closed-form (mean, var) of the ½+½ u-mixture, same as Julia `mixture_uniform_gaussian_convolution_mean_var`."""
    mu = (L_up - L_dn) * 0.25
    var = (5.0 * L_up**2 + 6.0 * L_up * L_dn + 5.0 * L_dn**2) / 48.0 + 0.5 * (s_dn**2 + s_up**2)
    return float(mu), float(var)


def mixture_convolution_quantile_halley(
    p: float, L_dn: float, s_dn: float, L_up: float, s_up: float
) -> float:
    """Unshifted u s.t. F_mixture(u) = p (one Halley step from moment-matched Gaussian, Julia production default)."""
    t = 2.0 * p - 1.0
    t = np.clip(t, -1.0 + 100.0 * _NUM_EPS, 1.0 - 100.0 * _NUM_EPS)
    z = np.sqrt(2.0) * erfinv(t)
    mu, varm = mixture_uniform_mean_var(L_dn, s_dn, L_up, s_up)
    sig = np.sqrt(max(varm, _NUM_EPS**2))
    u = mu + sig * z
    Fv = mixture_uniform_convolution_cdf(u, L_dn, s_dn, L_up, s_up)
    g = Fv - p
    fv = mixture_uniform_convolution_pdf(u, L_dn, s_dn, L_up, s_up)
    if abs(fv) < _NUM_EPS:
        return u
    fpv = mixture_uniform_convolution_pdf_prime(u, L_dn, s_dn, L_up, s_up)
    denom = 2.0 * fv**2 - g * fpv
    if denom == 0.0:
        denom = 1.0e-15
    denom = float(np.copysign(max(abs(denom), 1.0e-15), denom))
    return float(u - 2.0 * fv * g / denom)


# ---------------------------------------------------------------------------
# Generalized tail Γ(ν, t, b) = ∫_t^∞ τ^{ν-1} e^{-τ - b/τ} dτ  for
# ν ∈ {3/2, 1/2, −1/2} using only erfc/erfcx (ν=1/2) and the two recurrences
# (same as the “ExactAnalyticTailEngine” + method-notes Ψ_ν family).
# No Taylor loop in k, no mpmath.
# ---------------------------------------------------------------------------


def _upper_incomplete_gamma_ordinary(nu: float, t: float) -> float:
    """Ordinary upper incomplete Γ(ν, t) = ∫_t^∞ τ^{ν-1} e^{-τ} dτ (b = 0)."""
    t = max(float(t), _NUM_EPS)
    return float(gammaincc(nu, t) * gamma(nu))


def gamma_tail_nu_half(t: float, b: float) -> float:
    """
    ∫_t^∞ τ^{-1/2} e^{-τ - b/τ} dτ  (Julia ``_gamma_gen_upper(1/2, t, b)`` for b > 0 path).

    Closed form (ν = 1/2 “base” case) with `erfc` / `erfcx`; at t=0, ``√π e^{-2√b}``.
    """
    b = max(float(b), 0.0)
    t = float(t)
    if b <= _NUM_EPS**2:
        return _upper_incomplete_gamma_ordinary(0.5, t)
    if t <= _NUM_EPS:
        return float(np.sqrt(np.pi) * np.exp(-2.0 * np.sqrt(b)))
    t_safe = max(t, 1.0e-16)
    sqrt_t = np.sqrt(t_safe)
    sqrt_b = np.sqrt(b)
    sqrt_b_over_t = np.sqrt(b / t_safe)
    term1 = np.exp(-2.0 * sqrt_b) * erfc(sqrt_t - sqrt_b_over_t)
    term2 = np.exp(-t_safe - b / t_safe) * erfcx(sqrt_t + sqrt_b_over_t)
    return float(0.5 * np.sqrt(np.pi) * (term1 + term2))


def gamma_tail_nu_mhalf(t: float, b: float) -> float:
    r"""
    ν = -1/2:  Ψ\_{-1/2}(t,b) = \frac{1}{\sqrt{b}} \left[Ψ\_{1/2}(0,b) - Ψ\_{1/2}(b/t,b)\right].
    """
    b = max(float(b), 0.0)
    t = max(float(t), 0.0)
    if b <= _NUM_EPS**2:
        return _upper_incomplete_gamma_ordinary(-0.5, t)
    h0 = gamma_tail_nu_half(0.0, b)
    h1 = gamma_tail_nu_half(b / max(t, 1.0e-16), b)
    return float((h0 - h1) / np.sqrt(b))


def gamma_tail_nu_3halves(t: float, b: float) -> float:
    r"""
    ν = 3/2:  Ψ\_{3/2} = ½Ψ\_{1/2} + √t\,e^{-(t+b/t)} + b\,Ψ\_{-1/2}  (integration by parts).
    """
    b = max(float(b), 0.0)
    t = max(float(t), 0.0)
    if b <= _NUM_EPS**2:
        return _upper_incomplete_gamma_ordinary(1.5, t)
    t_safe = max(t, 1.0e-16)
    em = np.exp(-(t_safe + b / t_safe))
    return float(
        0.5 * gamma_tail_nu_half(t, b) + np.sqrt(t_safe) * em + b * gamma_tail_nu_mhalf(t, b)
    )


def _gamma_gen_upper(nu: float, t: float, b: float) -> float:
    """
    ``Γ(ν, t, b) = ∫_t^∞ τ^{ν-1} e^{-τ - b/τ} dτ`` for ν ∈ {3/2, 1/2, −1/2}.
    b ≈ 0: ordinary ``SpecialFunctions.gamma(ν, t)``; else closed forms.
    """
    b = max(float(b), 0.0)
    t = max(float(t), 0.0)
    if b <= _NUM_EPS**2:
        return _upper_incomplete_gamma_ordinary(nu, t)
    if abs(nu - 0.5) < 1.0e-12:
        return gamma_tail_nu_half(t, b)
    if abs(nu + 0.5) < 1.0e-12:
        return gamma_tail_nu_mhalf(t, b)
    if abs(nu - 1.5) < 1.0e-12:
        return gamma_tail_nu_3halves(t, b)
    raise ValueError(f"nu={nu} not in {{3/2, 1/2, -1/2}}")


def _gamma_gen_segment(nu: float, tau_lo: float, tau_hi: float, b: float) -> float:
    a = min(tau_lo, tau_hi)
    c = max(tau_lo, tau_hi)
    return _gamma_gen_upper(nu, a, b) - _gamma_gen_upper(nu, c, b)


def _psi_nu_tau_integrals(
    tau_lo: float, tau_hi: float, b: float
) -> tuple[float, float, float]:
    I32 = _gamma_gen_segment(1.5, tau_lo, tau_hi, b)
    I12 = _gamma_gen_segment(0.5, tau_lo, tau_hi, b)
    Im12 = _gamma_gen_segment(-0.5, tau_lo, tau_hi, b)
    return I32, I12, Im12


def _psi_nu_from_integrals(nu: float, b2: float, i_nu: float) -> float:
    eps2 = _NUM_EPS**2
    b2c = max(float(b2), eps2)
    return float(np.exp(nu * (np.log(2.0) - np.log(b2c))) * i_nu)


# --- half-cell linear-variance (Ψ_ν) — same branches as ``_half_cell_segment_linvar_*_psi`` in Julia


def _regularize_mu_slope(mu_slope: float, s_slope: float) -> float:
    eps = _NUM_EPS
    e_mu = max(eps, np.sqrt(eps) * max(1.0, abs(s_slope)))
    if abs(mu_slope) >= e_mu:
        return float(mu_slope)
    if abs(mu_slope) <= eps:
        return float(np.copysign(e_mu, s_slope))
    return float(np.copysign(e_mu, mu_slope))


def _G_mills(t: float) -> float:
    return float(t * _std_normal_cdf(t) + _std_normal_pdf(np.array([t]))[0])


def _half_cell_segment_linvar_const_sigma_cdf(
    u: float, L: float, mu0: float, mus: float, sig: float
) -> float:
    """
    ``F(u) = (1/L) ∫_0^L Φ((u - μ_0 - μ_s z)/σ) dz`` with constant ``σ``.

    Mills form: ``(σ/L)(G(t_a)-G(t_b))`` equals ``μ_s × F(u)`` for
    ``t_a=(u-μ_0)/σ``, ``t_b=(u-μ_0-μ_s L)/σ`` and ``G(t)=tΦ(t)+φ(t)``; divide
    by ``μ_s`` so ``F`` matches the defining integral above (checked in
    ``test_profile_rosenblatt_primitives.py`` vs quadrature). Neither language is
    the authority—only this integral definition is.
    """
    eps = _NUM_EPS
    L = max(L, eps)
    sigp = max(sig, eps)
    mu_l = mu0 + mus * L
    if abs(mus) <= eps * max(1.0, abs(mu_l)):
        return float(_std_normal_cdf((u - mu0) / sigp))
    ta = (u - mu0) / sigp
    tb = (u - mu_l) / sigp
    mills = float((sigp / L) * (_G_mills(ta) - _G_mills(tb)))
    return mills / float(mus)


def _half_cell_segment_linvar_const_sigma_pdf(
    u: float, L: float, mu0: float, mus: float, sig: float
) -> float:
    eps = _NUM_EPS
    L = max(L, eps)
    sigp = max(sig, eps)
    mu_l = mu0 + mus * L
    if abs(mus) <= eps * max(1.0, abs(mu_l)):
        y = (u - mu0) / sigp
        return float(_std_normal_pdf(np.array([y]))[0] / sigp)
    ta = (u - mu0) / sigp
    tb = (u - mu_l) / sigp
    return float((1.0 / (L * mus)) * (_std_normal_cdf(ta) - _std_normal_cdf(tb)))


def _half_cell_segment_linvar_const_sigma_pdf_prime(
    u: float, L: float, mu0: float, mus: float, sig: float
) -> float:
    eps = _NUM_EPS
    L = max(L, eps)
    sigp = max(sig, eps)
    mu_l = mu0 + mus * L
    if abs(mus) <= eps * max(1.0, abs(mu_l)):
        y = (u - mu0) / sigp
        return float((-y / (sigp**2)) * _std_normal_pdf(np.array([y]))[0])
    ta = (u - mu0) / sigp
    tb = (u - mu_l) / sigp
    return float(
        (1.0 / (L * sigp * mus))
        * (_std_normal_pdf(np.array([ta]))[0] - _std_normal_pdf(np.array([tb]))[0])
    )


def _half_cell_segment_linvar_cdf_psi(
    u: float, L: float, mu0: float, mus: float, s0sq: float, s_sl: float
) -> float:
    eps = _NUM_EPS
    esig = max(eps**2, 1.0e-10)
    L = max(L, eps)
    if abs(s_sl) * L <= esig * max(abs(s0sq), 1.0):
        sig = np.sqrt(max(s0sq, eps**2))
        return _half_cell_segment_linvar_const_sigma_cdf(u, L, mu0, mus, float(sig))
    mus_u = _regularize_mu_slope(mus, s_sl)
    b = mus_u / s_sl
    b2 = b**2
    if not (b2 > esig):
        sig = np.sqrt(max(s0sq, eps**2))
        return _half_cell_segment_linvar_const_sigma_cdf(u, L, mu0, mus, float(sig))
    s0c = max(s0sq, eps**2)
    v_dn = s0c
    v_up = max(s0sq + s_sl * L, eps**2)
    t_dn = (b2 * v_dn) * 0.5
    t_up = (b2 * v_up) * 0.5
    t_lo = min(t_dn, t_up)
    t_hi = max(t_dn, t_up)
    if not (t_hi > t_lo + eps * max(1.0, t_lo)):
        mu_l = mu0 + mus * L
        s_l = np.sqrt(v_up)
        return float(_std_normal_cdf((u - mu_l) / max(s_l, eps)))
    a = u - mu0 + mus_u * s0sq / s_sl
    bpar = (a**2 * b2) * 0.25
    i32, i12, im12 = _psi_nu_tau_integrals(t_lo, t_hi, bpar)
    y32 = _psi_nu_from_integrals(1.5, b2, i32)
    y12 = _psi_nu_from_integrals(0.5, b2, i12)
    ym12 = _psi_nu_from_integrals(-0.5, b2, im12)
    mu_l = mu0 + mus * L
    s_l = np.sqrt(max(s0sq + s_sl * L, eps**2))
    phi_l = float(_std_normal_cdf((u - mu_l) / s_l))
    # ``|s_sl|``: the closed form is invariant under sign of ``s_sl``; ``1/(L s_sl)``
    # incorrectly flips the Ψ correction when ``s_sl < 0`` (see defining ``z`` integral).
    c = np.exp(a * b) / (L * abs(s_sl) * np.sqrt(2.0 * np.pi))
    return float(phi_l + 0.5 * c * (b * y32 + (a - b * s0sq) * y12 - a * s0sq * ym12))


def _half_cell_segment_linvar_pdf_psi(
    u: float, L: float, mu0: float, mus: float, s0sq: float, s_sl: float
) -> float:
    eps = _NUM_EPS
    esig = max(eps**2, 1.0e-10)
    L = max(L, eps)
    if abs(s_sl) * L <= esig * max(abs(s0sq), 1.0):
        sig = np.sqrt(max(s0sq, eps**2))
        return _half_cell_segment_linvar_const_sigma_pdf(u, L, mu0, mus, float(sig))
    mus_u = _regularize_mu_slope(mus, s_sl)
    b = mus_u / s_sl
    b2 = b**2
    if not (b2 > esig):
        sig = np.sqrt(max(s0sq, eps**2))
        return _half_cell_segment_linvar_const_sigma_pdf(u, L, mu0, mus, float(sig))
    s0c = max(s0sq, eps**2)
    v_dn = s0c
    v_up = max(s0sq + s_sl * L, eps**2)
    t_dn = (b2 * v_dn) * 0.5
    t_up = (b2 * v_up) * 0.5
    t_lo = min(t_dn, t_up)
    t_hi = max(t_dn, t_up)
    if not (t_hi > t_lo + eps * max(1.0, t_lo)):
        mu_l = mu0 + mus * L
        s_l = np.sqrt(v_up)
        y = (u - mu_l) / max(s_l, eps)
        return float(_std_normal_pdf(np.array([y]))[0] / max(s_l, eps))
    a = u - mu0 + mus_u * s0sq / s_sl
    bpar = (a**2 * b2) * 0.25
    _i32, i12, _im12 = _psi_nu_tau_integrals(t_lo, t_hi, bpar)
    y12 = _psi_nu_from_integrals(0.5, b2, i12)
    c = np.exp(a * b) / (L * abs(s_sl) * np.sqrt(2.0 * np.pi))
    return float(c * y12)


def _half_cell_segment_linvar_pdf_prime_psi(
    u: float, L: float, mu0: float, mus: float, s0sq: float, s_sl: float
) -> float:
    eps = _NUM_EPS
    esig = max(eps**2, 1.0e-10)
    L = max(L, eps)
    if abs(s_sl) * L <= esig * max(abs(s0sq), 1.0):
        sig = np.sqrt(max(s0sq, eps**2))
        return _half_cell_segment_linvar_const_sigma_pdf_prime(u, L, mu0, mus, float(sig))
    mus_u = _regularize_mu_slope(mus, s_sl)
    b = mus_u / s_sl
    b2 = b**2
    if not (b2 > esig):
        sig = np.sqrt(max(s0sq, eps**2))
        return _half_cell_segment_linvar_const_sigma_pdf_prime(u, L, mu0, mus, float(sig))
    s0c = max(s0sq, eps**2)
    v_dn = s0c
    v_up = max(s0sq + s_sl * L, eps**2)
    t_dn = (b2 * v_dn) * 0.5
    t_up = (b2 * v_up) * 0.5
    t_lo = min(t_dn, t_up)
    t_hi = max(t_dn, t_up)
    if not (t_hi > t_lo + eps * max(1.0, t_lo)):
        mu_l = mu0 + mus * L
        s_l = np.sqrt(v_up)
        y = (u - mu_l) / max(s_l, eps)
        return float((-y / (max(s_l, eps) ** 2)) * _std_normal_pdf(np.array([y]))[0])
    a = u - mu0 + mus_u * s0sq / s_sl
    bpar = (a**2 * b2) * 0.25
    _i32, i12, im12 = _psi_nu_tau_integrals(t_lo, t_hi, bpar)
    y12 = _psi_nu_from_integrals(0.5, b2, i12)
    ym12 = _psi_nu_from_integrals(-0.5, b2, im12)
    c = np.exp(a * b) / (L * abs(s_sl) * np.sqrt(2.0 * np.pi))
    return float(c * (-a * ym12 + b * y12))


def _half_cell_segment_linvar_mean_var(
    L: float, mu0: float, mus: float, s0sq: float, s_sl: float
) -> tuple[float, float]:
    if L <= 0.0:
        return float(mu0), max(s0sq, 0.0)
    mean = mu0 + mus * L * 0.5
    var = max(s0sq + s_sl * L * 0.5, 0.0) + (mus**2) * (L**2) / 12.0
    return float(mean), float(var)


def mixture_linvar_cdf(
    u: float,
    mu0: float,
    L_dn: float, mus_dn: float, s0sq_dn: float, s_slope_dn: float,
    L_up: float, mus_up: float, s0sq_up: float, s_slope_up: float,
) -> float:
    f_dn = _half_cell_segment_linvar_cdf_psi(u, L_dn, mu0, mus_dn, s0sq_dn, s_slope_dn)
    f_up = _half_cell_segment_linvar_cdf_psi(u, L_up, mu0, mus_up, s0sq_up, s_slope_up)
    return 0.5 * (f_dn + f_up)


def mixture_linvar_pdf(
    u: float,
    mu0: float,
    L_dn: float, mus_dn: float, s0sq_dn: float, s_slope_dn: float,
    L_up: float, mus_up: float, s0sq_up: float, s_slope_up: float,
) -> float:
    f_dn = _half_cell_segment_linvar_pdf_psi(u, L_dn, mu0, mus_dn, s0sq_dn, s_slope_dn)
    f_up = _half_cell_segment_linvar_pdf_psi(u, L_up, mu0, mus_up, s0sq_up, s_slope_up)
    return 0.5 * (f_dn + f_up)


def mixture_linvar_pdf_prime(
    u: float,
    mu0: float,
    L_dn: float, mus_dn: float, s0sq_dn: float, s_slope_dn: float,
    L_up: float, mus_up: float, s0sq_up: float, s_slope_up: float,
) -> float:
    fp_dn = _half_cell_segment_linvar_pdf_prime_psi(
        u, L_dn, mu0, mus_dn, s0sq_dn, s_slope_dn
    )
    fp_up = _half_cell_segment_linvar_pdf_prime_psi(
        u, L_up, mu0, mus_up, s0sq_up, s_slope_up
    )
    return 0.5 * (fp_dn + fp_up)


def mixture_linvar_mean_var(
    mu0: float,
    L_dn: float, mus_dn: float, s0sq_dn: float, s_slope_dn: float,
    L_up: float, mus_up: float, s0sq_up: float, s_slope_up: float,
) -> tuple[float, float]:
    m_dn, v_dn = _half_cell_segment_linvar_mean_var(L_dn, mu0, mus_dn, s0sq_dn, s_slope_dn)
    m_up, v_up = _half_cell_segment_linvar_mean_var(L_up, mu0, mus_up, s0sq_up, s_slope_up)
    mu = 0.5 * (m_dn + m_up)
    within = 0.5 * (v_dn + v_up)
    between = 0.5 * ((m_dn - mu) ** 2 + (m_up - mu) ** 2)
    return float(mu), float(within + between)


def mixture_linvar_quantile_halley(
    p: float,
    mu0: float,
    L_dn: float, mus_dn: float, s0sq: float, s_slope_dn: float,
    L_up: float, mus_up: float, s_slope_up: float,
) -> float:
    """
    One Halley step on the linear-variance mixture CDF (same as Julia production).
    Both halves use the same center-face ``s0² = σ²_{u|v}(0)`` (``s0sq``) as in
    ``integrate_over_sgs_linear_profile``.
    """
    eps = _NUM_EPS
    t2 = 2.0 * p - 1.0
    t2 = float(np.clip(t2, -1.0 + 100.0 * eps, 1.0 - 100.0 * eps))
    z = np.sqrt(2.0) * erfinv(t2)
    m_mix, v_mix = mixture_linvar_mean_var(
        mu0, L_dn, mus_dn, s0sq, s_slope_dn, L_up, mus_up, s0sq, s_slope_up
    )
    sig = np.sqrt(max(v_mix, eps**2))
    u0 = m_mix + sig * z
    g = mixture_linvar_cdf(
        u0, mu0, L_dn, mus_dn, s0sq, s_slope_dn, L_up, mus_up, s0sq, s_slope_up
    ) - p
    fv = mixture_linvar_pdf(
        u0, mu0, L_dn, mus_dn, s0sq, s_slope_dn, L_up, mus_up, s0sq, s_slope_up
    )
    if abs(fv) < eps:
        return float(u0)
    fpv = mixture_linvar_pdf_prime(
        u0, mu0, L_dn, mus_dn, s0sq, s_slope_dn, L_up, mus_up, s0sq, s_slope_up
    )
    denom = 2.0 * fv**2 - g * fpv
    denom = float(np.copysign(max(abs(denom), 1.0e-15), denom))
    return float(u0 - 2.0 * fv * g / denom)


# --- Two-slope Rosenblatt parameters (translation of _two_slope_rosenblatt_params) ---


def two_slope_rosenblatt_params(
    *,
    mu_t: float,
    mu_q: float,
    sig_t2_c: float,
    sig_q2_c: float,
    rho_tq: float,
    m_t_dn: float,
    m_t_up: float,
    m_q_dn: float,
    m_q_up: float,
    s_t_dn: float,
    s_t_up: float,
    s_q_dn: float,
    s_q_up: float,
    H: float,
    mean_gradient: str = "avg",
) -> dict | None:
    """
    ``mean_gradient`` selects the **transport direction** for Rosenblatt outer axis and ``M_inv``:

    - ``\"avg\"`` — legacy centered gradient ``(m_{T,dn}+m_{T,up})/2`` (matches Julia
      ``_two_slope_rosenblatt_params`` before the two-half cell fix).
    - ``\"dn\"`` / ``\"up\"`` — use the DN-only or UP-only mean-gradient vector. Averaging
      two cubatures with ``\"dn\"`` and ``\"up\"`` targets the same **two-half** story as
      the mixture PDF (separate mean paths toward each face) far better than a single
      averaged axis when DN/UP slopes differ.
    """
    eps = _NUM_EPS
    mg = str(mean_gradient).lower()
    if mg == "avg":
        d_t = 0.5 * (m_t_dn + m_t_up)
        d_q = 0.5 * (m_q_dn + m_q_up)
    elif mg == "dn":
        d_t, d_q = float(m_t_dn), float(m_q_dn)
    elif mg == "up":
        d_t, d_q = float(m_t_up), float(m_q_up)
    else:
        raise ValueError(f"mean_gradient must be 'avg', 'dn', or 'up', got {mean_gradient!r}")
    a_sq = d_t * d_t + d_q * d_q
    if a_sq <= eps**2:
        return None
    inv_a = 1.0 / a_sq
    u_t = d_t * inv_a
    u_q = d_q * inv_a
    a_dn_raw = u_t * m_t_dn + u_q * m_q_dn
    a_up_raw = u_t * m_t_up + u_q * m_q_up
    a_dn = max(a_dn_raw, 0.0)
    a_up = max(a_up_raw, 0.0)
    L_dn = a_dn * (0.5 * H)
    L_up = a_up * (0.5 * H)
    half = 0.5 * H
    s_t2_fdn = max(sig_t2_c - half * s_t_dn, 0.0)
    s_t2_fup = max(sig_t2_c + half * s_t_up, 0.0)
    s_q2_fdn = max(sig_q2_c - half * s_q_dn, 0.0)
    s_q2_fup = max(sig_q2_c + half * s_q_up, 0.0)
    rh = float(np.clip(rho_tq, -1.0, 1.0))
    s_tq_c = rh * np.sqrt(max(sig_t2_c, 0.0) * max(sig_q2_c, 0.0))
    s_tq_fdn = rh * np.sqrt(s_t2_fdn * s_q2_fdn)
    s_tq_fup = rh * np.sqrt(s_t2_fup * s_q2_fup)
    s1c, s12c, s2c = _rotated_sigma(sig_t2_c, sig_q2_c, s_tq_c, u_t, u_q, d_t, d_q)
    s1f_dn, s12f_dn, s2f_dn = _rotated_sigma(
        s_t2_fdn, s_q2_fdn, s_tq_fdn, u_t, u_q, d_t, d_q
    )
    s1f_up, s12f_up, s2f_up = _rotated_sigma(
        s_t2_fup, s_q2_fup, s_tq_fup, u_t, u_q, d_t, d_q
    )
    s2c_eff = max(s2c, eps)
    s2fdn = max(s2f_dn, eps)
    s2fup = max(s2f_up, eps)
    s_u_c = np.sqrt(max(s1c - s12c**2 / s2c_eff, 0.0))
    s_u_dn = np.sqrt(max(s1f_dn - s12f_dn**2 / s2fdn, 0.0))
    s_u_up = np.sqrt(max(s1f_up - s12f_up**2 / s2fup, 0.0))
    s_v = float(np.sqrt(max(s2c, 0.0)))
    r_c = s12c / s2c_eff
    r_fdn = s12f_dn / s2fdn
    r_fup = s12f_up / s2fup
    m_inv = np.array(
        [[d_t, -d_q * inv_a], [d_q, d_t * inv_a]], dtype=float
    )
    s0sq = s_u_c**2
    if L_dn > eps:
        invldn = 1.0 / L_dn
    else:
        invldn = 0.0
    if L_up > eps:
        invlup = 1.0 / L_up
    else:
        invlup = 0.0
    s_s_dn = (s_u_dn**2 - s0sq) * invldn
    s_s_up = (s_u_up**2 - s0sq) * invlup
    tol_s = 100.0 * eps * max(1.0, s0sq)
    tol_r = 100.0 * eps * max(1.0, abs(r_c))
    use_linvar = bool(
        abs(s_s_dn) > tol_s
        or abs(s_s_up) > tol_s
        or abs(r_fdn - r_c) * invldn > tol_r
        or abs(r_fup - r_c) * invlup > tol_r
    )
    s_max_leg = max(s_u_c, s_u_dn, s_u_up, eps)

    return {
        "M_inv": m_inv,
        "s_v": s_v,
        "L_dn": L_dn,
        "L_up": L_up,
        "s_u_c": s_u_c,
        "s_u_dn": s_u_dn,
        "s_u_up": s_u_up,
        "r_c": r_c,
        "r_fdn": r_fdn,
        "r_fup": r_fup,
        "s0sq": s0sq,
        "s_slope_dn": s_s_dn,
        "s_slope_up": s_s_up,
        "use_linvar": use_linvar,
        "d_T": d_t,
        "d_q": d_q,
        "inv_α2": inv_a,
        "s_max": s_max_leg,
    }


def _rotated_sigma(
    sT2: float, sq2: float, sTq: float, uT: float, uq: float, dT: float, dQ: float
) -> tuple[float, float, float]:
    s1sq = uT**2 * sT2 + 2.0 * uT * uq * sTq + uq**2 * sq2
    s12 = (
        -dQ * uT * sT2 + (dT * uT - dQ * uq) * sTq + dT * uq * sq2
    )
    s2sq = dQ**2 * sT2 - 2.0 * dQ * dT * sTq + dT**2 * sq2
    return float(s1sq), float(s12), float(s2sq)


# --- map Hermite to physical (T,q) like Julia GaussianSGS get_physical_point ---


def hermite_from_gaussian_fluctuations(
    δq: np.ndarray, δT: np.ndarray, sig_q: float, sig_t: float, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of the usual χ → δ map: recover (χ1,χ2) from (δT,δq) fluctuations (white layer space)."""
    sig_q = max(sig_q, _NUM_EPS)
    sig_t = max(sig_t, _NUM_EPS)
    root2 = np.sqrt(2.0)
    χ1 = δq / (root2 * sig_q)
    sc = np.sqrt(max(1.0 - rho**2, 0.0)) * sig_t
    sc = max(sc, _NUM_EPS)
    χ2 = (δT - root2 * rho * sig_t * χ1) / (root2 * sc)
    return χ1, χ2


def physical_point_gaussian(
    χ1: np.ndarray,
    χ2: np.ndarray,
    mu_q: float,
    mu_t: float,
    sig_q: float,
    sig_t: float,
    rho: float,
    t_min: float,
    q_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    root2 = np.sqrt(2.0)
    ε = _NUM_EPS
    q_hat = np.clip(mu_q + root2 * sig_q * χ1, 0.0, q_max)
    χ1_eff = (q_hat - mu_q) / (root2 * max(sig_q, ε))
    sc = float(np.sqrt(max(1.0 - rho**2, 0.0)) * sig_t)
    sc = max(sc, ε)
    mu_c = mu_t + root2 * rho * sig_t * χ1_eff
    t_hat = np.maximum(t_min, mu_c + root2 * sc * χ2)
    return t_hat, q_hat


def fluc_to_tq(
    dT: np.ndarray,
    dq: np.ndarray,
    mu_t: float,
    mu_q: float,
    sig_t: float,
    sig_q: float,
    rho: float,
    t_min: float,
    q_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Julia ``_physical_Tq_from_fluctuations`` / ``GaussianSGS`` ``get_physical_point``.

    For **unclamped** fluctuations ``(δT, δq)`` drawn from the layer Gaussian
    ``N(0, Σ_{turb})``, this round-trip is **algebraically** ``(μ_T+δT, μ_q+δq)`` — the
    Hermite step is the inverse/forward pair for the **same** bivariate Normal. Deviations
    appear **only** when ``physical_point_gaussian`` clips ``q`` or floors ``T`` (physical
    bounds), not from choosing a different stochastic model than an affine ``μ+δ`` map.
    """
    χ1, χ2 = hermite_from_gaussian_fluctuations(dq, dT, sig_q, sig_t, rho)
    return physical_point_gaussian(χ1, χ2, mu_q, mu_t, sig_q, sig_t, rho, t_min, q_max)


# --- public API: condensate and node bundle ---


def _gl01(n: int) -> tuple[np.ndarray, np.ndarray]:
    t, w = leggauss(n)
    return (t + 1.0) * 0.5, w * 0.5


def profile_rosenblatt_cubature(
    n: int,
    par: dict,
    cov0: np.ndarray,
    H_layer: float,
    mu_t: float,
    mu_q: float,
    q_sat_fn: Callable[[np.ndarray], np.ndarray],
    *,
    mean_gradient: str = "avg",
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    str,
]:
    """
    Returns
    -------
    q_c, t_nodes, q_nodes, w_flat, u_in, v_in, use_linvar, status
        ``(u_in, v_in)`` are inner Rosenblatt coordinates before `M_inv` (same as
        the Julia state prior to `get_physical_point`). status is ``"ok"`` or
        ``"degenerate_gradient_fallback_center_gh"``.
    """
    s_t2 = max(float(cov0[0, 0]), _NUM_EPS**2)
    s_q2 = max(float(cov0[1, 1]), _NUM_EPS**2)
    c12 = float(cov0[0, 1])
    rho = c12 / (np.sqrt(s_t2 * s_q2) + _NUM_EPS)
    sig_t = float(np.sqrt(s_t2))
    sig_q = float(np.sqrt(s_q2))

    out = two_slope_rosenblatt_params(
        mu_t=mu_t,
        mu_q=mu_q,
        sig_t2_c=s_t2,
        sig_q2_c=s_q2,
        rho_tq=rho,
        m_t_dn=par["mT_dn"],
        m_t_up=par["mT_up"],
        m_q_dn=par["mQ_dn"],
        m_q_up=par["mQ_up"],
        s_t_dn=par["sT_dn"],
        s_t_up=par["sT_up"],
        s_q_dn=par["sQ_dn"],
        s_q_up=par["sQ_up"],
        H=H_layer,
        mean_gradient=mean_gradient,
    )
    t_min = 150.0
    q_max = 1.0e6
    n = int(n)
    p_nodes, p_w = _gl01(n)
    h_x, h_w = roots_hermite(n)
    w_outer = h_w / np.sqrt(np.pi)
    sqrt2 = np.sqrt(2.0)
    if out is None:
        return (
            0.0,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            False,
            "degenerate_gradient_fallback_center_gh",
        )

    M = out["M_inv"]
    s_v = out["s_v"]
    L_dn, L_up = out["L_dn"], out["L_up"]
    s0sq = out["s0sq"]
    r_c, r_fdn, r_fup = out["r_c"], out["r_fdn"], out["r_fup"]
    use_l = out["use_linvar"]
    s_s_dn, s_s_up = out["s_slope_dn"], out["s_slope_up"]
    s_udn, s_uup = out["s_u_dn"], out["s_u_up"]
    if (L_dn + L_up) <= _NUM_EPS and max(out["s_u_c"], s_udn, s_uup) <= _NUM_EPS:
        t_hat = np.full(1, mu_t)
        q_hat = np.full(1, mu_q)
        qs0 = q_sat_fn(t_hat)
        return (
            max(float(q_hat[0] - qs0[0]), 0.0),
            t_hat,
            q_hat,
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            use_l,
            "ok",
        )

    # ``δvec = M @ (u_i, v_j)`` matches Julia ``M_inv * SA.SVector(ui, vj)`` (see
    # ``subgrid_layer_profile_quadrature.jl``): columns are ``[d; d_⊥/‖d‖²]``, i.e. the same
    # orthonormal ``(T,q)`` rotation as notebook ``μ + M.T @ [u,v]`` after absorbing
    # ``‖d‖`` into the Rosenblatt ``u`` coordinate (``TestMInvVsNotebookRotation``).
    acc = 0.0
    t_list = []
    q_list = []
    w_list = []
    u_list = []
    v_list = []
    for j in range(n):
        vj = sqrt2 * s_v * h_x[j]
        wj = w_outer[j]
        mu0 = r_c * vj
        mus_dn = (r_fdn - r_c) * vj / max(L_dn, _NUM_EPS) if L_dn > _NUM_EPS else 0.0
        mus_up = (r_fup - r_c) * vj / max(L_up, _NUM_EPS) if L_up > _NUM_EPS else 0.0
        for i in range(n):
            pi_ = p_nodes[i]
            wi_ = p_w[i]
            if (L_dn + L_up) <= _NUM_EPS and max(out["s_u_c"], s_udn, s_uup) <= _NUM_EPS:
                ui = mu0
            elif use_l:
                ui = mixture_linvar_quantile_halley(
                    float(pi_),
                    mu0,
                    L_dn,
                    mus_dn,
                    s0sq,
                    s_s_dn,
                    L_up,
                    mus_up,
                    s_s_up,
                )
            else:
                ui = mixture_convolution_quantile_halley(float(pi_), L_dn, s_udn, L_up, s_uup) + mu0
            du = M[0, 0] * ui + M[0, 1] * vj
            dqv = M[1, 0] * ui + M[1, 1] * vj
            # Same center Gaussian as affine ``(μ_T+δT, μ_q+δq)`` until SGS bounds bite.
            t_hat, q_hat = fluc_to_tq(
                np.array([du], dtype=float),
                np.array([dqv], dtype=float),
                mu_t, mu_q, sig_t, sig_q, rho, t_min, q_max,
            )
            qs_ = q_sat_fn(t_hat)
            dq = max(float(q_hat[0] - qs_[0]), 0.0)
            wtot = wj * wi_
            acc += wtot * dq
            t_list.append(float(t_hat[0]))
            q_list.append(float(q_hat[0]))
            w_list.append(wtot)
            u_list.append(float(ui))
            v_list.append(float(vj))
    t_arr = np.array(t_list, dtype=float)
    q_arr = np.array(q_list, dtype=float)
    w_arr = np.array(w_list, dtype=float)
    u_arr = np.array(u_list, dtype=float)
    v_arr = np.array(v_list, dtype=float)
    return acc, t_arr, q_arr, w_arr, u_arr, v_arr, use_l, "ok"


def profile_rosenblatt_cubature_two_halves_cell(
    n: int,
    par: dict,
    cov0: np.ndarray,
    H_layer: float,
    mu_t: float,
    mu_q: float,
    q_sat_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    str,
]:
    """
    **Two-half cell** profile–Rosen condensate: average the DN-only cubature and the
    UP-only cubature along their respective mean-gradient axes (``½`` weight each when both
    succeed). This matches the explicit two-half mean paths used in the mixture
    marginal far better than a single transport built from **averaged** slopes.

    Physical ``(T,q)`` nodes count ``2 N^2`` when both halves are non-degenerate.
    Returns the same tuple shape as :func:`profile_rosenblatt_cubature`.
    """
    n = int(n)
    res_dn = profile_rosenblatt_cubature(
        n, par, cov0, H_layer, mu_t, mu_q, q_sat_fn, mean_gradient="dn"
    )
    res_up = profile_rosenblatt_cubature(
        n, par, cov0, H_layer, mu_t, mu_q, q_sat_fn, mean_gradient="up"
    )
    qc_dn, t_dn, q_dn, w_dn, u_dn, v_dn, use_ld, st_dn = res_dn
    qc_up, t_up, q_up, w_up, u_up, v_up, use_lu, st_up = res_up
    ok_dn = st_dn == "ok" and t_dn.size > 0
    ok_up = st_up == "ok" and t_up.size > 0
    w_dn_half = 0.5 if ok_dn else 0.0
    w_up_half = 0.5 if ok_up else 0.0
    w_sum = w_dn_half + w_up_half
    if w_sum <= 0.0:
        return (
            0.0,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            False,
            "degenerate_gradient_fallback_center_gh",
        )
    a_dn = w_dn_half / w_sum
    a_up = w_up_half / w_sum
    qc = float(a_dn * qc_dn + a_up * qc_up)
    t_list: list[np.ndarray] = []
    q_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    u_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []
    if ok_dn:
        t_list.append(t_dn)
        q_list.append(q_dn)
        w_list.append((a_dn * w_dn).astype(float))
        u_list.append(u_dn)
        v_list.append(v_dn)
    if ok_up:
        t_list.append(t_up)
        q_list.append(q_up)
        w_list.append((a_up * w_up).astype(float))
        u_list.append(u_up)
        v_list.append(v_up)
    t_arr = np.concatenate(t_list) if t_list else np.array([])
    q_arr = np.concatenate(q_list) if q_list else np.array([])
    w_arr = np.concatenate(w_list) if w_list else np.array([])
    u_arr = np.concatenate(u_list) if u_list else np.array([])
    v_arr = np.concatenate(v_list) if v_list else np.array([])
    use_l = bool(use_ld or use_lu)
    st = "ok" if ok_dn and ok_up else ("half_dn_only" if ok_dn else "half_up_only")
    return qc, t_arr, q_arr, w_arr, u_arr, v_arr, use_l, st


def _profile_half_physical_density_core(
    T: np.ndarray,
    Q: np.ndarray,
    *,
    mu_t: float,
    mu_q: float,
    out: dict,
) -> np.ndarray:
    """
    Closed-form ``(T,Q)`` density for **one** DN or UP Rosenblatt half, matching the
    factorization used in :func:`profile_rosenblatt_cubature`: same joint density ``f_{U,V}`` as
    in ``(u,v)`` space, then pushforward to ``(T,Q)`` via ``(δ_T,δ_Q)^T = M (u,v)^T`` with
    ``M = out['M_inv']`` (``|det M| = 1`` here). With **unclamped** :func:`fluc_to_tq`,
    ``(T,Q) = (μ_T+δ_T, μ_q+δ_Q)``, so ``f_{T,Q}(μ+δ) = f_{U,V}(M^{-1}δ)``. Here ``V ~ N(0, s_v^2)``,
    and ``U|V`` is either ``mixture_linvar`` (``use_linvar``) or ``mixture_uniform_convolution``
    (frozen-σ branch), **before** ``q``/``T`` physical clamps. Same ``M`` as production / notebook
    rotation tests.
    """
    M = np.asarray(out["M_inv"], dtype=float).reshape(2, 2)
    s_v = float(out["s_v"])
    L_dn = float(out["L_dn"])
    L_up = float(out["L_up"])
    s_udn = float(out["s_u_dn"])
    s_uup = float(out["s_u_up"])
    r_c = float(out["r_c"])
    r_fdn = float(out["r_fdn"])
    r_fup = float(out["r_fup"])
    s0sq = float(out["s0sq"])
    s_s_dn = float(out["s_slope_dn"])
    s_s_up = float(out["s_slope_up"])
    use_l = bool(out["use_linvar"])
    eps = _NUM_EPS

    dT = np.asarray(T, dtype=float) - float(mu_t)
    dQ = np.asarray(Q, dtype=float) - float(mu_q)
    shp = dT.shape
    dtr = dT.ravel()
    dqr = dQ.ravel()
    dvec = np.stack([dtr, dqr], axis=0)
    uv = np.linalg.solve(M, dvec)
    ur = uv[0]
    vr = uv[1]

    sv = max(s_v, eps)
    fV = (1.0 / (np.sqrt(2.0 * np.pi) * sv)) * np.exp(-0.5 * (vr / sv) ** 2)
    mu0r = r_c * vr
    inv_ldn = 1.0 / max(L_dn, eps)
    inv_lup = 1.0 / max(L_up, eps)
    mus_dn_v = np.where(L_dn > eps, (r_fdn - r_c) * vr * inv_ldn, 0.0)
    mus_up_v = np.where(L_up > eps, (r_fup - r_c) * vr * inv_lup, 0.0)

    n = ur.size
    fU = np.empty(n, dtype=float)
    if use_l:
        for k in range(n):
            fU[k] = mixture_linvar_pdf(
                float(ur[k]),
                float(mu0r[k]),
                L_dn,
                float(mus_dn_v[k]),
                s0sq,
                s_s_dn,
                L_up,
                float(mus_up_v[k]),
                s0sq,
                s_s_up,
            )
    else:
        for k in range(n):
            u0 = float(ur[k]) - float(mu0r[k])
            fU[k] = mixture_uniform_convolution_pdf(u0, L_dn, s_udn, L_up, s_uup)
    dens = (fU * fV).reshape(shp)
    return dens


def profile_rosenblatt_two_half_physical_density(
    T: np.ndarray,
    Q: np.ndarray,
    *,
    par: dict,
    cov0: np.ndarray,
    H_layer: float,
    mu_t: float,
    mu_q: float,
) -> tuple[np.ndarray, str]:
    r"""
    **Two-half** profile–Rosenblatt physical ``(T,Q)`` density (same transport weights as
    :func:`profile_rosenblatt_cubature_two_halves_cell`):

    ``p(T,Q) = a_{\mathrm{dn}}\,p_{\mathrm{dn}}(T,Q) + a_{\mathrm{up}}\,p_{\mathrm{up}}(T,Q)``,

    where each ``p_{\mathrm{half}}`` is the closed-form density from
    :func:`_profile_half_physical_density_core`, and ``a_{\mathrm{dn}},a_{\mathrm{up}}`` are
    ``½`` when both halves are non-degenerate (else the surviving half gets weight ``1``).

    This uses the **unclamped** ``(μ_T+δT,μ_q+δq)`` identification from the interior of
    :func:`fluc_to_tq` (no ``q\ge0`` / ``T\ge t_{\min}`` boundary mass). Callers evaluate on a
    ``(T,q)`` tensor grid as needed (e.g. tests in ``test_profile_rosenblatt_primitives.py``). Not for
    ``q_{\mathrm{sat}}`` condensate clipping.
    """
    s_t2 = max(float(cov0[0, 0]), _NUM_EPS**2)
    s_q2 = max(float(cov0[1, 1]), _NUM_EPS**2)
    c12 = float(cov0[0, 1])
    rho_tq = c12 / (np.sqrt(s_t2 * s_q2) + _NUM_EPS)

    out_dn = two_slope_rosenblatt_params(
        mu_t=float(mu_t),
        mu_q=float(mu_q),
        sig_t2_c=s_t2,
        sig_q2_c=s_q2,
        rho_tq=float(rho_tq),
        m_t_dn=float(par["mT_dn"]),
        m_t_up=float(par["mT_up"]),
        m_q_dn=float(par["mQ_dn"]),
        m_q_up=float(par["mQ_up"]),
        s_t_dn=float(par["sT_dn"]),
        s_t_up=float(par["sT_up"]),
        s_q_dn=float(par["sQ_dn"]),
        s_q_up=float(par["sQ_up"]),
        H=float(H_layer),
        mean_gradient="dn",
    )
    out_up = two_slope_rosenblatt_params(
        mu_t=float(mu_t),
        mu_q=float(mu_q),
        sig_t2_c=s_t2,
        sig_q2_c=s_q2,
        rho_tq=float(rho_tq),
        m_t_dn=float(par["mT_dn"]),
        m_t_up=float(par["mT_up"]),
        m_q_dn=float(par["mQ_dn"]),
        m_q_up=float(par["mQ_up"]),
        s_t_dn=float(par["sT_dn"]),
        s_t_up=float(par["sT_up"]),
        s_q_dn=float(par["sQ_dn"]),
        s_q_up=float(par["sQ_up"]),
        H=float(H_layer),
        mean_gradient="up",
    )
    ok_dn = out_dn is not None
    ok_up = out_up is not None
    w_dn_half = 0.5 if ok_dn else 0.0
    w_up_half = 0.5 if ok_up else 0.0
    w_sum = w_dn_half + w_up_half
    if w_sum <= 0.0:
        return np.zeros_like(T, dtype=float), "degenerate_gradient"
    a_dn = w_dn_half / w_sum
    a_up = w_up_half / w_sum

    p = np.zeros_like(np.asarray(T, dtype=float), dtype=float)
    if ok_dn:
        p += a_dn * _profile_half_physical_density_core(
            T, Q, mu_t=mu_t, mu_q=mu_q, out=out_dn
        )
    if ok_up:
        p += a_up * _profile_half_physical_density_core(
            T, Q, mu_t=mu_t, mu_q=mu_q, out=out_up
        )
    return p, "ok"


class ExactAnalyticTailEngine:
    """
    Same ν = 1/2 ``erfc`` / ``erfcx`` closed form as in ``Variance_Stuff`` derivations;
    delegates to :func:`gamma_tail_nu_half` (scalar or NumPy arrays).
    """

    def psi_1_2(self, t, b):
        t = np.asarray(t, dtype=float)
        b = np.asarray(b, dtype=float)
        return np.vectorize(gamma_tail_nu_half, otypes=[float])(t, b)
