#!/usr/bin/env python3
"""
Layer cell: **Julia-matched** slab-``\\Sigma`` vertical PDF + Halley notebook overlay.

1. **Heatmap / black reference:** :func:`brute_pdf_z_stack` only — Riemann
   ``(1/N_z)\\sum_k N(T,q;\\mu(\\zeta_k),\\Sigma_{\\mathrm{half}})``. On each half the
   **same** turbulent ``\\Sigma`` is used at every ``\\zeta_k``, built from cell-center
   variances with the **``dz_q = H/4`` walk** (``\\sigma^2_k \\pm dz_q\\,\\partial\\sigma^2_k/\\partial z``),
   matching ``profile_rosenblatt_two_half_cell.two_slope_rosenblatt_params`` and Julia
   ``SubgridProfileRosenblatt`` / ``_two_slope_rosenblatt_params`` (covariance sampled at
   **half-centers** ``\\pm H/4`` from center — see ``subgrid_layer_profile_quadrature.jl``
   docstring near ``±H/4``). **Means** ``\\mu(\\zeta)`` are piecewise-linear in ``\\zeta``.
   There is **no** affine-``\\Sigma(\\zeta)`` interpolation at every height in this script;
   use ``variance_dashboard_interactive.mixture_pdf_grid`` separately if you need that
   dashboard object.

2. **Halley ``eval_pdf`` / split-``p`` (purple):** For **constant** ``\\Sigma`` on each
   half and **linear** ``\\mu(\\zeta)``, the depth-uniform marginal is the closed form
   ``0.5\\,p_{\\mathrm{dn}} + 0.5\\,p_{\\mathrm{up}}`` in ``Halley_tests_gemini`` (see
   ``docs/layer_mean_cell_Tq_marginal_derivation.md`` §3). That **is** the same local
   Gaussian family as the **continuous** ``\\zeta``-average; :func:`brute_pdf_z_stack` is
   only a **Riemann** approximation of that average. **Important:** ``eval_pdf`` must use
   the **same** per-half ``(σ_T,σ_q,ρ)`` as the slab stack. This script wires
   :func:`slab_z_stack_half_stddevs` into :func:`notebook_analytical_pdf_grid` and
   :func:`notebook_split_p_quadrature` so purple and black describe **one** law (up to
   finite ``Nz``). The old demo choice ``s_{T,\\mathrm{up}}=0.8`` vs walked-upper-``σ``
   from ``cov0`` was a **parameter mismatch**, not a second physics.

3. **Nested quadrature (green):** same outer Riemann × inner 2D GH as the black discrete
   law — a numerical cross-check, not a replacement for the analytic line.

4. **Julia-style profile–Rosenblatt:** weighted sum from
   ``collect_tq_cloud_julia_style`` with **½** weight on each ``mean_gradient``
   pass (``dn`` / ``up``), matching ``0.5*(a_dn+a_up)`` — a **different** construction
   than the uniform-``\\zeta`` slab average in general (see layer-marginal notes).

Run::

    python calibration/experiments/variance_adjustments/local_python/profile_cell_brute_pdf_quadrature_convergence.py

Requires: numpy, scipy, matplotlib.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy import stats

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from plot_profile_rosenblatt_tq_nodes_julia_style import (  # noqa: E402
    collect_tq_cloud_julia_style,
)


# ---------------------------------------------------------------------------
# Thermodynamics (simple saturation for a scalar diagnostic)
# ---------------------------------------------------------------------------


def q_sat_kgkg(T: np.ndarray, p_pa: float = 85_000.0) -> np.ndarray:
    """Magnus-style saturation specific humidity [kg/kg]; toy pressures."""
    T = np.asarray(T, dtype=float)
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    return 0.622 * es / (p_pa - es + 1e-9)


def saturation_excess_kgkg(T: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, q - q_sat_kgkg(T))


# ---------------------------------------------------------------------------
# Notebook-style split-p quadrature (Halley_tests_gemini cell 3)
# ---------------------------------------------------------------------------


def _cholesky_inv_det(sig_t: float, sig_q: float, rho: float):
    rho = float(np.clip(rho, -0.99, 0.99))
    sig_t = max(sig_t, 1e-12)
    sig_q = max(sig_q, 1e-12)
    s11 = sig_t**2
    s22 = sig_q**2
    s12 = rho * sig_t * sig_q
    Sigma = np.array([[s11, s12], [s12, s22]], dtype=float)
    C = np.linalg.cholesky(Sigma)
    Ci = np.linalg.inv(C)
    det = float(sig_t * sig_q * np.sqrt(max(1.0 - rho**2, 1e-20)))
    return Sigma, C, Ci, det


def _f_half(x: np.ndarray, a: float, b: float, s: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if abs(b - a) < 1e-10:
        return stats.norm.pdf(x / s) / s
    return (stats.norm.cdf((x - a) / s) - stats.norm.cdf((x - b) / s)) / (b - a)


def _F_half(x: np.ndarray, a: float, b: float, s: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if abs(b - a) < 1e-10:
        return stats.norm.cdf(x / s)

    def mills(y):
        return y * stats.norm.cdf(y) + stats.norm.pdf(y)

    z_a = (x - a) / s
    z_b = (x - b) / s
    return (s / (b - a)) * (mills(z_a) - mills(z_b))


def _invert_leg(p: np.ndarray, L: float) -> np.ndarray:
    p_c = np.clip(p, 1e-10, 1.0 - 1e-10)
    u_unif = L * p_c
    u_gauss = np.sqrt(L**2 / 12.0 + 1.0) * stats.norm.ppf(p_c) + L / 2.0
    w = (1.0 / L) ** 2 / ((1.0 / L) ** 2 + 0.08) if L > 1e-12 else 1.0
    u0 = (1.0 - w) * u_unif + w * u_gauss
    # Halley notebook uses two passes; a few extra iterations cheaply tightens ``u`` for large ``N``.
    for _ in range(6):
        G = _F_half(u0, 0.0, L) - p_c
        Gp = _f_half(u0, 0.0, L)
        Gpp = (stats.norm.pdf(u0) - stats.norm.pdf(u0 - L)) / L
        u0 = np.where(np.abs(Gp) > 1e-12, u0 - (2 * G * Gp) / (2 * Gp**2 - G * Gpp), u0)
    return u0


def notebook_split_p_quadrature(
    n: int,
    T_c: float,
    q_c: float,
    *,
    dT_dn: float,
    dq_dn: float,
    sT_dn: float,
    sq_dn: float,
    r_dn: float,
    dT_up: float,
    dq_up: float,
    sT_up: float,
    sq_up: float,
    r_up: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T_nodes, q_nodes, weights) for Halley_tests_gemini split-p rule."""
    v_raw, v_w_raw = hermgauss(n)
    v_n, v_w = v_raw * np.sqrt(2.0), v_w_raw / np.sqrt(np.pi)
    u_raw, u_w_raw = leggauss(n)
    p_n, p_w = (u_raw + 1.0) / 2.0, u_w_raw / 2.0

    _, _, Ci_dn, _ = _cholesky_inv_det(sT_dn, sq_dn, r_dn)
    _, _, Ci_up, _ = _cholesky_inv_det(sT_up, sq_up, r_up)

    T_all: list[float] = []
    q_all: list[float] = []
    W_all: list[float] = []

    for p, wp in zip(p_n, p_w):
        if p < 0.5:
            w_vec = Ci_dn @ np.array([dT_dn, dq_dn], dtype=float)
            L = float(np.linalg.norm(w_vec))
            ang = float(np.arctan2(w_vec[1], w_vec[0]))
            _, C, _, _ = _cholesky_inv_det(sT_dn, sq_dn, r_dn)
            u_node = _invert_leg(np.atleast_1d(1.0 - 2.0 * p), L)[0]
        elif p > 0.5:
            w_vec = Ci_up @ np.array([dT_up, dq_up], dtype=float)
            L = float(np.linalg.norm(w_vec))
            ang = float(np.arctan2(w_vec[1], w_vec[0]))
            _, C, _, _ = _cholesky_inv_det(sT_up, sq_up, r_up)
            u_node = _invert_leg(np.atleast_1d(2.0 * p - 1.0), L)[0]
        else:
            u_node = 0.0
            ang = 0.0
            _, C, _, _ = _cholesky_inv_det(
                0.5 * (sT_dn + sT_up), 0.5 * (sq_dn + sq_up), 0.5 * (r_dn + r_up)
            )

        c, s = np.cos(ang), np.sin(ang)
        for vn, wv in zip(v_n, v_w):
            ur = u_node * c - vn * s
            vr = u_node * s + vn * c
            T_all.append(float(C[0, 0] * ur + C[0, 1] * vr + T_c))
            q_all.append(float(C[1, 0] * ur + C[1, 1] * vr + q_c))
            W_all.append(float(wp * wv))

    return np.asarray(T_all), np.asarray(q_all), np.asarray(W_all)


# ---------------------------------------------------------------------------
# z-stacked brute PDF on (T,q) absolute grid
# ---------------------------------------------------------------------------


def brute_pdf_z_stack(
    T_grid: np.ndarray,
    q_grid: np.ndarray,
    *,
    T_c: float,
    q_c: float,
    H: float,
    nz: int,
    dT_dn: float,
    dq_dn: float,
    dT_up: float,
    dq_up: float,
    s_t2_c: float,
    s_q2_c: float,
    rho_c: float,
    sT_dn: float,
    sT_up: float,
    sQ_dn: float,
    sQ_up: float,
) -> np.ndarray:
    """
    ``p(T,q) ≈ (1/Nz) Σ_k N((T,q); μ(ζ_k), Σ_{\\mathrm{dn/up}})`` with piecewise-linear
    ``μ(ζ)`` and **one** lower-half and **one** upper-half covariance matrix.

    Variances use ``dz_q = 0.25 H`` (half-center offset from layer midpoint), i.e. the same
    ``σ²_k(center) ± dz_q · ∂σ²_k/∂z`` construction as ``two_slope_rosenblatt_params`` in
    ``profile_rosenblatt_two_half_cell.py`` and the Julia profile-Rosenblatt note on
    sampling ``Σ`` at ``±H/4`` (not a full affine ``Σ(ζ)`` at every ``ζ``).
    """
    dz_q = 0.25 * H
    s_t2_lo = max(s_t2_c - dz_q * sT_dn, 0.0)
    s_t2_hi = max(s_t2_c + dz_q * sT_up, 0.0)
    s_q2_lo = max(s_q2_c - dz_q * sQ_dn, 0.0)
    s_q2_hi = max(s_q2_c + dz_q * sQ_up, 0.0)
    rho = float(np.clip(rho_c, -0.99, 0.99))
    sig11_lo = max(s_t2_lo, 1e-20)
    sig22_lo = max(s_q2_lo, 1e-20)
    sig12_lo = rho * np.sqrt(sig11_lo * sig22_lo)
    sig11_hi = max(s_t2_hi, 1e-20)
    sig22_hi = max(s_q2_hi, 1e-20)
    sig12_hi = rho * np.sqrt(sig11_hi * sig22_hi)

    z_centers = (np.arange(nz, dtype=float) + 0.5) / nz * H - 0.5 * H  # (-H/2, H/2]

    TT, QQ = np.meshgrid(T_grid, q_grid, indexing="xy")
    pdf = np.zeros_like(TT, dtype=float)

    for z in z_centers:
        if z <= 0.0:
            mu_t = T_c + dT_dn * (-2.0 * z / H)
            mu_q = q_c + dq_dn * (-2.0 * z / H)
            cov = np.array([[sig11_lo, sig12_lo], [sig12_lo, sig22_lo]], dtype=float)
        else:
            mu_t = T_c + dT_up * (2.0 * z / H)
            mu_q = q_c + dq_up * (2.0 * z / H)
            cov = np.array([[sig11_hi, sig12_hi], [sig12_hi, sig22_hi]], dtype=float)

        rv = stats.multivariate_normal(mean=np.array([mu_t, mu_q]), cov=cov)
        pdf += rv.pdf(np.stack([TT, QQ], axis=-1))

    pdf /= float(nz)
    return pdf


def slab_z_stack_half_stddevs(
    *,
    H: float,
    s_t2_c: float,
    s_q2_c: float,
    rho_c: float,
    sT_dn: float,
    sT_up: float,
    sQ_dn: float,
    sQ_up: float,
) -> tuple[float, float, float, float, float]:
    """
    Marginal std devs ``σ_T``, ``σ_q`` on lower / upper half and ``ρ``, matching the
    ``2×2`` blocks in :func:`brute_pdf_z_stack` (``dz_q = H/4`` walk, center ``ρ``).

    Use these as ``(sT_dn, sq_dn, r_dn)`` / ``(sT_up, sq_up, r_up)`` in
    :func:`notebook_analytical_pdf_grid` and :func:`notebook_split_p_quadrature` so the
    Halley ``eval_pdf`` density refers to the **same** frozen half-slab MVN as the z-stack.
    """
    dz_q = 0.25 * H
    s_t2_lo = max(s_t2_c - dz_q * sT_dn, 0.0)
    s_t2_hi = max(s_t2_c + dz_q * sT_up, 0.0)
    s_q2_lo = max(s_q2_c - dz_q * sQ_dn, 0.0)
    s_q2_hi = max(s_q2_c + dz_q * sQ_up, 0.0)
    rho = float(np.clip(rho_c, -0.99, 0.99))
    sig11_lo = max(s_t2_lo, 1e-20)
    sig22_lo = max(s_q2_lo, 1e-20)
    sig11_hi = max(s_t2_hi, 1e-20)
    sig22_hi = max(s_q2_hi, 1e-20)
    s_t_lo = float(np.sqrt(sig11_lo))
    s_q_lo = float(np.sqrt(sig22_lo))
    s_t_hi = float(np.sqrt(sig11_hi))
    s_q_hi = float(np.sqrt(sig22_hi))
    return s_t_lo, s_q_lo, s_t_hi, s_q_hi, rho


def pdf_on_fluctuation_grid(
    *,
    T_c: float,
    q_c: float,
    extent_t: float,
    extent_q_kgkg: float,
    n_grid: int,
    **kwargs_z,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slab-``Σ`` per half + Riemann ``ζ``: :func:`brute_pdf_z_stack` on ``(dT,dq)`` (default ``main`` path)."""
    dT = np.linspace(-extent_t, extent_t, n_grid)
    dq = np.linspace(-extent_q_kgkg, extent_q_kgkg, n_grid)
    T_lin = T_c + dT
    q_lin = q_c + dq
    pdf = brute_pdf_z_stack(T_lin, q_lin, T_c=T_c, q_c=q_c, **kwargs_z)
    return dT, dq, pdf, (dT[1] - dT[0]) * (dq[1] - dq[0])


def slab_z_stack_nested_gh_expectation(
    n: int,
    f,
    *,
    T_c: float,
    q_c: float,
    H: float,
    nz: int,
    dT_dn: float,
    dq_dn: float,
    dT_up: float,
    dq_up: float,
    s_t2_c: float,
    s_q2_c: float,
    rho_c: float,
    sT_dn: float,
    sT_up: float,
    sQ_dn: float,
    sQ_up: float,
) -> float:
    """
    Same outer measure as :func:`brute_pdf_z_stack`: ``(1/N_z) Σ_ζ`` with piecewise
    ``μ(ζ)`` and half-slab ``Σ`` (``dz_q = H/4`` variance walk). Inner expectation at
    each ``ζ`` is a **2D tensor Gauss–Hermite** rule for ``N(μ, Σ)`` (independent
    standard normals via Cholesky). Converges to ``∫ f p`` as ``n → ∞`` (up to ``ζ``
    Riemann error from finite ``nz``).
    """
    dz_q = 0.25 * H
    s_t2_lo = max(s_t2_c - dz_q * sT_dn, 0.0)
    s_t2_hi = max(s_t2_c + dz_q * sT_up, 0.0)
    s_q2_lo = max(s_q2_c - dz_q * sQ_dn, 0.0)
    s_q2_hi = max(s_q2_c + dz_q * sQ_up, 0.0)
    rho = float(np.clip(rho_c, -0.99, 0.99))
    sig11_lo = max(s_t2_lo, 1e-20)
    sig22_lo = max(s_q2_lo, 1e-20)
    sig12_lo = rho * np.sqrt(sig11_lo * sig22_lo)
    sig11_hi = max(s_t2_hi, 1e-20)
    sig22_hi = max(s_q2_hi, 1e-20)
    sig12_hi = rho * np.sqrt(sig11_hi * sig22_hi)

    z_centers = (np.arange(nz, dtype=float) + 0.5) / nz * H - 0.5 * H
    x_gh, w_raw = hermgauss(n)
    chi = x_gh * np.sqrt(2.0)
    w_gh = w_raw / np.sqrt(np.pi)

    wz = 1.0 / float(nz)
    acc = 0.0
    for z in z_centers:
        if z <= 0.0:
            mu_t = T_c + dT_dn * (-2.0 * z / H)
            mu_q = q_c + dq_dn * (-2.0 * z / H)
            cov = np.array([[sig11_lo, sig12_lo], [sig12_lo, sig22_lo]], dtype=float)
        else:
            mu_t = T_c + dT_up * (2.0 * z / H)
            mu_q = q_c + dq_up * (2.0 * z / H)
            cov = np.array([[sig11_hi, sig12_hi], [sig12_hi, sig22_hi]], dtype=float)
        L = np.linalg.cholesky(cov)
        for i in range(n):
            for j in range(n):
                d = L @ np.array([chi[i], chi[j]], dtype=float)
                T = mu_t + d[0]
                q = mu_q + d[1]
                acc += wz * w_gh[i] * w_gh[j] * float(f(T, q))
    return float(acc)


def slab_z_stack_slice_gh_fluctuation_points(
    n: int,
    z: float,
    *,
    T_c: float,
    q_c: float,
    H: float,
    dT_dn: float,
    dq_dn: float,
    dT_up: float,
    dq_up: float,
    s_t2_c: float,
    s_q2_c: float,
    rho_c: float,
    sT_dn: float,
    sT_up: float,
    sQ_dn: float,
    sQ_up: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D GH tensor nodes in ``(T-T_c, q-q_c)`` at one fixed ``ζ`` (same ``Σ`` as z-stack)."""
    dz_q = 0.25 * H
    s_t2_lo = max(s_t2_c - dz_q * sT_dn, 0.0)
    s_t2_hi = max(s_t2_c + dz_q * sT_up, 0.0)
    s_q2_lo = max(s_q2_c - dz_q * sQ_dn, 0.0)
    s_q2_hi = max(s_q2_c + dz_q * sQ_up, 0.0)
    rho = float(np.clip(rho_c, -0.99, 0.99))
    sig11_lo = max(s_t2_lo, 1e-20)
    sig22_lo = max(s_q2_lo, 1e-20)
    sig12_lo = rho * np.sqrt(sig11_lo * sig22_lo)
    sig11_hi = max(s_t2_hi, 1e-20)
    sig22_hi = max(s_q2_hi, 1e-20)
    sig12_hi = rho * np.sqrt(sig11_hi * sig22_hi)

    if z <= 0.0:
        cov = np.array([[sig11_lo, sig12_lo], [sig12_lo, sig22_lo]], dtype=float)
    else:
        cov = np.array([[sig11_hi, sig12_hi], [sig12_hi, sig22_hi]], dtype=float)

    x_gh, w_raw = hermgauss(n)
    chi = x_gh * np.sqrt(2.0)
    w_gh = w_raw / np.sqrt(np.pi)
    L = np.linalg.cholesky(cov)

    dT_list: list[float] = []
    dq_list: list[float] = []
    w_list: list[float] = []
    for i in range(n):
        for j in range(n):
            d = L @ np.array([chi[i], chi[j]], dtype=float)
            dT_list.append(float(d[0]))
            dq_list.append(float(d[1]))
            w_list.append(float(w_gh[i] * w_gh[j]))
    return np.asarray(dT_list), np.asarray(dq_list), np.asarray(w_list)


def notebook_analytical_pdf_grid(
    T_lin: np.ndarray,
    q_lin: np.ndarray,
    *,
    T_c: float,
    q_c: float,
    dT_dn: float,
    dq_dn: float,
    sT_dn: float,
    sq_dn: float,
    r_dn: float,
    dT_up: float,
    dq_up: float,
    sT_up: float,
    sq_up: float,
    r_up: float,
) -> np.ndarray:
    """``0.5\\,p_{\\mathrm{dn}} + 0.5\\,p_{\\mathrm{up}}`` from ``Halley_tests_gemini`` ``eval_pdf``.

    For the **layer** slab test, pass ``(sT_dn,\\ldots)`` / ``(sT_up,\\ldots)`` from
    :func:`slab_z_stack_half_stddevs` so each half matches :func:`brute_pdf_z_stack`'s
    frozen ``2×2`` block (constant ``Σ`` per half, linear ``μ(ζ)``; see
    ``layer_mean_cell_Tq_marginal_derivation.md`` §3).
    """
    T_g, q_g = np.meshgrid(T_lin, q_lin, indexing="xy")

    def eval_pdf(T, q, dT, dq, sT, sq, r):
        _, _, Ci, det = _cholesky_inv_det(sT, sq, r)
        w_vec = Ci @ np.array([dT, dq], dtype=float)
        L = float(np.linalg.norm(w_vec))
        ang = float(np.arctan2(w_vec[1], w_vec[0]))
        rel_T, rel_q = T - T_c, q - q_c
        u_w = Ci[0, 0] * rel_T + Ci[0, 1] * rel_q
        v_w = Ci[1, 0] * rel_T + Ci[1, 1] * rel_q
        c, s = np.cos(-ang), np.sin(-ang)
        ua = u_w * c - v_w * s
        va = u_w * s + v_w * c
        return _f_half(ua, 0.0, L) * stats.norm.pdf(va, 0.0, 1.0) / det

    return 0.5 * eval_pdf(
        T_g, q_g, dT_dn, dq_dn, sT_dn, sq_dn, r_dn
    ) + 0.5 * eval_pdf(T_g, q_g, dT_up, dq_up, sT_up, sq_up, r_up)


# ---------------------------------------------------------------------------
# Julia-style weighted expectation
# ---------------------------------------------------------------------------


def julia_profile_expectation(
    n: int,
    f,
    *,
    par: dict,
    cov0: np.ndarray,
    H_layer: float,
    mu_t: float,
    mu_q: float,
) -> float:
    s = 0.0
    for mg in ("dn", "up"):
        T, q, w, _ = collect_tq_cloud_julia_style(
            n=n,
            par=par,
            cov0=cov0,
            H_layer=H_layer,
            mu_t=mu_t,
            mu_q=mu_q,
            mean_gradients=(mg,),
        )
        if T.size == 0:
            continue
        s += 0.5 * float(np.sum(w * f(T, q)))
    return s


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def main() -> None:
    import matplotlib.pyplot as plt

    # --- same spirit as Halley_tests_gemini defaults (slopes in *absolute* T,q
    # offsets over each half-slab of thickness H/2) ---
    H = 400.0
    T_c = 285.0
    # Place layer center **slightly subsaturated** at ``T_c`` so the joint bulk straddles the CC
    # line (previously ``q_c=0.012`` was already supersaturated vs ``q_sat(T_c)``, so almost all
    # mass sat above the boundary in fluctuation space).
    q_sat_c = float(q_sat_kgkg(np.array([T_c]))[0])
    q_c = q_sat_c - 4.0e-4  # kg/kg; tune 2e-4–6e-4 to move mass across the boundary
    # Halley_tests_gemini defaults: ``dT`` in K, ``dq`` in **kg/kg** (sliders show g/kg but code adds to ``q_mid``).
    dT_dn, dq_dn = -1.5, -0.0008
    dT_up, dq_up = 1.0, 0.0015
    # Notebook Cholesky uses **std dev** ``sT_dn=1``, ``sq_dn=0.6`` **g/kg** → SI:
    s_t_dn_nb, s_q_dn_nb = 1.0, 0.6e-3
    s_t_up_nb, s_q_up_nb = 0.8, 1.2e-3
    sT_dn, sT_up = 1.0e-6, 2.0e-6  # variance slopes dσ²/dz [SI units / m]
    sQ_dn, sQ_up = 1.0e-9, 2.0e-9
    cov0 = np.array(
        [[s_t_dn_nb**2, 0.9 * s_t_dn_nb * s_q_dn_nb], [0.9 * s_t_dn_nb * s_q_dn_nb, s_q_dn_nb**2]],
        dtype=float,
    )
    rho_c = float(cov0[0, 1] / (np.sqrt(cov0[0, 0] * cov0[1, 1]) + 1e-15))

    par = {
        "mT_dn": dT_dn / (0.5 * H),
        "mT_up": dT_up / (0.5 * H),
        "mQ_dn": dq_dn / (0.5 * H),
        "mQ_up": dq_up / (0.5 * H),
        "sT_dn": sT_dn,
        "sT_up": sT_up,
        "sQ_dn": sQ_dn,
        "sQ_up": sQ_up,
    }

    s_t2_c, s_q2_c = float(cov0[0, 0]), float(cov0[1, 1])

    f = saturation_excess_kgkg

    n_grid = 260  # finer grid tightens ``truth_*`` vs tensor quadrature
    nz = 400
    extent_t = 6.0
    extent_q = 0.006
    dT, dq, pdf_z, dA = pdf_on_fluctuation_grid(
        T_c=T_c,
        q_c=q_c,
        extent_t=extent_t,
        extent_q_kgkg=extent_q,
        n_grid=n_grid,
        H=H,
        nz=nz,
        dT_dn=dT_dn,
        dq_dn=dq_dn,
        dT_up=dT_up,
        dq_up=dq_up,
        s_t2_c=s_t2_c,
        s_q2_c=s_q2_c,
        rho_c=rho_c,
        sT_dn=sT_dn,
        sT_up=sT_up,
        sQ_dn=sQ_dn,
        sQ_up=sQ_up,
    )

    # ``pdf_z[j,i]`` and ``pdf_nb[j,i]`` use ``meshgrid(..., indexing='xy')``: row ``j`` → ``q``,
    # column ``i`` → ``T``.
    TT, QQ = np.meshgrid(T_c + dT, q_c + dq, indexing="xy")
    phi = f(TT, QQ)
    truth_z = float(np.sum(pdf_z * phi) * dA)

    s_t_lo, s_q_lo, s_t_hi, s_q_hi, rho_half = slab_z_stack_half_stddevs(
        H=H,
        s_t2_c=s_t2_c,
        s_q2_c=s_q2_c,
        rho_c=rho_c,
        sT_dn=sT_dn,
        sT_up=sT_up,
        sQ_dn=sQ_dn,
        sQ_up=sQ_up,
    )

    T_lin = T_c + dT
    q_lin = q_c + dq
    # Halley ``eval_pdf`` closed form for the **same** half-slab MVNs as ``brute_pdf_z_stack``
    # (not the old demo ``s_t_up_nb=0.8`` vs walked ``σ`` mismatch).
    pdf_nb = notebook_analytical_pdf_grid(
        T_lin,
        q_lin,
        T_c=T_c,
        q_c=q_c,
        dT_dn=dT_dn,
        dq_dn=dq_dn,
        sT_dn=s_t_lo,
        sq_dn=s_q_lo,
        r_dn=rho_half,
        dT_up=dT_up,
        dq_up=dq_up,
        sT_up=s_t_hi,
        sq_up=s_q_hi,
        r_up=rho_half,
    )
    truth_nb = float(np.sum(pdf_nb * phi * dA))

    z_stack_kw = dict(
        T_c=T_c,
        q_c=q_c,
        H=H,
        nz=nz,
        dT_dn=dT_dn,
        dq_dn=dq_dn,
        dT_up=dT_up,
        dq_up=dq_up,
        s_t2_c=s_t2_c,
        s_q2_c=s_q2_c,
        rho_c=rho_c,
        sT_dn=sT_dn,
        sT_up=sT_up,
        sQ_dn=sQ_dn,
        sQ_up=sQ_up,
    )

    ns = list(range(2, 17))
    julia_vals = [julia_profile_expectation(n, f, par=par, cov0=cov0, H_layer=H, mu_t=T_c, mu_q=q_c) for n in ns]
    nested_vals = [slab_z_stack_nested_gh_expectation(n, f, **z_stack_kw) for n in ns]
    note_vals: list[float] = []
    for n in ns:
        Tn, qn, Wn = notebook_split_p_quadrature(
            n,
            T_c,
            q_c,
            dT_dn=dT_dn,
            dq_dn=dq_dn,
            sT_dn=s_t_lo,
            sq_dn=s_q_lo,
            r_dn=rho_half,
            dT_up=dT_up,
            dq_up=dq_up,
            sT_up=s_t_hi,
            sq_up=s_q_hi,
            r_up=rho_half,
        )
        note_vals.append(float(np.sum(Wn * f(Tn, qn))))

    # Nodes for display at N_show
    N_show = 3
    T_j, q_j, w_j, tag = collect_tq_cloud_julia_style(
        n=N_show,
        par=par,
        cov0=cov0,
        H_layer=H,
        mu_t=T_c,
        mu_q=q_c,
        mean_gradients=("dn", "up"),
    )
    w_plot = np.where(tag < 2, 0.5 * w_j, 0.5 * w_j)

    z_centers_lin = (np.arange(nz, dtype=float) + 0.5) / nz * H - 0.5 * H
    z_ref = float(z_centers_lin[max(1, nz // 10)])
    slice_kw = {k: v for k, v in z_stack_kw.items() if k != "nz"}
    dT_gh, dq_gh, w_gh = slab_z_stack_slice_gh_fluctuation_points(N_show, z_ref, **slice_kw)

    T_nb, q_nb, W_nb = notebook_split_p_quadrature(
        N_show,
        T_c,
        q_c,
        dT_dn=dT_dn,
        dq_dn=dq_dn,
        sT_dn=s_t_lo,
        sq_dn=s_q_lo,
        r_dn=rho_half,
        dT_up=dT_up,
        dq_up=dq_up,
        sT_up=s_t_hi,
        sq_up=s_q_hi,
        r_up=rho_half,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2))

    # ``pdf_z[j,i]`` at ``(dT[i], dq[j])`` (``meshgrid(..., indexing='xy')``).
    # **Do not** transpose — ``.T`` swaps T and q axes and misaligns the heatmap from contours.
    im = ax1.imshow(
        np.asarray(pdf_z, dtype=float),
        origin="lower",
        extent=[dT[0], dT[-1], dq[0], dq[-1]],
        aspect="auto",
        cmap="Blues",
        interpolation="nearest",
        zorder=1,
    )
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="pdf [(kg/kg·K)⁻¹]")

    # Exact μ(ζ) locus (piecewise linear; same face offsets as black segments).
    z_track = np.linspace(-0.5 * H, 0.5 * H, 600)
    dT_mu = np.where(z_track <= 0.0, dT_dn * (-2.0 * z_track / H), dT_up * (2.0 * z_track / H))
    dq_mu = np.where(z_track <= 0.0, dq_dn * (-2.0 * z_track / H), dq_up * (2.0 * z_track / H))
    ax1.plot(
        dT_mu,
        dq_mu,
        color="cyan",
        lw=1.6,
        alpha=0.9,
        zorder=3,
        label=r"$\bar\mu(\zeta)$ locus",
    )

    # Faint leg-aligned ridges (notebook ``eval_pdf``) — whitened vs raw mean line.
    DTg, DQg = np.meshgrid(dT, dq, indexing="xy")
    try:
        ax1.contour(
            DTg,
            DQg,
            pdf_nb,
            levels=8,
            colors="w",
            linewidths=0.45,
            alpha=0.55,
            linestyles="--",
            zorder=2,
        )
    except ValueError:
        pass

    # Supersaturated region q > q_sat(T) (Clausius–Clapeyron), in fluctuation coords.
    dT_cc = np.linspace(dT[0], dT[-1], max(256, n_grid))
    dq_sat_cc = q_sat_kgkg(T_c + dT_cc) - q_c
    y_top = float(dq[-1])
    ax1.fill_between(
        dT_cc,
        dq_sat_cc,
        y_top,
        where=(y_top > dq_sat_cc),
        interpolate=True,
        color="0.35",
        alpha=0.18,
        zorder=2,
        label=r"$q > q_{\mathrm{sat}}(T)$ (CC / condensation)",
    )

    ax1.plot(
        [0.0, dT_dn],
        [0.0, dq_dn],
        "k-",
        lw=2.5,
        zorder=3,
        label="raw mean segment (slopes in physical T,q)",
    )
    ax1.plot(
        [0.0, dT_up],
        [0.0, dq_up],
        "k-",
        lw=2.5,
        zorder=3,
        label="raw mean segment (upper)",
    )
    # Tags match ``collect_tq_cloud_julia_style``: 0/2 = u− on [-L_dn,0]; 1/3 = u+ on [0,L_up];
    # tags 0,1 = first ``mean_gradient`` entry; 2,3 = second (below- vs above-center slopes).
    tag_labels = (
        "tag0: u−, mg[0]",
        "tag1: u+, mg[0]",
        "tag2: u−, mg[1]",
        "tag3: u+, mg[1]",
    )
    # Distinct from split-p markers (avoid tab10 ``C1`` red–orange vs notebook red).
    colors = ("#0173B2", "#029E73", "#DE8F05", "#CC78BC")
    w_scale = (w_plot / (w_plot.max() + 1e-15)) * 120 + 15
    for tid in range(4):
        m = tag == tid
        if not np.any(m):
            continue
        ax1.scatter(
            T_j[m] - T_c,
            q_j[m] - q_c,
            s=w_scale[m],
            c=colors[tid],
            alpha=0.55,
            edgecolors="none",
            linewidths=0.0,
            zorder=5,
            label=f"Julia {tag_labels[tid]} (N={N_show})",
        )
    ax1.scatter(
        T_nb - T_c,
        q_nb - q_c,
        s=(W_nb / (W_nb.max() + 1e-15)) * 95 + 12,
        marker="D",
        facecolors=(0.55, 0.12, 0.65, 0.35),
        edgecolors="#3B1C5A",
        linewidths=0.9,
        zorder=5,
        label=rf"Split-$p$ / Halley (N={N_show}; slab-$\Sigma$ stds)",
    )
    ax1.scatter(
        dT_gh,
        dq_gh,
        s=(w_gh / (w_gh.max() + 1e-15)) * 70 + 10,
        marker="P",
        facecolors=(0.1, 0.55, 0.2, 0.35),
        edgecolors="#0d3d14",
        linewidths=0.65,
        zorder=4,
        label=rf"Nested GH @ $\zeta$={z_ref:.1f} m (N={N_show}²)",
    )
    ax1.set_xlabel("T − T_center (K)")
    ax1.set_ylabel("q − q_center (kg/kg)")
    ax1.set_title(
        f"$p(T,q)$: slab $\\Sigma$ + $H/4$ variance walk (Julia-matched), Nz={nz}, grid {n_grid}²\n"
        f"+ CC + μ(ζ) + Halley $p_{{nb}}$ contours (white)"
    )
    ax1.legend(loc="upper left", fontsize=6.5, framealpha=0.92)

    ax2.axhline(
        truth_z,
        color="black",
        ls="--",
        lw=2,
        label=rf"$\int f\,p$ layer marginal (slab $\Sigma$, $H/4$ + $\zeta$-Riemann) = {truth_z:.6g}",
    )
    ax2.plot(ns, julia_vals, "bo-", ms=6, label="Julia-style (½ dn + ½ up pass)")
    ax2.plot(
        ns,
        nested_vals,
        "s-",
        color="#1a6b2e",
        ms=5,
        lw=1.8,
        label=r"Nested $\zeta$-Riemann × 2D GH (cross-check)",
    )
    ax2.plot(
        ns,
        note_vals,
        "D--",
        color="#5c2d91",
        ms=4,
        lw=1.4,
        label=r"Split-$p$ + $p_{\mathrm{Halley}}$ (slab-$\Sigma$ stds = black law)",
    )
    ax2.set_xlabel("Nodes per dimension N")
    ax2.set_ylabel("∫ saturation_excess × p  dT dq  [kg/kg]")
    ax2.set_title(
        r"Black = ζ-Riemann marginal; purple = analytic uniform-ζ + split-$p$; green = nested GH check"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=7)
    fig.tight_layout()
    outp = _HERE / "profile_cell_brute_pdf_quadrature_convergence.png"
    fig.savefig(outp, dpi=150)
    print(f"wrote {outp}")
    print(f"  ∫f·p  layer marginal (slab Σ, H/4, ζ-Riemann) = {truth_z:.8e}")
    print(
        f"  ∫f·p_Halley  same grid (eval_pdf w/ slab half-stds; should ≈ black as Nz→∞) = {truth_nb:.8e}"
    )
    print(f"  (Halley grid ∫ − ζ-Riemann ∫) / ζ-Riemann ∫ = {(truth_nb - truth_z) / truth_z:+.4e}")
    for n, jv, nv, sp in zip(ns, julia_vals, nested_vals, note_vals):
        print(
            f"  N={n:2d}  julia={jv:.6e}  nested_GH={nv:.6e}  split-p={sp:.6e}  "
            f"err_julia={(jv - truth_z) / truth_z:+.3e}  err_nested={(nv - truth_z) / truth_z:+.3e}  "
            f"err_splitp={(sp - truth_z) / truth_z:+.3e}  err_splitp_vs_Halley_grid={(sp - truth_nb) / truth_nb:+.3e}"
        )


if __name__ == "__main__":
    main()
