#!/usr/bin/env python3
"""
Scatter **(T, q)** nodes for **Julia-style** profile–Rosenblatt (split-``p`` inner rule):

  * **Split-``p`` (same mean-gradient pass):** GL abscissa ``p_i`` on ``[0,1]``; if both
    half-legs are on, ``p_i < 0.5`` → lower half only (remap ``2p``), ``p_i > 0.5`` → upper
    (``2p-1``), ``p_i \\approx 0.5`` → bridge sample (cell-center ``\\sigma_{T,q}``, ``u = \\mu_0``).
    **One** ``(T,q)`` per ``(j,i)`` → ``N^2`` nodes per pass (same idea as
    ``notebook_split_p_quadrature`` in ``profile_cell_brute_pdf_quadrature_convergence.py``).

  * **Two mean-gradient passes:** when DN and UP axes are both valid, Julia still averages
    ``0.5*(\\mathrm{dn} + \\mathrm{up})`` → **``2 N^2``** samples for plotting.

This script mirrors ``subgrid_layer_profile_quadrature.jl`` for visualization only.

Run from repo root (or adjust ``sys.path``):

    python calibration/experiments/variance_adjustments/local_python/plot_profile_rosenblatt_tq_nodes_julia_style.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import roots_hermite

# Reuse helpers from the port module (same directory).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from profile_rosenblatt_two_half_cell import (  # noqa: E402
    _NUM_EPS,
    _ug_cdf_shifted,
    fluc_to_tq,
    two_slope_rosenblatt_params,
)


def _gl01(n: int) -> tuple[np.ndarray, np.ndarray]:
    t, w = leggauss(n)
    return (t + 1.0) * 0.5, w * 0.5


def _half_leg_u_quantile_brent(
    p: float, L: float, s: float, a: float, b: float
) -> float:
    """``u`` with ``F_leg(u) = p`` for shifted uniform⊛Gaussian on ``[a,b]`` (``b-a = L``)."""
    p = float(np.clip(p, _NUM_EPS, 1.0 - _NUM_EPS))
    L = max(float(L), _NUM_EPS)
    s = max(float(s), _NUM_EPS)
    smax = max(s, L)
    lo = float(a) - 6.0 * smax
    hi = float(b) + 6.0 * smax

    def g(x: float) -> float:
        return float(_ug_cdf_shifted(x, a, b, s)) - p

    return float(brentq(g, lo, hi))


def collect_tq_cloud_julia_style(
    *,
    n: int,
    par: dict,
    cov0: np.ndarray,
    H_layer: float,
    mu_t: float,
    mu_q: float,
    mean_gradients: tuple[str, ...] = ("dn", "up"),
    t_min: float = 150.0,
    q_max: float = 1.0e6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns ``(T, q, w, tag)`` where ``tag`` is integer 0..3 for plotting:
    0/2 = inner u- mixture (uniform on ``[-L_dn, 0]`` in Rosenblatt ``u``); 1/3 = u+ on ``[0, L_up]``.
    Even/odd index = first vs second ``mean_gradient`` entry (below- vs above-center axis slopes).
    """
    s_t2 = max(float(cov0[0, 0]), _NUM_EPS**2)
    s_q2 = max(float(cov0[1, 1]), _NUM_EPS**2)
    c12 = float(cov0[0, 1])
    rho = c12 / (np.sqrt(s_t2 * s_q2) + _NUM_EPS)
    dz_q = 0.25 * H_layer
    s_t2_fdn = max(s_t2 - dz_q * par["sT_dn"], 0.0)
    s_t2_fup = max(s_t2 + dz_q * par["sT_up"], 0.0)
    s_q2_fdn = max(s_q2 - dz_q * par["sQ_dn"], 0.0)
    s_q2_fup = max(s_q2 + dz_q * par["sQ_up"], 0.0)
    sig_t_fdn = float(np.sqrt(s_t2_fdn))
    sig_q_fdn = float(np.sqrt(s_q2_fdn))
    sig_t_fup = float(np.sqrt(s_t2_fup))
    sig_q_fup = float(np.sqrt(s_q2_fup))

    p_nodes, p_w = _gl01(n)
    h_x, h_w = roots_hermite(n)
    w_outer = h_w / np.sqrt(np.pi)
    sqrt2 = np.sqrt(2.0)

    T_list: list[float] = []
    q_list: list[float] = []
    w_list: list[float] = []
    tag_list: list[int] = []

    for mg_i, mg in enumerate(mean_gradients):
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
            mean_gradient=mg,
        )
        if out is None:
            continue
        M = out["M_inv"]
        s_v_fdn = out["s_v_fdn"]
        s_v_fup = out["s_v_fup"]
        L_dn, L_up = out["L_dn"], out["L_up"]
        r_fdn, r_fup = out["r_fdn"], out["r_fup"]
        s_udn, s_uup = out["s_u_dn"], out["s_u_up"]
        eps = _NUM_EPS
        use_dn = L_dn > eps
        use_up = L_up > eps

        for j in range(n):
            chi_j = float(h_x[j])
            wj = float(w_outer[j])
            vj_fdn = sqrt2 * s_v_fdn * chi_j
            vj_fup = sqrt2 * s_v_fup * chi_j
            mu0_fdn = r_fdn * vj_fdn
            mu0_fup = r_fup * vj_fup
            for i in range(n):
                pi_ = float(p_nodes[i])
                wi_ = float(p_w[i])
                if use_dn and use_up:
                    mid, tol = 0.5, 64.0 * eps
                    s_v_deg = max(s_v_fdn, s_v_fup, eps)
                    vj_deg = sqrt2 * s_v_deg * chi_j
                    mu0_deg = float(out["r_c"]) * vj_deg
                    if abs(pi_ - mid) <= tol:
                        hw = mu0_deg
                        vj, mu0, sigt, sigq = vj_deg, mu0_deg, float(np.sqrt(s_t2)), float(
                            np.sqrt(s_q2)
                        )
                        leg_tag = 2
                    elif pi_ < mid:
                        p_leg = 2.0 * pi_
                        hw = (
                            _half_leg_u_quantile_brent(p_leg, L_dn, s_udn, -L_dn, 0.0)
                            + mu0_fdn
                        )
                        vj, mu0, sigt, sigq, leg_tag = (
                            vj_fdn,
                            mu0_fdn,
                            sig_t_fdn,
                            sig_q_fdn,
                            0,
                        )
                    else:
                        p_leg = 2.0 * pi_ - 1.0
                        hw = (
                            _half_leg_u_quantile_brent(p_leg, L_up, s_uup, 0.0, L_up)
                            + mu0_fup
                        )
                        vj, mu0, sigt, sigq, leg_tag = (
                            vj_fup,
                            mu0_fup,
                            sig_t_fup,
                            sig_q_fup,
                            1,
                        )
                    dT = M[0, 0] * (hw - mu0) + M[0, 1] * vj
                    dq = M[1, 0] * (hw - mu0) + M[1, 1] * vj
                    t_hat, q_hat = fluc_to_tq(
                        np.array([dT]),
                        np.array([dq]),
                        mu_t,
                        mu_q,
                        sigt,
                        sigq,
                        rho,
                        t_min,
                        q_max,
                    )
                    T_list.append(float(t_hat[0]))
                    q_list.append(float(q_hat[0]))
                    w_list.append(wj * wi_)
                    tag_list.append(2 * mg_i + leg_tag)
                elif use_dn:
                    hw = _half_leg_u_quantile_brent(pi_, L_dn, s_udn, -L_dn, 0.0)
                    dT = M[0, 0] * hw + M[0, 1] * vj_fdn
                    dq = M[1, 0] * hw + M[1, 1] * vj_fdn
                    t_hat, q_hat = fluc_to_tq(
                        np.array([dT]),
                        np.array([dq]),
                        mu_t,
                        mu_q,
                        sig_t_fdn,
                        sig_q_fdn,
                        rho,
                        t_min,
                        q_max,
                    )
                    T_list.append(float(t_hat[0]))
                    q_list.append(float(q_hat[0]))
                    w_list.append(wj * wi_)
                    tag_list.append(2 * mg_i + 0)
                elif use_up:
                    hw = _half_leg_u_quantile_brent(pi_, L_up, s_uup, 0.0, L_up)
                    dT = M[0, 0] * hw + M[0, 1] * vj_fup
                    dq = M[1, 0] * hw + M[1, 1] * vj_fup
                    t_hat, q_hat = fluc_to_tq(
                        np.array([dT]),
                        np.array([dq]),
                        mu_t,
                        mu_q,
                        sig_t_fup,
                        sig_q_fup,
                        rho,
                        t_min,
                        q_max,
                    )
                    T_list.append(float(t_hat[0]))
                    q_list.append(float(q_hat[0]))
                    w_list.append(wj * wi_)
                    tag_list.append(2 * mg_i + 1)

    return (
        np.asarray(T_list, float),
        np.asarray(q_list, float),
        np.asarray(w_list, float),
        np.asarray(tag_list, int),
    )


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plotting. "
            "Install with e.g. `pip install matplotlib` in your env."
        ) from e

    n = 5
    H_layer = 400.0
    mu_t, mu_q = 285.0, 0.012
    cov0 = np.array([[0.25, 0.0], [0.0, 1e-6]], dtype=float)
    par = {
        "mT_dn": 0.0,
        "mT_up": 0.0,
        "mQ_dn": 1e-3,
        "mQ_up": 1.2e-3,
        "sT_dn": 0.0,
        "sT_up": 0.0,
        "sQ_dn": 1e-9,
        "sQ_up": 2e-9,
    }

    T, q, w, tag = collect_tq_cloud_julia_style(
        n=n,
        par=par,
        cov0=cov0,
        H_layer=H_layer,
        mu_t=mu_t,
        mu_q=mu_q,
        mean_gradients=("dn", "up"),
    )
    print(f"N = {n}, collected {T.size} (T,q) samples (max 4 N^2 = {4 * n * n})")

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    colors = np.array(["C0", "C1", "C2", "C3"])
    c = colors[np.clip(tag, 0, 3)]
    ax.scatter(q, T, s=18, c=c, alpha=0.65, edgecolors="none")
    ax.set_xlabel("q (kg/kg)")
    ax.set_ylabel("T (K)")
    # "dn/up leg" here is **inner first-coordinate u**: uniform on [-L_dn,0] vs [0,L_up]
    # in the *rotated* (T,q) frame — not "use only lower/upper *physical* half slopes" for that leg.
    # "axis below/above center" is mean_gradient_axis: which *piecewise* mean slope (dT,dq)/dz
    # defines the u direction and M_inv for that full cubature pass (both physical half-slopes
    # still enter L_dn, L_up, σ at ±H/4, etc. inside two_slope_rosenblatt_params).
    ax.set_title(
        "Profile–Rosenblatt (Julia-style)\n"
        "C0=u− mixture / axis=below-center slope; C1=u+ mixture / axis=below-center; "
        "C2=u− / axis=above-center; C3=u+ / axis=above-center"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    outp = _HERE / "profile_rosenblatt_tq_cloud_julia_style.png"
    fig.savefig(outp, dpi=150)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
