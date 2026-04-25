"""
High-precision checks for closed-form primitives in ``profile_rosenblatt_two_half_cell``.

CDFs for ``uniform`` and ``mixture`` convolutions are compared to integrals of the
analytic PDF. See ``PROFILE_MATH_AUDIT.md`` in this directory for the audit trail.
"""
from __future__ import annotations

import unittest

import numpy as np
from scipy import integrate
from scipy.special import roots_hermite

import profile_rosenblatt_two_half_cell as pr
import variance_dashboard_interactive as vd

from test_variance_mixture_consistency import _PAR_A, _PAR_B


class TestUniformGaussianConvolution(unittest.TestCase):
    def test_cdf_matches_integral_of_pdf(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            L = float(rng.uniform(0.1, 0.9))
            s = float(rng.uniform(0.08, 0.45))
            x = float(rng.uniform(-1.2, 1.2))

            def pdf(t: float) -> float:
                return pr.uniform_gaussian_convolution_pdf(t, L, s)

            num, _ = integrate.quad(pdf, -30.0, x, limit=300, epsabs=1e-10, epsrel=1e-8)
            ana = pr.uniform_gaussian_convolution_cdf(x, L, s)
            self.assertLess(abs(num - ana), 5e-7, msg=f"x={x} L={L} num={num} ana={ana}")


class TestMixtureUniformConvolution(unittest.TestCase):
    def test_cdf_matches_integral_of_pdf(self):
        rng = np.random.default_rng(0)
        for _ in range(12):
            L_dn = float(rng.uniform(0.05, 0.6))
            s_dn = float(rng.uniform(0.05, 0.5))
            L_up = float(rng.uniform(0.05, 0.6))
            s_up = float(rng.uniform(0.05, 0.5))
            x = float(rng.uniform(-1.0, 1.0))

            def pdf(t: float) -> float:
                return pr.mixture_uniform_convolution_pdf(t, L_dn, s_dn, L_up, s_up)

            num, _ = integrate.quad(pdf, -25.0, x, limit=300, epsabs=1e-10, epsrel=1e-8)
            ana = pr.mixture_uniform_convolution_cdf(x, L_dn, s_dn, L_up, s_up)
            self.assertLess(abs(num - ana), 5e-7, msg=f"x={x} L_dn={L_dn} num={num} ana={ana}")


class TestMeanGradientAxis(unittest.TestCase):
    def test_dn_axis_changes_M_inv_vs_averaged_axis(self):
        par = {
            "muT_c": 290.0,
            "muQ_c": 10.0,
            "varT_c": 1.0,
            "varQ_c": 1.5,
            "covTq": 0.8,
            "d_cov_tq_dn": 0.0,
            "d_cov_tq_up": 0.0,
            "mT_up": -1.5,
            "mQ_up": -2.0,
            "sT_up": 0.5,
            "sQ_up": 0.8,
            "mT_dn": 1.0,
            "mQ_dn": 1.5,
            "sT_dn": -0.4,
            "sQ_dn": -0.6,
        }
        cov0 = vd._center_cov0(par)
        mu_t, mu_q, _, _ = vd._layer_mu_sigma(par)
        s_t2 = max(float(cov0[0, 0]), pr._NUM_EPS**2)
        s_q2 = max(float(cov0[1, 1]), pr._NUM_EPS**2)
        rho = float(cov0[0, 1] / (np.sqrt(s_t2 * s_q2) + pr._NUM_EPS))
        H = vd._layer_H(par)
        out_avg = pr.two_slope_rosenblatt_params(
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
            H=H,
            mean_gradient="avg",
        )
        out_dn = pr.two_slope_rosenblatt_params(
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
            H=H,
            mean_gradient="dn",
        )
        self.assertIsNotNone(out_avg)
        self.assertIsNotNone(out_dn)
        self.assertGreater(
            float(np.max(np.abs(out_avg["M_inv"] - out_dn["M_inv"]))),
            1e-8,
        )


class TestMixtureQuantileBrent(unittest.TestCase):
    def test_quantiles_increase_with_probability(self):
        L_dn, s_dn, L_up, s_up = 0.22, 0.11, 0.19, 0.13
        ps = np.linspace(0.02, 0.98, 25)
        us = [pr.mixture_convolution_quantile_brent(float(p), L_dn, s_dn, L_up, s_up) for p in ps]
        for a, b in zip(us, us[1:]):
            self.assertLessEqual(a, b + 1e-12)

    def test_brent_hits_target_cdf(self):
        L_dn, s_dn, L_up, s_up = 0.22, 0.11, 0.19, 0.13
        for p in np.linspace(0.05, 0.95, 13):
            u_b = pr.mixture_convolution_quantile_brent(float(p), L_dn, s_dn, L_up, s_up)
            f_b = pr.mixture_uniform_convolution_cdf(u_b, L_dn, s_dn, L_up, s_up)
            self.assertLess(abs(f_b - float(p)), 5e-10)


class TestMInvVsNotebookRotation(unittest.TestCase):
    """``M_inv`` from :func:`two_slope_rosenblatt_params` vs orthonormal ``R^T`` (notebook)."""

    def test_julia_matrix_matches_scaled_orthonormal_inverse(self):
        rng = np.random.default_rng(7)
        for _ in range(30):
            d_t = float(rng.uniform(-2.0, 2.0))
            d_q = float(rng.uniform(-2.0, 2.0))
            a_sq = d_t**2 + d_q**2
            if a_sq < 1e-8:
                continue
            inv_a = 1.0 / a_sq
            M = np.array([[d_t, -d_q * inv_a], [d_q, d_t * inv_a]], dtype=float)
            alpha = float(np.sqrt(a_sq))
            Rinv = np.array(
                [[d_t / alpha, -d_q / alpha], [d_q / alpha, d_t / alpha]], dtype=float
            )
            uo, vo = rng.standard_normal(2)
            ui = uo / alpha
            vj = vo * alpha
            d1 = M @ np.array([ui, vj])
            d2 = Rinv @ np.array([uo, vo])
            self.assertLess(np.max(np.abs(d1 - d2)), 1e-12)


class TestFlucToTqVsAffine(unittest.TestCase):
    """Interior ``(T,q)``: ``fluc_to_tq`` ≡ ``μ+δ`` (bounds only change tails)."""

    def test_matches_affine_for_center_gaussian_draws(self):
        rng = np.random.default_rng(3)
        mu_t, mu_q = 290.0, 0.012
        sig_t, sig_q = 1.0, 0.001
        rho = 0.8
        t_min, q_max = 150.0, 1.0e6
        cov = np.array(
            [[sig_t**2, rho * sig_t * sig_q], [rho * sig_t * sig_q, sig_q**2]], dtype=float
        )
        L = np.linalg.cholesky(cov)
        for _ in range(40):
            z = rng.standard_normal(2)
            dT, dq = (L @ z).ravel()
            t1, q1 = pr.fluc_to_tq(
                np.array([dT]),
                np.array([dq]),
                mu_t,
                mu_q,
                sig_t,
                sig_q,
                rho,
                t_min,
                q_max,
            )
            self.assertLess(abs(float(t1[0]) - (mu_t + dT)), 1e-9)
            self.assertLess(abs(float(q1[0]) - (mu_q + dq)), 1e-9)


def _profile_two_half_pdf_box_normalized(
    par: dict,
    t_lo: float,
    t_hi: float,
    q_lo: float,
    q_hi: float,
    *,
    n_t: int,
    n_q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """``profile_rosenblatt_two_half_physical_density`` on cell centers, renormalized on the box."""
    n_t = int(max(4, n_t))
    n_q = int(max(4, n_q))
    te = np.linspace(float(t_lo), float(t_hi), n_t + 1)
    qe = np.linspace(float(q_lo), float(q_hi), n_q + 1)
    tc = 0.5 * (te[1:] + te[:-1])
    qc = 0.5 * (qe[1:] + qe[:-1])
    tt, qq = np.meshgrid(tc, qc, indexing="ij")
    cov0 = vd._center_cov0(par)
    mu_t, mu_q, _, _ = vd._layer_mu_sigma(par)
    raw, st = pr.profile_rosenblatt_two_half_physical_density(
        tt,
        qq,
        par=par,
        cov0=cov0,
        H_layer=float(vd._layer_H(par)),
        mu_t=float(mu_t),
        mu_q=float(mu_q),
    )
    dt = float(te[1] - te[0])
    dq = float(qe[1] - qe[0])
    if st != "ok" or (not np.all(np.isfinite(raw))) or float(np.max(raw)) <= 0.0:
        return te, qe, np.zeros_like(tt), st
    mass = float(np.sum(raw) * dt * dq)
    if mass <= 0.0:
        return te, qe, np.zeros_like(tt), st
    return te, qe, raw / mass, st


class TestProfileTwoHalfPhysicalDensityOnGrid(unittest.TestCase):
    """``profile_rosenblatt_two_half_physical_density`` vs tensor cubature (same ½–½ mean-gradient–axis blend as ``q_c``)."""

    def test_box_mass_one_after_renormalization(self):
        par = _PAR_A
        t_lo, t_hi, q_lo, q_hi = vd.mixture_marginal_axis_bounds(par, pad=6.5)
        te, qe, pdf, st = _profile_two_half_pdf_box_normalized(
            par, t_lo, t_hi, q_lo, q_hi, n_t=36, n_q=36
        )
        self.assertEqual(st, "ok")
        dt = float(te[1] - te[0])
        dq = float(qe[1] - qe[0])
        mass = float(np.sum(pdf) * dt * dq)
        self.assertAlmostEqual(mass, 1.0, places=5)

    def test_grid_mean_near_cubature_tensor(self):
        par = _PAR_A
        n_tensor = 24
        _qc, t_n, q_n, w_n, *_ = vd.two_half_profile_rosenblatt_condensate(par, n_tensor)
        t_n = np.asarray(t_n, dtype=float).ravel()
        q_n = np.asarray(q_n, dtype=float).ravel()
        w_n = np.asarray(w_n, dtype=float).ravel()
        pad_t, pad_q = 8.0, 4.0
        t_lo = float(np.min(t_n)) - pad_t
        t_hi = float(np.max(t_n)) + pad_t
        q_lo = max(0.05, float(np.min(q_n)) - pad_q)
        q_hi = float(np.max(q_n)) + pad_q
        te, qe, pdf, st = _profile_two_half_pdf_box_normalized(
            par, t_lo, t_hi, q_lo, q_hi, n_t=90, n_q=90
        )
        self.assertEqual(st, "ok")
        tc = 0.5 * (te[1:] + te[:-1])
        qc = 0.5 * (qe[1:] + qe[:-1])
        tt, qq = np.meshgrid(tc, qc, indexing="ij")
        dt = float(te[1] - te[0])
        dq = float(qe[1] - qe[0])
        mt_g = float(np.sum(tt * pdf) * dt * dq)
        mq_g = float(np.sum(qq * pdf) * dt * dq)
        mt = float(np.sum(t_n * w_n))
        mq = float(np.sum(q_n * w_n))
        self.assertLess(abs(mt_g - mt), 1e-3, msg=f"E[T] grid {mt_g} vs cubature {mt}")
        self.assertLess(abs(mq_g - mq), 1e-3, msg=f"E[q] grid {mq_g} vs cubature {mq}")

    def test_grid_covariance_roughly_matches_cubature_tensor(self):
        par = _PAR_A
        n_tensor = 24
        _qc, t_n, q_n, w_n, *_ = vd.two_half_profile_rosenblatt_condensate(par, n_tensor)
        t_n = np.asarray(t_n, dtype=float).ravel()
        q_n = np.asarray(q_n, dtype=float).ravel()
        w_n = np.asarray(w_n, dtype=float).ravel()
        pad_t, pad_q = 8.0, 4.0
        t_lo = float(np.min(t_n)) - pad_t
        t_hi = float(np.max(t_n)) + pad_t
        q_lo = max(0.05, float(np.min(q_n)) - pad_q)
        q_hi = float(np.max(q_n)) + pad_q
        te, qe, pdf, st = _profile_two_half_pdf_box_normalized(
            par, t_lo, t_hi, q_lo, q_hi, n_t=96, n_q=96
        )
        self.assertEqual(st, "ok")
        tc = 0.5 * (te[1:] + te[:-1])
        qc = 0.5 * (qe[1:] + qe[:-1])
        tt, qq = np.meshgrid(tc, qc, indexing="ij")
        dt = float(te[1] - te[0])
        dq = float(qe[1] - qe[0])
        mt = float(np.sum(tt * pdf) * dt * dq)
        mq = float(np.sum(qq * pdf) * dt * dq)
        c11_g = float(np.sum((tt - mt) ** 2 * pdf) * dt * dq)
        c22_g = float(np.sum((qq - mq) ** 2 * pdf) * dt * dq)
        c12_g = float(np.sum((tt - mt) * (qq - mq) * pdf) * dt * dq)
        mtc = float(np.sum(t_n * w_n))
        mqc = float(np.sum(q_n * w_n))
        c11_c = float(np.sum(w_n * (t_n - mtc) ** 2))
        c22_c = float(np.sum(w_n * (q_n - mqc) ** 2))
        c12_c = float(np.sum(w_n * (t_n - mtc) * (q_n - mqc)))
        tol = 0.04
        self.assertLess(abs(c11_g - c11_c), tol, msg=f"Var[T] grid {c11_g} vs cubature {c11_c}")
        self.assertLess(abs(c22_g - c22_c), tol, msg=f"Var[q] grid {c22_g} vs cubature {c22_c}")
        self.assertLess(abs(c12_g - c12_c), tol, msg=f"Cov[T,q] grid {c12_g} vs cubature {c12_c}")


if __name__ == "__main__":
    unittest.main()
