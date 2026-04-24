"""
Regression tests for the variance-adjustments **toy** stack.

**Definitions vs code:** see ``ORACLE_SGS.md``. **Ground-truth** ``q_c`` = two-half profile–Rosen
(:func:`truth_condensate_gpkg`). MVN mixture paths are **diagnostic** (different integral than that
profile cell rule; see ``ORACLE_SGS.md``).

- **Face-anchored σ²(z)** in the mixture PDF matches the column half-cell σ² reconstruction
  at faces. Off-diagonals use `cov_mixture_half` (center cov plus optional per-half linear drift).

- **Column-tensor GH** (:func:`mixture_condensate_nested_gh`) uses **full** tensor GH
  at each ``ζ`` (uniform stack on ``[-L½,L½]``); converges to the fine-grid reference as ``N`` grows.
  The **budget** ``N^2`` χ₁-partition rule is :func:`mixture_condensate_nested_gh_budget` / middle scatter only (generally **not** identical to full tensor).

- **Two-half profile–Rosen** (:func:`two_half_profile_rosenblatt_condensate`) — ``2 N^2`` cubature when both DN/UP halves succeed;
  at ``N`` = :data:`TRUTH_PROFILE_ORDER` this **is** :func:`truth_condensate_gpkg`.

- **Dashboard** — **Black** + **blue**: :func:`truth_condensate_gpkg` and :func:`convergence_two_half_profile_rosenblatt_N`
  (profile ``N``-sweep). **Middle:** **purple** = layer-mean column-tensor ``p(T,q)``
  (:func:`mixture_marginal_uniform_z_on_grid`, GL in ``ζ``);
  **GH** ``N^2`` at cell center from :func:`gauss_hermite_nodes_bivariate` + :func:`column_tensor_mu_cov_zeta``;
  **black** polyline = :func:`column_mean_Tq`.
  **Mixture** GH: :func:`convergence_mixture_tensor_gh_uniform_z` vs :func:`mixture_truth_fine_grid_gpkg`.

Run (needs SciPy; use your named conda env, e.g. WeatherQuest):

    conda run -n WeatherQuest python test_variance_mixture_consistency.py

or

    conda run -n WeatherQuest python -m pytest test_variance_mixture_consistency.py
"""
from __future__ import annotations

import unittest

import numpy as np

import variance_dashboard_interactive as vd
from variance_dashboard_interactive import H_LAYER, L_HALF, MU_T0, MU_Q0, VAR_T0, VAR_Q0
from profile_rosenblatt_two_half_cell import two_slope_rosenblatt_params

# Fixed toy layer (typical dashboard defaults)
_PAR_A = {
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

_PAR_B = {
    "muT_c": 290.0,
    "muQ_c": 10.0,
    "varT_c": 1.0,
    "varQ_c": 1.5,
    "covTq": 0.0,
    "d_cov_tq_dn": 0.0,
    "d_cov_tq_up": 0.0,
    "mT_up": 0.2,
    "mQ_up": 0.1,
    "sT_up": 0.1,
    "sQ_up": 0.2,
    "mT_dn": 0.3,
    "mQ_dn": 0.1,
    "sT_dn": 0.15,
    "sQ_dn": 0.05,
}


def _face_sig2_julia(
    par: dict,
) -> tuple[float, float, float, float]:
    """Mirror `subgrid_layer_profile_quadrature` σ² at each face (T and q)."""
    half = 0.5 * H_LAYER
    sT2_c = float(par.get("varT_c", VAR_T0))
    sq2_c = float(par.get("varQ_c", VAR_Q0))
    sT2_fdn = max(sT2_c - half * par["sT_dn"], 0.0)
    sT2_fup = max(sT2_c + half * par["sT_up"], 0.0)
    sq2_fdn = max(sq2_c - half * par["sQ_dn"], 0.0)
    sq2_fup = max(sq2_c + half * par["sQ_up"], 0.0)
    return sT2_fdn, sT2_fup, sq2_fdn, sq2_fup


class TestColumnMeanPath(unittest.TestCase):
    def test_column_mean_Tq_matches_signed_zeta_slopes(self):
        """DN face ζ=-L uses m_dn; UP face ζ=+L uses m_up (same as column_tensor_mu_cov_zeta mean)."""
        par = _PAR_A
        lh = vd._layer_L_half(par)
        mu_dn = vd.column_mean_Tq(-lh, par)
        mu_ct = vd.column_mean_Tq(0.0, par)
        mu_up = vd.column_mean_Tq(lh, par)
        mt_c, mq_c, _, _ = vd._layer_mu_sigma(par)
        np.testing.assert_allclose(mu_ct, [mt_c, mq_c], rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            mu_dn,
            [mt_c - lh * par["mT_dn"], mq_c - lh * par["mQ_dn"]],
            rtol=0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            mu_up,
            [mt_c + lh * par["mT_up"], mq_c + lh * par["mQ_up"]],
            rtol=0,
            atol=1e-9,
        )


class TestLayerMarginalMoments(unittest.TestCase):
    def test_layer_marginal_mean_Tq_matches_zeta_average_of_column_mean(self):
        """Under uniform ζ, E[T]=μ_c+L(m_up-m_dn)/4 (piecewise μ); discrete ∫∫ T·p dT dq must match."""
        par = _PAR_A
        lh = vd._layer_L_half(par)
        mt_c, mq_c, _, _ = vd._layer_mu_sigma(par)
        exp_t = mt_c + 0.25 * lh * (par["mT_up"] - par["mT_dn"])
        exp_q = mq_c + 0.25 * lh * (par["mQ_up"] - par["mQ_dn"])
        t_lo, t_hi, q_lo, q_hi = vd.mixture_marginal_axis_bounds(par, pad=5.5)
        te, qe, pg = vd.mixture_marginal_uniform_z_on_grid(
            par, t_lo, t_hi, q_lo, q_hi, n_t=120, n_q=120, nz=160
        )
        tc = 0.5 * (te[1:] + te[:-1])
        qc = 0.5 * (qe[1:] + qe[:-1])
        tt, qq = np.meshgrid(tc, qc, indexing="ij")
        dt = float(te[1] - te[0])
        dq = float(qe[1] - qe[0])
        et = float(np.sum(pg * tt) * dt * dq)
        eq = float(np.sum(pg * qq) * dt * dq)
        self.assertAlmostEqual(et, exp_t, delta=0.04)
        self.assertAlmostEqual(eq, exp_q, delta=0.04)


class TestMixtureFaceAnchoredHalfcell(unittest.TestCase):
    def test_face_variances_match_z_equals_L_half(self):
        for par in (_PAR_A, _PAR_B):
            sT2_fdn, sT2_fup, sq2_fdn, sq2_fup = _face_sig2_julia(par)
            cdn = vd.cov_mixture_half(L_HALF, par, "dn")
            cup = vd.cov_mixture_half(L_HALF, par, "up")
            self.assertAlmostEqual(cdn[0, 0], sT2_fdn, places=6)
            self.assertAlmostEqual(cup[0, 0], sT2_fup, places=6)
            self.assertAlmostEqual(cdn[1, 1], sq2_fdn, places=6)
            self.assertAlmostEqual(cup[1, 1], sq2_fup, places=6)

    def test_center_covariance_both_halves(self):
        cov0 = 0.75
        par = {
            "muT_c": 290.0,
            "muQ_c": 10.0,
            "varT_c": float(VAR_T0),
            "varQ_c": float(VAR_Q0),
            "covTq": cov0,
            "d_cov_tq_dn": 0.0,
            "d_cov_tq_up": 0.0,
            "sT_dn": 0.3,
            "sQ_dn": 0.2,
            "sT_up": 0.3,
            "sQ_up": 0.2,
        }
        for half in ("dn", "up"):
            c = vd.cov_mixture_half(0.0, par, half)
            self.assertAlmostEqual(c[0, 0], float(VAR_T0), places=10)
            self.assertAlmostEqual(c[1, 1], float(VAR_Q0), places=10)
            self.assertAlmostEqual(c[0, 1], cov0, places=6)

    def test_cov_slope_changes_off_diagonal(self):
        par = dict(_PAR_A)
        par["d_cov_tq_dn"] = 0.1
        z = 0.25
        c = vd.cov_mixture_half(z, par, "dn")
        cmax = 0.999 * float(np.sqrt(c[0, 0] * c[1, 1]))
        expect = float(np.clip(par["covTq"] + 0.1 * z, -cmax, cmax))
        self.assertAlmostEqual(c[0, 1], expect, places=6)


class TestTruthAndProfileInternal(unittest.TestCase):
    def test_gauss_hermite_chi1_slice_full_range_matches_bivariate(self):
        """χ₁-slice with full ``j`` range must match :func:`gauss_hermite_nodes_bivariate` exactly."""
        par = _PAR_A
        C0 = vd._center_cov0(par)
        mu = np.array([par["muT_c"], par["muQ_c"]])
        n = 5
        t0, q0, w0 = vd.gauss_hermite_nodes_bivariate(n, mu, C0)
        t1, q1, w1 = vd.gauss_hermite_nodes_bivariate_chi1_slice(n, mu, C0, 0, n - 1)
        self.assertEqual(t0.size, n * n)
        np.testing.assert_allclose(t0, t1, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(q0, q1, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(w0, w1, rtol=0.0, atol=1e-14)

    def test_nested_gh_tracks_fine_grid_asymmetric_halves(self):
        """Nested GH and fine-grid truth both integrate the same mixture law."""
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
            "mT_dn": -3.25,
            "mQ_dn": 4.0,
            "sT_dn": -0.4,
            "sQ_dn": -0.6,
        }
        truth = vd.mixture_truth_fine_grid_gpkg(par, 180, 180, nz_pdf=48)
        nest = vd.mixture_condensate_nested_gh(par, 24, 48)
        scale = max(abs(truth), 1e-12)
        self.assertLess(
            abs(truth - nest) / scale,
            0.06,
            msg=f"nested GH should track fine-grid truth; truth={truth}, nest={nest}",
        )

    def test_oracle_scalar_is_profile_at_truth_order(self):
        par = _PAR_A
        oracle = vd.truth_condensate_gpkg(par, 0, 0, 0)
        pr, *_ = vd.two_half_profile_rosenblatt_condensate(par, int(vd.TRUTH_PROFILE_ORDER))
        self.assertAlmostEqual(float(oracle), float(pr), places=10)

    def test_two_half_profile_emits_two_halves_node_count(self):
        par = _PAR_A
        qc, t, q, w, u, v, use_l, st = vd.two_half_profile_rosenblatt_condensate(par, 7)
        if st == "ok":
            self.assertEqual(t.size, 2 * 7 * 7)
            self.assertEqual(q.size, 2 * 7 * 7)
            self.assertEqual(w.size, 2 * 7 * 7)
        self.assertGreaterEqual(qc, 0.0)

    def test_two_half_tensor_halves_weighted_means_split(self):
        n_tensor = 2
        nh = n_tensor**2  # 4 nodes per half → 8 total for st == "ok"
        ft = np.array([0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 15.0, 15.0], dtype=float)
        fq = np.zeros(8, dtype=float)
        fw = np.ones(8, dtype=float) * (1.0 / 8.0)
        m_dn, m_up = vd._two_half_tensor_halves_weighted_means(ft, fq, fw, n_tensor, "ok")
        assert m_dn is not None and m_up is not None
        self.assertAlmostEqual(m_dn[0], 5.0)
        self.assertAlmostEqual(m_up[0], 10.0)

    def test_convergence_mixture_tensor_gh_curve(self):
        par = _PAR_A
        ns, ser = vd.convergence_mixture_tensor_gh_uniform_z(par, n_max=8)
        ref = vd.mixture_condensate_nested_gh(par, 8, int(vd.TRUTH_Z_SAMPLES))
        self.assertAlmostEqual(float(ser[-1]), ref, places=12)
        self.assertEqual(len(ns), 7)

    def test_convergence_two_half_profile_Nsweep_idempotent(self):
        par = _PAR_A
        ns_a, ser_a = vd.convergence_two_half_profile_rosenblatt_N(par, n_max=8)
        ns_b, ser_b = vd.convergence_two_half_profile_rosenblatt_N(par, n_max=8)
        np.testing.assert_allclose(ns_a, ns_b)
        np.testing.assert_allclose(ser_a, ser_b)

    def test_convergence_two_half_curve_matches_condensate_calls(self):
        par = _PAR_A
        ns, ser = vd.convergence_two_half_profile_rosenblatt_N(par, n_max=8)
        self.assertEqual(len(ns), 7)
        for n, q in zip(ns.astype(int), ser, strict=True):
            ref, *_ = vd.two_half_profile_rosenblatt_condensate(par, int(n))
            self.assertAlmostEqual(float(q), float(ref), places=12)

    def test_mixture_glz_tracks_uniform_z_reference(self):
        """GL-z mixture ref should sit near uniform-z GH when z resolution is generous."""
        par = _PAR_A
        n_tq = 10
        u_hi = vd.mixture_condensate_nested_gh(par, n_tq, 200)
        glz = vd.mixture_condensate_nested_gh_glz(par, n_tq, int(vd.Z_GL_ORDER))
        scale = max(abs(u_hi), 1e-12)
        self.assertLess(
            abs(u_hi - glz) / scale,
            0.04,
            msg=f"GL-z vs fine uniform-z mixture GH: {glz} vs {u_hi}",
        )

    def test_nested_node_bundle_matches_scalar_condensate(self):
        par = _PAR_A
        qc_a = vd.mixture_condensate_nested_gh(par, 5, 16)
        qc_b, *_ = vd.mixture_nested_gh_node_bundle(par, 5, 16)
        self.assertAlmostEqual(qc_a, qc_b, places=10)

    def test_nested_gh_near_truth_default_par(self):
        """Full-tensor nested GH vs fine-grid oracle: use **relative** error when |oracle| is O(1)."""
        par = _PAR_A
        nz = int(vd.TRUTH_Z_SAMPLES)
        truth = vd.mixture_truth_fine_grid_gpkg(par, 200, 200, nz_pdf=nz)
        nest = vd.mixture_condensate_nested_gh(par, 20, nz)
        err = abs(truth - nest)
        # Relative error is ill-defined when the oracle is ~0; then bound absolute error only.
        if abs(truth) >= 1e-5:
            self.assertLess(
                err / abs(truth),
                0.14,
                msg=f"full-tensor nested vs fine grid (relative): truth={truth}, nest={nest}",
            )
        else:
            self.assertLess(
                err,
                5e-4,
                msg=f"full-tensor nested vs fine grid (|truth| tiny, abs err): truth={truth}, nest={nest}",
            )

    def test_budget_nested_gh_differs_from_full_at_moderate_n(self):
        """χ₁-budget path (odd N, three strata with ⅓ mass) is not the full GH tensor estimator."""
        par = _PAR_A
        nz = 32
        # Odd tensor order → three χ₁ strata; renormalized ⅓ weights do not reproduce full 2D GH.
        n_odd = 7
        full = vd.mixture_condensate_nested_gh(par, n_odd, nz)
        bud = vd.mixture_condensate_nested_gh_budget(par, n_odd, nz)
        scale = max(abs(full), 1e-12)
        self.assertGreater(
            abs(full - bud) / scale,
            0.02,
            msg=f"budget should not match full tensor: full={full}, bud={bud}",
        )

    def test_truth_refines_with_grid(self):
        par = _PAR_A
        t_lo = vd.mixture_truth_fine_grid_gpkg(par, nt=100, nq=100, nz_pdf=48)
        t_hi = vd.mixture_truth_fine_grid_gpkg(par, nt=200, nq=200, nz_pdf=96)
        scale = max(abs(t_hi), 1e-12)
        self.assertLess(
            abs(t_lo - t_hi) / scale,
            0.02,
            msg=f"truth should refine: 100² vs 200² gave {t_lo} vs {t_hi}",
        )

    def test_truth_z_average_coarsening_moves_integral(self):
        """Mixture truth integrates a z-average; too few z samples bias the scalar."""
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
            "mT_dn": -3.25,
            "mQ_dn": 4.0,
            "sT_dn": -0.4,
            "sQ_dn": -0.6,
        }
        fine_z = vd.mixture_truth_fine_grid_gpkg(par, 140, 140, nz_pdf=96)
        coarse_z = vd.mixture_truth_fine_grid_gpkg(par, 140, 140, nz_pdf=6)
        scale = max(abs(fine_z), 1e-12)
        self.assertGreater(
            abs(fine_z - coarse_z) / scale,
            0.005,
            msg=f"z discretization should matter: fine_z={fine_z}, coarse_z={coarse_z}",
        )

    def test_dashboard_mixture_path_uses_high_internal_z_samples(self):
        self.assertGreaterEqual(int(vd.TRUTH_Z_SAMPLES), 64)

    def test_truth_profile_order_is_consistent(self):
        self.assertGreaterEqual(int(vd.TRUTH_PROFILE_ORDER), 32)

    def test_profile_stable_in_n(self):
        """Two-half profile cubature: N=32 vs N=48 should be nearly settled for toy par."""
        par = _PAR_A
        q32, *_ = vd.two_half_profile_rosenblatt_condensate(par, 32)
        q48, *_ = vd.two_half_profile_rosenblatt_condensate(par, 48)
        scale = max(abs(q48), 1e-12)
        self.assertLess(
            abs(q32 - q48) / scale,
            0.02,
            msg=f"profile N=32 vs 48: {q32} vs {q48}",
        )

    def test_two_branches_at_z_is_exactly_n2_partitioned(self):
        """N² budget scatter: even N → two χ₁ bands; odd N → two equal wings + middle stratum."""
        par = _PAR_A
        for n in (3, 4, 5):
            t, q, w, half, zk = vd.mixture_quadrature_two_branches_at_z(par, n, z_frac=0.5)
            self.assertEqual(t.size, n * n)
            self.assertAlmostEqual(float(np.sum(w)), 1.0, places=12)
            self.assertTrue(np.all(w >= 0.0))
            self.assertGreater(zk, 0.0)
            nw = (n - 1) // 2
            if n % 2 == 0:
                n1 = n // 2
                self.assertEqual(int(np.sum(half == 0)), n1 * n)
                self.assertEqual(int(np.sum(half == 1)), n1 * n)
            else:
                self.assertEqual(int(np.sum(half == 0)), nw * n)
                self.assertEqual(int(np.sum(half == 1)), n)
                self.assertEqual(int(np.sum(half == 2)), nw * n)

    def test_mixture_marginal_uniform_z_on_grid_renormalizes_to_one(self):
        """Discrete ∫∫ pdf dT dq on the box should be 1 after renormalization (column MVN+ζ)."""
        par = _PAR_A
        t_lo, t_hi, q_lo, q_hi = vd.mixture_marginal_axis_bounds(par, pad=6.5)
        te, qe, pdf = vd.mixture_marginal_uniform_z_on_grid(
            par, t_lo, t_hi, q_lo, q_hi, n_t=36, n_q=36, nz=48
        )
        dt = float(te[1] - te[0])
        dq = float(qe[1] - qe[0])
        mass = float(np.sum(pdf) * dt * dq)
        self.assertAlmostEqual(mass, 1.0, places=5)

    def test_column_slab_marginal_glz_near_uniform_z_refinement(self):
        """Gauss–Legendre ζ-average vs fine uniform ζ-Riemann: same law, similar grid moments."""
        par = _PAR_A
        t_lo, t_hi, q_lo, q_hi = vd.mixture_marginal_axis_bounds(par, pad=6.5)
        te_gl, qe_gl, pdf_gl = vd.mixture_marginal_uniform_z_on_grid(
            par,
            t_lo,
            t_hi,
            q_lo,
            q_hi,
            n_t=48,
            n_q=48,
            z_quadrature="gauss_legendre",
            n_z_gl=int(vd.Z_GL_ORDER),
        )
        te_u, qe_u, pdf_u = vd.mixture_marginal_uniform_z_on_grid(
            par, t_lo, t_hi, q_lo, q_hi, n_t=48, n_q=48, nz=400, z_quadrature="uniform"
        )
        tc = 0.5 * (te_gl[1:] + te_gl[:-1])
        qc = 0.5 * (qe_gl[1:] + qe_gl[:-1])
        tt, qq = np.meshgrid(tc, qc, indexing="ij")
        dt = float(te_gl[1] - te_gl[0])
        dq = float(qe_gl[1] - qe_gl[0])
        mt_gl = float(np.sum(pdf_gl * tt) * dt * dq)
        mq_gl = float(np.sum(pdf_gl * qq) * dt * dq)
        mt_u = float(np.sum(pdf_u * tt) * dt * dq)
        mq_u = float(np.sum(pdf_u * qq) * dt * dq)
        self.assertLess(abs(mt_gl - mt_u), 0.08, msg=f"E[T] gl={mt_gl} uni={mt_u}")
        self.assertLess(abs(mq_gl - mq_u), 0.08, msg=f"E[q] gl={mq_gl} uni={mq_u}")

    def test_mixture_pdf_grid_mass_near_one_on_axis_box(self):
        """Analytic column MVN+ζ marginal should integrate to ~1 on mixture_marginal_axis_bounds (Riemann)."""
        par = _PAR_A
        t_lo, t_hi, q_lo, q_hi = vd.mixture_marginal_axis_bounds(par, pad=6.5)
        nt, nq = 100, 100
        te = np.linspace(t_lo, t_hi, nt + 1)
        qe = np.linspace(q_lo, q_hi, nq + 1)
        tc = 0.5 * (te[1:] + te[:-1])
        qc = 0.5 * (qe[1:] + qe[:-1])
        tt, qq = np.meshgrid(tc, qc, indexing="ij")
        pdf = vd.mixture_pdf_grid(tt, qq, par, nz=64)
        dt = float(te[1] - te[0])
        dq = float(qe[1] - qe[0])
        mass = float(np.sum(pdf) * dt * dq)
        self.assertGreater(mass, 0.85, msg=f"integrated p_mix={mass}")
        self.assertLessEqual(mass, 1.0 + 1e-6)

    def test_two_slope_params_nondegenerate(self):
        """Profile path must use same H and center moments as the dashboard module."""
        C0 = vd._center_cov0(_PAR_A)
        out = two_slope_rosenblatt_params(
            mu_t=_PAR_A["muT_c"],
            mu_q=_PAR_A["muQ_c"],
            sig_t2_c=float(C0[0, 0]),
            sig_q2_c=float(C0[1, 1]),
            rho_tq=float(C0[0, 1] / np.sqrt(C0[0, 0] * C0[1, 1])),
            m_t_dn=_PAR_A["mT_dn"],
            m_t_up=_PAR_A["mT_up"],
            m_q_dn=_PAR_A["mQ_dn"],
            m_q_up=_PAR_A["mQ_up"],
            s_t_dn=_PAR_A["sT_dn"],
            s_t_up=_PAR_A["sT_up"],
            s_q_dn=_PAR_A["sQ_dn"],
            s_q_up=_PAR_A["sQ_up"],
            H=H_LAYER,
        )
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
