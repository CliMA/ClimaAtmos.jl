r"""
Interactive two-half-cell (T,q) dashboard for variance-adjustments exploration.

**Left panel** — ground-truth ``q_c`` (black line) from :func:`truth_condensate_gpkg`: the
two-half profile–Rosen condensate at tensor order :data:`TRUTH_PROFILE_ORDER`.  Blue curve
sweeps the same rule vs N and must converge to black.

**Middle panel** — layer-mean column-tensor ``p(T,q)`` (purple):

    p_cell(T,q) = (1/2L) ∫_{-L}^{L} N(μ(ζ), Σ(ζ)) dζ

evaluated by Gauss–Legendre in ζ via :func:`mixture_marginal_uniform_z_on_grid`.
``μ(ζ)`` and ``Σ(ζ)`` come from :func:`column_tensor_mu_cov_zeta` (piecewise-linear
DN/UP slopes).  This is the natural reference PDF for the column model; it is **not**
the profile–Rosen ``p(T,q)`` used for ``q_c``.  See
``docs/layer_mean_cell_Tq_marginal_derivation.md`` for derivation status.

Blue markers: N×N Gauss–Hermite at cell center (ζ=0) from :func:`gauss_hermite_nodes_bivariate`.
Black polyline: mean (T,q) at depth ζ ∈ {-L, 0, +L} from :func:`column_mean_Tq`.

Units: T [K], q [g/kg].  Run from ``Variance_Stuff.ipynb`` in this directory.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from profile_rosenblatt_two_half_cell import profile_rosenblatt_cubature_two_halves_cell
from scipy.special import roots_hermite, roots_legendre
from scipy.stats import multivariate_normal as mvn

# ---------------------------------------------------------------------------
# Physical constants (cell-center reference)
# ---------------------------------------------------------------------------
L_HALF = 0.5  # half-cell thickness (arbitrary length unit consistent with slopes)
H_LAYER = (
    2.0 * L_HALF
)  # full column thickness (centered-gradient layer; same H as profile cubature)


def _layer_L_half(par: dict) -> float:
    """Distance center → each face along the column (must match ``z`` range in mixture integral)."""
    return float(np.clip(float(par.get("L_half", L_HALF)), 1e-6, 1e3))


def _layer_H(par: dict) -> float:
    return 2.0 * _layer_L_half(par)


# ``z`` samples for :func:`mixture_truth_fine_grid_gpkg` / :func:`mixture_pdf_grid` (MVN-mixture path).
TRUTH_Z_SAMPLES = 128
# Tensor order for the **condensate oracle** (:func:`truth_condensate_gpkg`): two-half profile–Rosen.
TRUTH_PROFILE_ORDER = 96
# Gauss–Legendre order for :func:`mixture_condensate_nested_gh_glz` (tests / ORACLE_SGS cross-check).
Z_GL_ORDER = 16
MU_T0 = 290.0
MU_Q0 = 10.0  # g/kg
VAR_T0 = 1.0
VAR_Q0 = 1.5


def _layer_mu_sigma(par: dict) -> tuple[float, float, float, float]:
    """Cell-center (μ_T, μ_q) and center variances (σ²_T, σ²_q) from ``par`` with module defaults."""
    mu_t = float(par.get("muT_c", MU_T0))
    mu_q = float(par.get("muQ_c", MU_Q0))
    v_t = max(float(par.get("varT_c", VAR_T0)), 1e-12)
    v_q = max(float(par.get("varQ_c", VAR_Q0)), 1e-12)
    return mu_t, mu_q, v_t, v_q


def q_sat_gpkg(T_k: np.ndarray) -> np.ndarray:
    """Saturation specific humidity [g/kg] — same Clausius–Clapeyron form as `Variance_Stuff.ipynb`."""
    return (
        6.112 * np.exp(17.67 * (T_k - 273.15) / (T_k - 29.65)) * 0.622 / 1000.0 * 1000.0
    )


def _center_cov_matrix(par: dict) -> np.ndarray:
    """
    2×2 turbulent covariance at **cell center**: ``par['varT_c']``, ``par['varQ_c']``,
    ``covTq`` (clipped). Used for center-Σ GH diagnostics and as ``C0`` fed into
    ``profile_rosenblatt_cubature`` (outer Gaussian axis). Independent of the mixture’s
    per-half ``d_cov_tq_*`` (those only affect the mixture PDF / fine-grid truth).
    """
    _, _, vt, vq = _layer_mu_sigma(par)
    c = float(
        np.clip(par["covTq"], -0.999 * np.sqrt(vt * vq), 0.999 * np.sqrt(vt * vq))
    )
    return np.array([[vt, c], [c, vq]], dtype=float)


def cov_mixture_half(z: float, par: dict, half: str) -> np.ndarray:
    """
    Bivariate turbulent Σ(z) on one **half** for the **mixture** PDF (fine-grid truth).

    Diagonals: same face-anchored linear ``σ²_k(z)`` as elsewhere in this file (DN vs UP
    use ``sT_dn`` / ``sT_up`` etc.).

    Off-diagonal (turbulent ``cov_Tq``): **linear in z** from the cell center toward
    that half’s face,

        ``cov_Tq(z) = covTq(0) + d_cov_tq_* · z``,

    with **separate** slopes ``par['d_cov_tq_dn']`` and ``par['d_cov_tq_up']``. Values
    are clipped so the 2×2 matrix stays symmetric PSD for the given ``σ²_T(z)``,
    ``σ²_q(z)``. Defaults ``d_cov_tq_* = 0`` give **constant** ``cov_Tq`` along each
    half (implicitly varying correlation if ``σ²`` changes); non-zero slopes let you
    explore closures that are **not** locked to ``ρ·√(σ²_T σ²_q)`` with fixed ρ.

    ``half`` is ``"dn"`` or ``"up"``; ``z`` is distance from center toward that face.
    """
    z = max(float(z), 0.0)
    half = half.lower()
    cov0 = float(par["covTq"])
    _, _, v_tc, v_qc = _layer_mu_sigma(par)
    if half == "dn":
        s_t, s_q = float(par["sT_dn"]), float(par["sQ_dn"])
        k = float(par.get("d_cov_tq_dn", 0.0))
        v_t = max(v_tc - s_t * z, 1e-8)
        v_q = max(v_qc - s_q * z, 1e-8)
    elif half == "up":
        s_t, s_q = float(par["sT_up"]), float(par["sQ_up"])
        k = float(par.get("d_cov_tq_up", 0.0))
        v_t = max(v_tc + s_t * z, 1e-8)
        v_q = max(v_qc + s_q * z, 1e-8)
    else:
        raise ValueError(f"half must be 'dn' or 'up', got {half!r}")
    c_raw = cov0 + k * z
    cmax = 0.999 * float(np.sqrt(v_t * v_q))
    c = float(np.clip(c_raw, -cmax, cmax))
    return np.array([[v_t, c], [c, v_q]], dtype=float)


def _pick_half_slope(zeta: float, s_dn: float, s_up: float) -> float:
    """Same branch as Julia ``_pick_half_slope``: DN slopes for ``ζ < 0``, UP for ``ζ ≥ 0``."""
    return float(s_up) if zeta >= 0.0 else float(s_dn)


def column_tensor_mu_cov_zeta(zeta: float, par: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(μ(ζ), Σ(ζ))`` aligned with Julia ``SubgridColumnTensor`` in
    ``integrate_over_sgs`` (long-arity): piecewise slopes from cell center, diagonal
    ``σ_k^2(ζ) = σ_{k,c}^2 + ζ · ∂σ_k^2/∂z`` with the DN vs UP gradient chosen by ``sign(ζ)``,
    and **fixed** correlation coefficient from cell center (``covTq / √(σ_{T,c}^2 σ_{q,c}^2)``)
    mapped to ``ρ · σ_T(ζ) σ_q(ζ)`` on the off-diagonal — matching ``PhysicalPointTransform``
    with evolving marginals and constant ``ρ``.
    """
    mu_t, mu_q, v_tc, v_qc = _layer_mu_sigma(par)
    zeta = float(zeta)
    sm_t = _pick_half_slope(zeta, float(par["mT_dn"]), float(par["mT_up"]))
    sm_q = _pick_half_slope(zeta, float(par["mQ_dn"]), float(par["mQ_up"]))
    svt = _pick_half_slope(zeta, float(par["sT_dn"]), float(par["sT_up"]))
    svq = _pick_half_slope(zeta, float(par["sQ_dn"]), float(par["sQ_up"]))
    mt = mu_t + zeta * sm_t
    mq = mu_q + zeta * sm_q
    v_t = max(v_tc + zeta * svt, 1e-12)
    v_q = max(v_qc + zeta * svq, 1e-12)
    rho_c = float(par["covTq"]) / (np.sqrt(v_tc * v_qc) + 1e-18)
    rho_c = float(np.clip(rho_c, -0.9999, 0.9999))
    st, sq = np.sqrt(v_t), np.sqrt(v_q)
    c12 = rho_c * st * sq
    cov = np.array([[v_t, c12], [c12, v_q]], dtype=float)
    mu = np.array([mt, mq], dtype=float)
    return mu, cov


def column_mean_Tq(zeta: float, par: dict) -> np.ndarray:
    """Column mean ``(μ_T, μ_q)`` at signed offset ``ζ`` from center — same as :func:`column_tensor_mu_cov_zeta` mean."""
    mu, _ = column_tensor_mu_cov_zeta(float(zeta), par)
    return mu


def par_from_ipywidgets(wd: dict) -> dict:
    """Build physics dict from `ipywidgets` value attributes (not matplotlib Slider)."""
    return {
        "muT_c": float(wd["muT_c"].value),
        "muQ_c": float(wd["muQ_c"].value),
        "varT_c": float(wd["varT_c"].value),
        "varQ_c": float(wd["varQ_c"].value),
        "covTq": float(wd["cov"].value),
        "d_cov_tq_dn": float(wd["d_cov_dn"].value),
        "d_cov_tq_up": float(wd["d_cov_up"].value),
        "mT_up": float(wd["mT_up"].value),
        "mQ_up": float(wd["mQ_up"].value),
        "sT_up": float(wd["sT_up"].value),
        "sQ_up": float(wd["sQ_up"].value),
        "mT_dn": float(wd["mT_dn"].value),
        "mQ_dn": float(wd["mQ_dn"].value),
        "sT_dn": float(wd["sT_dn"].value),
        "sQ_dn": float(wd["sQ_dn"].value),
        "L_half": float(wd["L_half"].value),
    }


def _physics_key(par: dict) -> tuple[float, ...]:
    """Hashable key for (cov + slopes) — N, nz, truth grid sizes not included."""
    return (
        float(par["muT_c"]),
        float(par["muQ_c"]),
        float(par["varT_c"]),
        float(par["varQ_c"]),
        float(par["covTq"]),
        float(par["d_cov_tq_dn"]),
        float(par["d_cov_tq_up"]),
        float(par["mT_up"]),
        float(par["mQ_up"]),
        float(par["sT_up"]),
        float(par["sQ_up"]),
        float(par["mT_dn"]),
        float(par["mQ_dn"]),
        float(par["sT_dn"]),
        float(par["sQ_dn"]),
        _layer_L_half(par),
    )


def mixture_pdf_grid(
    T: np.ndarray,
    Q: np.ndarray,
    par: dict,
    nz: int,
) -> np.ndarray:
    """
    Layer-averaged **column-tensor** MVN density in ``(T,q)``: at each ``ζ ∈ [-L_{1/2}, L_{1/2}]``
    (signed distance from **cell center**), one bivariate Normal
    ``N(μ(ζ), Σ(ζ))`` from :func:`column_tensor_mu_cov_zeta`, matching Julia
    ``SubgridColumnTensor`` / ``_pick_half_slope`` (DN branch for ``ζ<0``, UP for ``ζ≥0``).

    Returns the Riemann mean ``(1/n_z) \\sum_k p(T,Q|ζ_k)`` for ``ζ_k`` uniform on the full layer
    (same scaling convention as :func:`mixture_condensate_nested_gh`).

    **``nz``:** number of uniform samples on ``[-L_{1/2}, L_{1/2}]``.
    """
    lh = _layer_L_half(par)
    zetas = np.linspace(-lh, lh, int(nz))
    pos = np.dstack((T, Q))
    acc = np.zeros_like(T, dtype=float)
    for zeta in zetas:
        mu, cov = column_tensor_mu_cov_zeta(float(zeta), par)
        acc += mvn(mu, cov, allow_singular=True).pdf(pos)
    return acc / float(nz)


def mixture_pdf_grid_glz(
    T: np.ndarray,
    Q: np.ndarray,
    par: dict,
    *,
    n_z_gl: int,
) -> np.ndarray:
    """
    Same **column-tensor** layer-mean integrand as :func:`mixture_pdf_grid`, but the outer
    ``ζ``-average on ``[-L_{1/2}, L_{1/2}]`` uses **Gauss–Legendre** on ``[-1, 1]`` (same scaling
    as :func:`mixture_condensate_nested_gh_glz`):

    ``(1/(2L)) \\int_{-L}^{L} p(T,Q\\mid\\zeta)\\,d\\zeta
      = \\frac{1}{2}\\int_{-1}^{1} p(T,Q\\mid Lx)\\,dx
      \\approx \\frac{1}{2}\\sum_k w_k\\,p(T,Q\\mid \\zeta_k)``,

    with ``\\zeta_k = L_{1/2} x_k``. At each ``\\zeta_k``, ``p(\\cdot\\mid\\zeta_k)`` is the **exact**
    bivariate Normal PDF from :func:`column_tensor_mu_cov_zeta` — not histograms of ``(T,q)``
    samples and not profile condensate quadrature.
    """
    lh = _layer_L_half(par)
    nz_gl = int(max(2, n_z_gl))
    x, wx = roots_legendre(nz_gl)
    pos = np.dstack((T, Q))
    acc = np.zeros_like(T, dtype=float)
    for k in range(nz_gl):
        zeta = float(lh) * float(x[k])
        mu, cov = column_tensor_mu_cov_zeta(zeta, par)
        acc += 0.5 * float(wx[k]) * mvn(mu, cov, allow_singular=True).pdf(pos)
    return acc


# Uniform-ζ MVN slab: Riemann resolution in ζ and (T,q) for diagnostics / tests.
MVN_UNIFORM_Z_SLICES = 1000
MVN_TQ_GRID_RES = 200
# Middle panel: purple = layer-mean column-tensor p(T,q) (GL in ζ).
PURPLE_PCOLORMESH_ALPHA = 0.55


def mixture_marginal_uniform_z_on_grid(
    par: dict,
    t_lo: float,
    t_hi: float,
    q_lo: float,
    q_hi: float,
    *,
    n_t: int = MVN_TQ_GRID_RES,
    n_q: int = MVN_TQ_GRID_RES,
    nz: int = MVN_UNIFORM_Z_SLICES,
    z_quadrature: str = "uniform",
    n_z_gl: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Column-tensor **layer-mean** ``p(T,q)`` on a uniform ``(T,q)`` cell grid.

    At each height offset ``ζ`` from the cell center, the conditional law is the **exact**
    ``N(μ(ζ), Σ(ζ))`` PDF (SciPy ``multivariate_normal.pdf``). The layer mean is

    ``(1/(2L)) ∫_{-L}^{L} p(T,q \\mid ζ)\\,dζ``.

    - ``z_quadrature="uniform"`` (default): Riemann mean via :func:`mixture_pdf_grid` with ``nz``
      equally spaced ``ζ`` (same convention as :func:`mixture_condensate_nested_gh`).
    - ``z_quadrature="gauss_legendre"``: the same integral via Gauss–Legendre on ``[-1,1]`` at
      ``ζ = L_{1/2} x_k`` with weights ``w_k`` (same outer rule as :func:`mixture_condensate_nested_gh_glz`);
      ``n_z_gl`` defaults to :data:`Z_GL_ORDER`. This is **not** condensate ``(T,q)`` sampling; it is
      quadrature of the **ζ** axis only, with a closed-form Gaussian at each node.

    Then **box-normalize** so ``∑ pdf·ΔT·Δq = 1`` on the window.

    Returns
    -------
    t_edges, q_edges, pdf_norm
        Edge arrays (length ``n_t+1``, ``n_q+1``) and cell-centered density ``pdf_norm`` with
        shape ``(n_t, n_q)`` (``indexing="ij"``: row ``i`` = ``T``, col ``j`` = ``q``). For
        ``matplotlib.pyplot.pcolormesh(t_edges, q_edges, …)`` with 1D edges, pass ``pdf_norm.T``
        so array rows match ``q`` and columns match ``T``.
    """
    n_t = int(max(4, n_t))
    n_q = int(max(4, n_q))
    te = np.linspace(float(t_lo), float(t_hi), n_t + 1)
    qe = np.linspace(float(q_lo), float(q_hi), n_q + 1)
    tc = 0.5 * (te[1:] + te[:-1])
    qc = 0.5 * (qe[1:] + qe[:-1])
    tt, qq = np.meshgrid(tc, qc, indexing="ij")
    zq = str(z_quadrature).lower().strip()
    if zq == "uniform":
        nz_u = int(max(2, nz))
        raw = mixture_pdf_grid(tt, qq, par, nz=nz_u)
    elif zq in ("gauss_legendre", "gl", "glz"):
        nz_gl = int(Z_GL_ORDER if n_z_gl is None else max(2, int(n_z_gl)))
        raw = mixture_pdf_grid_glz(tt, qq, par, n_z_gl=nz_gl)
    else:
        raise ValueError(
            "z_quadrature must be 'uniform' or 'gauss_legendre' (aliases: gl, glz), "
            f"got {z_quadrature!r}"
        )
    dt = float(te[1] - te[0])
    dq = float(qe[1] - qe[0])
    mass = float(np.sum(raw) * dt * dq)
    pdf_norm = raw / max(mass, 1e-300)
    return te, qe, pdf_norm


def mixture_marginal_axis_bounds(
    par: dict,
    *,
    pad: float = 6.5,
    n_z_scan: int = 40,
) -> tuple[float, float, float, float]:
    """
    Conservative axis-aligned ``(T,q)`` box for the **mixture** marginal.

    Scanning ``ζ ∈ [-L_{1/2}, L_{1/2}]`` with :func:`column_tensor_mu_cov_zeta`, take each local mean
    ``± pad · √(diag Σ(ζ))``. Using only ``μ ± pad·σ_center`` misses displaced
    mass when mean slopes and face variances are strong; truncating the domain
    and re-normalizing ``pdf`` then **blows up** ``q_c`` (spurious huge condensate).
    """
    lh = _layer_L_half(par)
    mu_t, mu_q, _, _ = _layer_mu_sigma(par)
    t_lo, t_hi = np.inf, -np.inf
    q_lo, q_hi = np.inf, -np.inf
    for zeta in np.linspace(-lh, lh, int(max(4, n_z_scan))):
        mu, c = column_tensor_mu_cov_zeta(float(zeta), par)
        mt, mq = float(mu[0]), float(mu[1])
        st = float(np.sqrt(max(c[0, 0], 1e-14)))
        sq = float(np.sqrt(max(c[1, 1], 1e-14)))
        t_lo = min(t_lo, mt - pad * st)
        t_hi = max(t_hi, mt + pad * st)
        q_lo = min(q_lo, mq - pad * sq)
        q_hi = max(q_hi, mq + pad * sq)
    tm = 0.5 * (t_lo + t_hi)
    qs_mid = float(q_sat_gpkg(np.array([tm]))[0])
    q_hi = max(q_hi, qs_mid + 5.0, mu_q + 1e-3)
    q_lo = max(0.05, min(q_lo, qs_mid - 3.0))
    return float(t_lo), float(t_hi), float(q_lo), float(q_hi)


def mixture_truth_fine_grid_gpkg(
    par: dict,
    nt: int,
    nq: int,
    nz_pdf: int,
    *,
    tq_bounds: tuple[float, float, float, float] | None = None,
    mixture_pad: float = 6.5,
) -> float:
    """
    Fine-grid ``∫∫ max(q-q_sat,0) · p_{mix}(T,q) \\,dT\\,dQ`` [g/kg] for the **uniform-``z`` MVN
    mixture** (:func:`mixture_pdf_grid`). Cross-check for :func:`mixture_condensate_nested_gh`; not
    the two-half profile condensate oracle.
    """
    if tq_bounds is None:
        t_min, t_max, q_min, q_max = mixture_marginal_axis_bounds(par, pad=mixture_pad)
    else:
        t_min, t_max, q_min, q_max = tq_bounds
    te = np.linspace(t_min, t_max, nt + 1)
    qe = np.linspace(q_min, q_max, nq + 1)
    tc = 0.5 * (te[1:] + te[:-1])
    qc = 0.5 * (qe[1:] + qe[:-1])
    d_t = te[1] - te[0]
    d_q = qe[1] - qe[0]
    tt, qq = np.meshgrid(tc, qc, indexing="ij")
    pdf = mixture_pdf_grid(tt, qq, par, nz=nz_pdf)
    s = float(np.sum(pdf) * d_t * d_q)
    if s <= 0.0:
        return 0.0
    p = pdf / s
    qs = q_sat_gpkg(tt)
    return float(np.sum(np.maximum(qq - qs, 0.0) * p) * d_t * d_q)


def truth_condensate_gpkg(
    par: dict,
    # nt, nq, nz_pdf accepted but ignored — kept for test call-site compatibility only.
    # Tests call truth_condensate_gpkg(par, 0, 0, 0); those args are meaningless here.
    *_ignored,
    profile_order: int | None = None,
    **_ignored_kw,
) -> float:
    """
    Ground-truth condensate [g/kg]: ``E[max(q-q_sat(T),0)]`` from **two-half**
    profile–Rosen cubature at tensor order ``profile_order`` (default
    :data:`TRUTH_PROFILE_ORDER`).

    This is the single authoritative ``q_c`` scalar for this stack.
    For the column-tensor MVN condensate diagnostic use :func:`mixture_truth_fine_grid_gpkg`.
    """
    n = int(TRUTH_PROFILE_ORDER if profile_order is None else profile_order)
    n = max(2, min(160, n))
    qc, *_rest = two_half_profile_rosenblatt_condensate(par, n)
    return float(qc)


def gauss_hermite_nodes(n: int, par: dict):
    """
    2D tensor GH in (T, q) with μ = (``muT_c``, ``muQ_c``) and Σ(0) from *center* only.
    Half-cell mean/variance **slopes** are **not** used (diagnostic “center-Σ” cubature).
    """
    x, w = roots_hermite(int(n))
    g_t, g_q = np.meshgrid(x, x)
    w_t, w_q = np.meshgrid(w, w)
    nodes_std = np.vstack([g_t.ravel(), g_q.ravel()]) * np.sqrt(2.0)
    wgh = (w_t * w_q).ravel() / np.pi
    c0 = _center_cov_matrix(par)
    l_chol = np.linalg.cholesky(c0 + 1e-12 * np.eye(2))
    mu_t, mu_q, _, _ = _layer_mu_sigma(par)
    mu = np.array([mu_t, mu_q])
    phys = mu[:, None] + l_chol @ nodes_std
    return phys[0], phys[1], wgh


def condensate_from_nodes(
    t_nodes: np.ndarray, q_nodes: np.ndarray, w: np.ndarray
) -> float:
    """Cubature estimate E[max(q-q_sat,0)] with weights summing to ~1 for exact GH; GL is biased."""
    qs = q_sat_gpkg(t_nodes)
    integrand = np.maximum(q_nodes - qs, 0.0)
    return float(np.sum(w * integrand))


def condensate_stats(
    t_nodes: np.ndarray, q_nodes: np.ndarray, w: np.ndarray
) -> tuple[float, int]:
    """Return (cubature qc, number of nodes with positive condensate increment)."""
    qs = q_sat_gpkg(t_nodes)
    integrand = np.maximum(q_nodes - qs, 0.0)
    return float(np.sum(w * integrand)), int(np.sum(integrand > 0.0))


def gauss_hermite_nodes_bivariate(
    n: int, mu: np.ndarray, cov: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D tensor Gauss–Hermite nodes for ``N(μ, Σ)``; weights integrate to 1 on ℝ²."""
    n = int(n)
    mu = np.asarray(mu, dtype=float).reshape(2)
    cov = np.asarray(cov, dtype=float).reshape(2, 2)
    x, w = roots_hermite(n)
    g_t, g_q = np.meshgrid(x, x)
    w_t, w_q = np.meshgrid(w, w)
    nodes_std = np.vstack([g_t.ravel(), g_q.ravel()]) * np.sqrt(2.0)
    wgh = (w_t * w_q).ravel() / np.pi
    l_chol = np.linalg.cholesky(cov + 1e-14 * np.eye(2))
    phys = mu[:, None] + l_chol @ nodes_std
    return phys[0], phys[1], wgh


def _gh_tensor_chi1_split(n: int) -> tuple[int, int]:
    """Split ``χ_1`` into two contiguous bands (even-``N`` budget path): ``n_1 = n // 2``, ``n_2 = n - n_1``."""
    n = max(2, int(n))
    n1 = n // 2
    return n1, n - n1


def _chi1_wing_width_odd(n: int) -> int:
    """For odd ``N``, each of the two outer χ₁ bands has ``(N-1)//2`` abscissas; one middle index remains."""
    n = max(3, int(n))
    return (n - 1) // 2


def gauss_hermite_nodes_bivariate_chi1_slice(
    n: int,
    mu: np.ndarray,
    cov: np.ndarray,
    j0: int,
    j1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sub-tensor of :func:`gauss_hermite_nodes_bivariate` with the same node pairs as
    ``meshgrid(..., indexing="xy")`` / ravel: ``(χ_1, χ_2) = (x_j, x_i)``, restricting the
    **first** Gauss–Hermite factor ``χ_1 = x_j`` to ``j ∈ [j0, j1]`` and keeping all ``i``.

    Raw weights ``w_i w_j / π``; **not** renormalized here.
    """
    n = max(2, int(n))
    j0 = max(0, int(j0))
    j1 = min(n - 1, int(j1))
    if j1 < j0:
        return np.array([]), np.array([]), np.array([])
    mu = np.asarray(mu, dtype=float).reshape(2)
    cov = np.asarray(cov, dtype=float).reshape(2, 2)
    x, w = roots_hermite(n)
    t_list: list[np.ndarray] = []
    q_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    l_chol = np.linalg.cholesky(cov + 1e-14 * np.eye(2))
    # Same sequence as ``meshgrid(..., indexing="xy")`` C-order ravel: outer ``i``, inner ``j``.
    for i in range(n):
        for j in range(j0, j1 + 1):
            nodes_std = np.array([[x[j]], [x[i]]], dtype=float) * np.sqrt(2.0)
            wgh = float(w[i] * w[j] / np.pi)
            phys = mu[:, None] + l_chol @ nodes_std
            t_list.append(phys[0, 0])
            q_list.append(phys[1, 0])
            w_list.append(wgh)
    return (
        np.asarray(t_list, dtype=float),
        np.asarray(q_list, dtype=float),
        np.asarray(w_list, dtype=float),
    )


def mixture_quadrature_two_branches_at_z(
    par: dict,
    n_tq: int,
    *,
    z_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Exactly ``n_tq^2`` quadrature nodes on ``(T,q)`` at offset ``ζ = z_frac * L_{1/2}`` (from center
    toward **UP** face when ``z_frac>0``), using the **column-tensor** MVN
    :func:`column_tensor_mu_cov_zeta`.

    **Even** ``N``: two equal χ₁ bands (left / right ``j`` wings of the **same** tensor); display
    weights ``½`` per band.

    **Odd** ``N``: two outer χ₁ bands + one middle column (same MVN); **equal** ``⅓`` mass per
    stratum. ``stratum`` labels χ₁ bands (``0/1`` or ``0/1/2``) for test assertions.

    Returns ``t, q, w_display, stratum``, ``ζ_k``.
    """
    n_tq = max(2, int(n_tq))
    lh = _layer_L_half(par)
    zk = float(np.clip(float(z_frac), 0.0, 1.0)) * lh
    t, q, w, half = _mixture_partitioned_t_q_w_stratum_at_z(par, n_tq, zk)
    if t.size == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([], dtype=np.int8),
            zk,
        )
    return t, q, w, half, zk


def _mixture_partitioned_t_q_w_stratum_at_z(
    par: dict, n_tq: int, zk: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ``N^2`` budget cubature nodes and display weights summing to 1; ``stratum`` labels χ₁ bands
    (same MVN :func:`column_tensor_mu_cov_zeta` at ``ζ = zk``).
    """
    n = max(2, int(n_tq))
    mu, cov = column_tensor_mu_cov_zeta(float(zk), par)
    emp_t = np.array([], dtype=float)
    emp_h = np.array([], dtype=np.int8)

    if n % 2 == 0:
        n1 = n // 2
        td, qd, wd_raw = gauss_hermite_nodes_bivariate_chi1_slice(n, mu, cov, 0, n1 - 1)
        tu, qu, wu_raw = gauss_hermite_nodes_bivariate_chi1_slice(n, mu, cov, n1, n - 1)
        s_dn = float(np.sum(wd_raw))
        s_up = float(np.sum(wu_raw))
        if s_dn <= 0.0 or s_up <= 0.0:
            return emp_t, emp_t, emp_t, emp_h
        w_dn = 0.5 * wd_raw / s_dn
        w_up = 0.5 * wu_raw / s_up
        t = np.concatenate([td, tu])
        q = np.concatenate([qd, qu])
        w = np.concatenate([w_dn, w_up])
        half = np.concatenate(
            [np.zeros(td.shape[0], dtype=np.int8), np.ones(tu.shape[0], dtype=np.int8)]
        )
        return t, q, w, half

    # Odd N: left χ₁ wing | middle column | right χ₁ wing (same MVN)
    nw = _chi1_wing_width_odd(n)
    j_mid = nw
    tl, ql, wl_raw = gauss_hermite_nodes_bivariate_chi1_slice(n, mu, cov, 0, nw - 1)
    tm, qm, wm_raw = gauss_hermite_nodes_bivariate_chi1_slice(n, mu, cov, j_mid, j_mid)
    tr, qr, wr_raw = gauss_hermite_nodes_bivariate_chi1_slice(n, mu, cov, nw + 1, n - 1)
    s_l = float(np.sum(wl_raw))
    s_m = float(np.sum(wm_raw))
    s_r = float(np.sum(wr_raw))
    if s_l <= 0.0 or s_m <= 0.0 or s_r <= 0.0:
        return emp_t, emp_t, emp_t, emp_h
    # Equal display mass on each of the three χ₁ strata (sums to 1); same MVN on each stratum.
    m_third = 1.0 / 3.0
    w_l = m_third * wl_raw / s_l
    w_m = m_third * wm_raw / s_m
    w_r = m_third * wr_raw / s_r
    t = np.concatenate([tl, tm, tr])
    q = np.concatenate([ql, qm, qr])
    w = np.concatenate([w_l, w_m, w_r])
    half = np.concatenate(
        [
            np.zeros(tl.shape[0], dtype=np.int8),
            np.full(tm.shape[0], 1, dtype=np.int8),
            np.full(tr.shape[0], 2, dtype=np.int8),
        ]
    )
    return t, q, w, half


def _mixture_condensate_dn_up_partitioned_at_z(
    par: dict, n_tq: int, zk: float
) -> tuple[float, float]:
    """Legacy split: total partitioned condensate in ``qcd``; ``qcu`` is 0 (sum ``qcd+qcu`` is the estimate)."""
    t, q, w, _ = _mixture_partitioned_t_q_w_stratum_at_z(par, n_tq, zk)
    if t.size == 0:
        return 0.0, 0.0
    qs = q_sat_gpkg(t)
    total = float(np.sum(w * np.maximum(q - qs, 0.0)))
    return total, 0.0


def mixture_condensate_nested_gh_budget(par: dict, n_tq: int, nz_z: int) -> float:
    r"""
    Same outer ``ζ``-average as :func:`mixture_condensate_nested_gh`, but inner ``(T,q)`` uses the
    **χ₁-partitioned** ``N^2`` rule per ``ζ`` (:func:`_mixture_condensate_dn_up_partitioned_at_z`).
    Display-style **½ / ⅓** stratum renormalization is **not** the same quadrature as full tensor GH
    in general; for **even** ``N`` and one MVN it can coincide with the full tensor when each χ₁
    band carries half the Hermite weight mass. Same inner node layout as
    :func:`mixture_quadrature_two_branches_at_z` (used in **tests**).
    """
    nz = max(2, int(nz_z))
    n_tq = max(2, int(n_tq))
    lh = _layer_L_half(par)
    zetas = np.linspace(-lh, lh, nz)
    acc = 0.0
    for zeta in zetas:
        qcd, qcu = _mixture_condensate_dn_up_partitioned_at_z(par, n_tq, float(zeta))
        acc += (qcd + qcu) / float(nz)
    return float(acc)


def mixture_condensate_nested_gh(par: dict, n_tq: int, nz_z: int) -> float:
    r"""
    ``E[\max(q-q_{\mathrm{sat}}(T),0)]`` under the **column-tensor** MVN law matching
    :func:`mixture_pdf_grid`: ``ζ_k`` uniform on ``nz`` points in ``[-L_{1/2}, L_{1/2}]``; at each
    ``ζ`` one full ``N\times N`` Gauss–Hermite tensor for :func:`column_tensor_mu_cov_zeta`.
    Converges to :func:`mixture_truth_fine_grid_gpkg` as ``N\to\infty``. Budget χ-slice estimator:
    :func:`mixture_condensate_nested_gh_budget`.
    """
    nz = max(2, int(nz_z))
    n_tq = max(2, int(n_tq))
    lh = _layer_L_half(par)
    zetas = np.linspace(-lh, lh, nz)
    acc = 0.0
    for zeta in zetas:
        mu, cov = column_tensor_mu_cov_zeta(float(zeta), par)
        td, qd, wd = gauss_hermite_nodes_bivariate(n_tq, mu, cov)
        qcd, _ = condensate_stats(td, qd, wd)
        acc += qcd / float(nz)
    return float(acc)


def mixture_condensate_nested_gh_glz(par: dict, n_tq: int, n_z_gl: int) -> float:
    r"""
    Same expectation as :func:`mixture_condensate_nested_gh`, but the outer **ζ**-average on
    ``[-L_{1/2}, L_{1/2}]`` uses **Gauss–Legendre** nodes ``ζ_k = L_{1/2} · x_k`` for ``x_k`` on
    ``[-1,1]`` with weights ``w_k``:

    \[
      \frac{1}{2}\int_{-1}^{1} f(L_{1/2} x)\,dx \approx \frac{1}{2}\sum_k w_k f(\zeta_k).
    \]

    Inner rule: full tensor GH for :func:`column_tensor_mu_cov_zeta` at each ``ζ_k``.
    """
    nz_gl = max(2, int(n_z_gl))
    n_tq = max(2, int(n_tq))
    lh = _layer_L_half(par)
    x, wx = roots_legendre(nz_gl)
    acc = 0.0
    for k in range(nz_gl):
        zeta = float(lh) * float(x[k])
        mu, cov = column_tensor_mu_cov_zeta(zeta, par)
        td, qd, wd = gauss_hermite_nodes_bivariate(n_tq, mu, cov)
        qcd, _ = condensate_stats(td, qd, wd)
        acc += 0.5 * float(wx[k]) * qcd
    return float(acc)


def mixture_nested_gh_node_bundle(
    par: dict, n_tq: int, nz_z: int
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Same expectation as :func:`mixture_condensate_nested_gh`, plus all ``(T,q)`` nodes stacked over
    ``ζ`` (weights ``(1/n_z)`` times GH weights).
    """
    nz = max(2, int(nz_z))
    n_tq = max(2, int(n_tq))
    lh = _layer_L_half(par)
    zetas = np.linspace(-lh, lh, nz)
    t_all: list[float] = []
    q_all: list[float] = []
    w_all: list[float] = []
    acc = 0.0
    wz = 1.0 / float(nz)
    for zeta in zetas:
        mu, cov = column_tensor_mu_cov_zeta(float(zeta), par)
        td, qd, wd = gauss_hermite_nodes_bivariate(n_tq, mu, cov)
        qcd, _ = condensate_stats(td, qd, wd)
        acc += qcd * wz
        t_all.extend(td.tolist())
        q_all.extend(qd.tolist())
        w_all.extend((wz * wd).tolist())
    return float(acc), np.asarray(t_all), np.asarray(q_all), np.asarray(w_all)


def convergence_mixture_nested_gh(
    par: dict, nz_z: int, n_max: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Mixture nested GH ``q_c`` vs tensor order ``N = 2 … n_max`` at fixed ``nz_z``."""
    ns = np.arange(2, int(n_max) + 1)
    ser = np.empty_like(ns, dtype=float)
    for i, n in enumerate(ns):
        ser[i] = mixture_condensate_nested_gh(par, int(n), int(nz_z))
    return ns.astype(float), ser


def convergence_mixture_nested_gh_glz(
    par: dict, n_z_gl: int, n_max: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Mixture nested GH with **Gauss--Legendre** ``z``-average vs ``N = 2 … n_max`` (see :func:`mixture_condensate_nested_gh_glz`)."""
    ns = np.arange(2, int(n_max) + 1)
    ser = np.empty_like(ns, dtype=float)
    for i, n in enumerate(ns):
        ser[i] = mixture_condensate_nested_gh_glz(par, int(n), int(n_z_gl))
    return ns.astype(float), ser


def _center_cov0(par: dict) -> np.ndarray:
    return _center_cov_matrix(par)


def two_half_profile_rosenblatt_condensate(
    par: dict, n: int
) -> tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, str
]:
    """
    Layer–aware Profile–Rosen condensate and abscissae: **two-half cell** average of DN-only and
    UP-only cubature along their mean-gradient axes (:func:`profile_rosenblatt_cubature_two_halves_cell`).
    On degenerate gradient for **both** halves, falls back to 2D center-Σ Gauss–Hermite
    (same as :func:`gauss_hermite_nodes`).
    """
    C0 = _center_cov0(par)
    mu_t, mu_q, _, _ = _layer_mu_sigma(par)
    qc, t_n, q_n, w, u_n, v_n, use_l, st = profile_rosenblatt_cubature_two_halves_cell(
        int(n), par, C0, _layer_H(par), mu_t, mu_q, q_sat_gpkg
    )
    if st != "ok" or t_n.size == 0:
        t_n, q_n, w = gauss_hermite_nodes(int(n), par)
        qc = condensate_from_nodes(t_n, q_n, w)
        u_n = np.zeros_like(t_n)
        v_n = np.zeros_like(t_n)
        return float(qc), t_n, q_n, w, u_n, v_n, False, "fallback_center_gh"
    return float(qc), t_n, q_n, w, u_n, v_n, use_l, st


def convergence_mixture_tensor_gh_uniform_z(
    par: dict, n_max: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``q_c`` vs ``N`` for full tensor GH in ``(T,q)`` at each uniform ``z`` (mixture law); converges
    to :func:`mixture_truth_fine_grid_gpkg` as ``N → ∞``.
    """
    return convergence_mixture_nested_gh(
        par, nz_z=int(TRUTH_Z_SAMPLES), n_max=int(n_max)
    )


def convergence_two_half_profile_rosenblatt_N(
    par: dict, n_max: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``q_c`` vs tensor order ``N`` for the **two-half** profile–Rosen cubature — same target as
    :func:`truth_condensate_gpkg` (:func:`two_half_profile_rosenblatt_condensate`).
    """
    ns = np.arange(2, int(n_max) + 1)
    ser = np.empty_like(ns, dtype=float)
    for i, n in enumerate(ns):
        ser[i] = float(two_half_profile_rosenblatt_condensate(par, int(n))[0])
    return ns.astype(float), ser


def _two_half_tensor_halves_weighted_means(
    ft: np.ndarray,
    fq: np.ndarray,
    fw: np.ndarray,
    n_tensor: int,
    status: str,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """
    Weight-averaged ``(T,q)`` centroid for each tensor half (DN block first, then UP block),
    following the node ordering of :func:`profile_rosenblatt_cubature_two_halves_cell`.
    Returns ``(None, None)`` for any degenerate input.  Used in tests only.
    """
    fa = np.asarray(ft, dtype=float).ravel()
    qa = np.asarray(fq, dtype=float).ravel()
    wa = np.asarray(fw, dtype=float).ravel()
    if fa.size == 0 or wa.size == 0 or fa.size != wa.size:
        return None, None
    nh = int(n_tensor) * int(n_tensor)
    st = str(status)

    def _wm(sl: slice) -> tuple[float, float] | None:
        t_ = fa[sl]
        q_ = qa[sl]
        w_ = wa[sl]
        s = float(np.sum(w_))
        if s <= 0.0:
            return None
        return float(np.sum(t_ * w_) / s), float(np.sum(q_ * w_) / s)

    if st == "ok" and fa.size == 2 * nh:
        return _wm(slice(0, nh)), _wm(slice(nh, 2 * nh))
    if st == "half_dn_only" and fa.size == nh:
        return _wm(slice(0, nh)), None
    if st == "half_up_only" and fa.size == nh:
        return None, _wm(slice(0, nh))
    return None, None


def _fmt_qc(x: float) -> str:
    ax = abs(x)
    if ax == 0.0:
        return "0"
    if ax < 1e-5:
        return f"{x:.3e}"
    return f"{x:.5f}"


def run_dashboard(
    *,
    figsize: tuple[float, float] | None = None,
    collapse_help: bool = True,
):
    """
    Do not import matplotlib.pyplot at module import time: the notebook cell must run
    `%matplotlib widget` first; otherwise pyplot locks a non-interactive backend and
    Jupyter shows no figure output.

    Parameters
    ----------
    figsize
        Matplotlib figure size ``(w, h)`` in inches. Default ``(17, 5.2)`` (two panels).
    collapse_help
        If True (default), long explanatory HTML is inside a closed ``Accordion`` so
        sliders sit closer to the plots and the cell needs less vertical scrolling.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        get_ipython = lambda: None  # type: ignore[misc, assignment]

    ip = get_ipython()
    try:
        if ip is not None:
            ip.run_line_magic("matplotlib", "widget")
    except Exception:
        pass

    import matplotlib

    try:
        import ipympl  # noqa: F401 — registers the widget backend when present
    except ImportError:
        pass

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Avoid stacking duplicate ipympl figures when re-running the cell.
    plt.close("all")

    try:
        import ipywidgets as W
    except ImportError as e:
        raise ImportError(
            "ipywidgets is required for the control panel (Jupyter / JupyterLab). "
            "Install with: conda install ipywidgets  (or pip install ipywidgets)"
        ) from e
    from IPython.display import display

    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # `%matplotlib widget` often reports the backend as the short name "widget", not a string
    # containing "ipympl" — the old check falsely warned even when ipympl was installed.
    backend = matplotlib.get_backend()
    b = backend.lower()
    if ip is not None and not any(
        x in b for x in ("ipympl", "widget", "module://ipympl", "nbagg")
    ):
        print(
            "matplotlib backend is",
            repr(backend),
            "— for interactive figures install ipympl, then: %matplotlib widget",
        )

    # Plots only — controls are `ipywidgets` below the figure (HTML layout, no overlapping labels).
    _fs = figsize if figsize is not None else (17.0, 5.2)
    fig = plt.figure(figsize=_fs, constrained_layout=False, facecolor="white")
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.15, 1.05],
        wspace=0.28,
        left=0.07,
        right=0.99,
        top=0.90,
        bottom=0.12,
    )
    ax_conv = fig.add_subplot(gs[0, 0])
    ax_p1 = fig.add_subplot(gs[0, 1])

    for ax in (ax_conv, ax_p1):
        ax.tick_params(labelsize=9)
    ax_conv.set_xlabel(r"$N$ (points per dimension)", fontsize=10)
    ax_conv.set_ylabel(r"$q_c$ [g/kg]", fontsize=10)
    for _a in (ax_conv, ax_p1):
        _a.set_facecolor("white")

    # T–q window for contours is built inside ``update()`` from ``par`` (center μ, σ²).

    # Recompute only when the corresponding inputs change.
    _cache: dict = {
        "ph_conv": None,
        "ns": None,
        "pr_ser": None,
        "truth_key": None,
        "truth": 0.0,
        "purple_pdf_key": None,
        "purple_te": None,
        "purple_qe": None,
        "purple_pdf": None,
    }

    _L = W.Layout(flex="1 1 45%", min_width="200px", max_width="460px")
    _dw = "10.5rem"
    _style = {"description_width": _dw}

    def _fs(v, lo, hi, st, desc, fmt: str = ".3f"):
        # continuous_update=False: recompute when the user releases the handle (huge win vs 10+ heavy passes/s).
        return W.FloatSlider(
            value=v,
            min=lo,
            max=hi,
            step=st,
            description=desc,
            readout_format=fmt,
            style=_style,
            layout=_L,
            continuous_update=False,
        )

    def _is(v, lo, hi, st, desc, continuous_update: bool = False):
        return W.IntSlider(
            value=v,
            min=lo,
            max=hi,
            step=st,
            description=desc,
            style=_style,
            layout=_L,
            continuous_update=continuous_update,
        )

    wd: dict = {
        "muT_c": _fs(290.0, 273.0, 315.0, 0.5, "center μ_T [K]:", ".1f"),
        "muQ_c": _fs(10.0, 4.0, 22.0, 0.2, "center μ_q [g/kg]:", ".2f"),
        "varT_c": _fs(1.0, 0.05, 5.0, 0.05, "center σ²_T:", ".2f"),
        "varQ_c": _fs(1.5, 0.05, 6.0, 0.05, "center σ²_q:", ".2f"),
        "cov": _fs(0.8, -1.5, 1.5, 0.01, "cov Tq(0) [center]:"),
        "d_cov_dn": _fs(0.0, -1.0, 1.0, 0.02, "DN d(covTq)/dz:", ".2f"),
        "d_cov_up": _fs(0.0, -1.0, 1.0, 0.02, "UP d(covTq)/dz:", ".2f"),
        "mT_up": _fs(-1.5, -4, 4, 0.05, "UP mean dT/dz:", ".2f"),
        "mQ_up": _fs(-2.0, -4, 4, 0.05, "UP mean dq/dz:", ".2f"),
        "sT_up": _fs(0.5, -1.5, 1.5, 0.02, "UP dσ²T/dz:", ".2f"),
        "sQ_up": _fs(0.8, -1.5, 1.5, 0.02, "UP dσ²q/dz:", ".2f"),
        "mT_dn": _fs(1.0, -4, 4, 0.05, "DN mean dT/dz:", ".2f"),
        "mQ_dn": _fs(1.5, -4, 4, 0.05, "DN mean dq/dz:", ".2f"),
        "sT_dn": _fs(-0.4, -1.5, 1.5, 0.02, "DN dσ²T/dz:", ".2f"),
        "sQ_dn": _fs(-0.6, -1.5, 1.5, 0.02, "DN dσ²q/dz:", ".2f"),
        "L_half": _fs(L_HALF, 0.08, 2.5, 0.02, "L½ center→face [len]:", ".2f"),
        "fix_tq_window": W.Checkbox(
            value=False,
            description="Fix (T,q) plot window (σ² sliders then change cloud width)",
            style=_style,
            layout=W.Layout(width="100%", max_width="920px"),
        ),
        "Nquad": _is(6, 2, 16, 1, "N (quad, GH N² @ ζ=0 + left vline):", True),
        "n_truth": _is(96, 8, 160, 1, "N tensor (ground-truth q_c):"),
    }

    def _row(a, b):
        return W.HBox(
            [a, b],
            layout=W.Layout(
                width="100%",
                align_items="center",
                justify_content="space-between",
                gap="16px",
            ),
        )

    _help_long = W.HTML(
        r"""
                <div style="font-size:0.9em;line-height:1.38;max-width:58rem">
                <p style="margin:0.2em 0 0.45em 0"><b>This dashboard</b> — <b>one</b> ground-truth scalar <code>q_c</code> from the <b>two-half</b> profile–Rosen rule (<b>left</b> panel). The <b>middle</b> panel: <b>purple</b> = <b>layer-mean</b> column-tensor <code>p(T,q)</code> (depth average of <code>N(μ(ζ),Σ(ζ))</code>; Gauss–Legendre in <code>ζ</code> — see <code>docs/layer_mean_cell_Tq_marginal_derivation.md</code>); <b>not</b> the profile <code>p(T,q)</code> used for that <code>q_c</code>. <b>Black polyline</b> = <code>μ(ζ)</code> along depth; <b>blue</b> = <code>N×N</code> GH at <code>ζ=0</code>. See <code>ORACLE_SGS.md</code>.</p>
                <p style="margin:0.2em 0 0.35em 0"><b>N tensor</b> (8…160): tensor order for ground-truth <code>q_c</code> only. <b>N (quad)</b> (2…16): GH order for the <code>N²</code> middle-panel nodes and the left-panel marker.</p>
                </div>
                """
    )
    if collapse_help:
        _help_acc = W.Accordion(
            children=[_help_long],
            layout=W.Layout(width="100%", max_width="960px"),
        )
        _help_acc.set_title(0, "Help — one ground truth + optional MVN diagnostic")
        try:
            _help_acc.selected_index = None
        except Exception:
            pass
        _help_block = _help_acc
    else:
        _help_block = _help_long

    controls = W.VBox(
        [
            _row(wd["muT_c"], wd["muQ_c"]),
            _row(wd["varT_c"], wd["varQ_c"]),
            _row(wd["cov"], wd["d_cov_dn"]),
            W.HBox(
                [wd["d_cov_up"]],
                layout=W.Layout(
                    width="100%",
                    align_items="center",
                    justify_content="flex-start",
                    gap="16px",
                ),
            ),
            _row(wd["mT_up"], wd["mQ_up"]),
            _row(wd["sT_up"], wd["sQ_up"]),
            _row(wd["mT_dn"], wd["mQ_dn"]),
            _row(wd["sT_dn"], wd["sQ_dn"]),
            W.HBox(
                [wd["L_half"]],
                layout=W.Layout(
                    width="100%",
                    align_items="center",
                    justify_content="flex-start",
                    gap="16px",
                ),
            ),
            W.HBox(
                [wd["fix_tq_window"]],
                layout=W.Layout(
                    width="100%",
                    align_items="center",
                    justify_content="flex-start",
                    gap="16px",
                ),
            ),
            _row(wd["Nquad"], wd["n_truth"]),
            _help_block,
        ],
        layout=W.Layout(width="100%", max_width="960px", margin="6px 0 0 0"),
    )

    (line_truth,) = ax_conv.plot(
        [],
        [],
        color="k",
        lw=2.0,
        label="Ground-truth q_c (two-half profile–Rosen)",
    )
    (line_pr,) = ax_conv.plot(
        [],
        [],
        "-",
        color="tab:blue",
        lw=2.0,
        label="Two-half profile cubature vs N (same law as black)",
    )
    vline_n = ax_conv.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.8)
    ax_conv.legend(loc="best", fontsize=8, framealpha=0.92)
    ax_conv.grid(True, alpha=0.35)
    ax_conv.set_xlim(1.5, 16.5)

    def update(_=None):
        par = par_from_ipywidgets(wd)
        n_q = int(wd["Nquad"].value)
        n_tr = int(wd["n_truth"].value)
        phk = _physics_key(par)

        mu_t, mu_q, v_tc, v_qc = _layer_mu_sigma(par)
        n_t_vis, n_q_vis = 160, 160

        if bool(wd["fix_tq_window"].value):
            q_hi_pad = max(mu_q + 8.0, float(q_sat_gpkg(np.array([mu_t]))[0]) + 2.0)
            tq_bounds_plot = (
                float(mu_t - 12.0),
                float(mu_t + 12.0),
                float(max(0.1, mu_q - 5.0)),
                float(q_hi_pad),
            )
        else:
            tq_bounds_plot = mixture_marginal_axis_bounds(par, pad=6.5)
        t_lin = np.linspace(tq_bounds_plot[0], tq_bounds_plot[1], n_t_vis)
        q_lin = np.linspace(tq_bounds_plot[2], tq_bounds_plot[3], n_q_vis)

        purple_key = (
            "layer_mean_mvn_gl_purple_v1",
            phk,
            float(tq_bounds_plot[0]),
            float(tq_bounds_plot[1]),
            float(tq_bounds_plot[2]),
            float(tq_bounds_plot[3]),
            int(n_t_vis),
            int(n_q_vis),
            int(Z_GL_ORDER),
        )
        if _cache.get("purple_pdf_key") != purple_key:
            te_p, qe_p, pg_p = mixture_marginal_uniform_z_on_grid(
                par,
                float(tq_bounds_plot[0]),
                float(tq_bounds_plot[1]),
                float(tq_bounds_plot[2]),
                float(tq_bounds_plot[3]),
                n_t=int(n_t_vis),
                n_q=int(n_q_vis),
                z_quadrature="gauss_legendre",
                n_z_gl=int(Z_GL_ORDER),
            )
            _cache["purple_te"] = te_p
            _cache["purple_qe"] = qe_p
            _cache["purple_pdf"] = pg_p
            _cache["purple_pdf_key"] = purple_key

        # 1) Blue: two-half profile–Rosen vs N (same law as black).
        if _cache["ph_conv"] != phk:
            _cache["ns"], _cache["pr_ser"] = convergence_two_half_profile_rosenblatt_N(
                par, n_max=16
            )
            _cache["ph_conv"] = phk
        ns, pr_ser = _cache["ns"], _cache["pr_ser"]

        # 2) Middle panel purple: layer-mean column-tensor p(T,q) (cached above).

        # 3) Black: ground-truth q_c at TRUTH_PROFILE_ORDER (fine grid set by N slider).
        truth_key = (phk, n_tr)
        if _cache["truth_key"] != truth_key:
            _cache["truth"] = truth_condensate_gpkg(
                par,
                n_tr,
                n_tr,
                nz_pdf=int(TRUTH_Z_SAMPLES),
                tq_bounds=tq_bounds_plot,
                profile_order=int(n_tr),
            )
            _cache["truth_key"] = truth_key
        truth = _cache["truth"]
        pr_n16 = float(pr_ser[-1])
        gh_residual = float(abs(truth - pr_n16))

        line_truth.set_data([1.5, 16.5], [truth, truth])
        line_pr.set_data(ns, pr_ser)
        ax_conv.relim()
        ax_conv.autoscale_view()
        series_max = [
            float(np.max(pr_ser)),
            truth,
        ]
        y_hi = max(series_max) * 1.15 + 1e-12
        ax_conv.set_ylim(0.0, max(y_hi, truth * 1.05 + 1e-9, 1e-9))
        ylim = ax_conv.get_ylim()
        vline_n.set_data([float(n_q), float(n_q)], list(ylim))

        def _draw_tq_panel(ax, title_sub):
            ax.clear()
            ax.set_facecolor("white")
            te_p = _cache.get("purple_te")
            qe_p = _cache.get("purple_qe")
            pg_p = _cache.get("purple_pdf")
            if (
                te_p is not None
                and qe_p is not None
                and pg_p is not None
                and np.asarray(pg_p).size > 0
            ):
                pg_arr = np.asarray(pg_p, dtype=float)
                pvmax = float(np.max(pg_arr))
                if pvmax > 0.0:
                    ax.pcolormesh(
                        te_p,
                        qe_p,
                        pg_arr.T,
                        shading="auto",
                        cmap="Purples",
                        norm=Normalize(vmin=0.0, vmax=pvmax),
                        alpha=PURPLE_PCOLORMESH_ALPHA,
                        rasterized=True,
                        zorder=0.35,
                    )
            nn = int(max(2, int(n_q)))
            mu0, cov0 = column_tensor_mu_cov_zeta(0.0, par)
            t_gh, q_gh, _wgh = gauss_hermite_nodes_bivariate(
                nn, np.asarray(mu0, dtype=float), np.asarray(cov0, dtype=float)
            )
            s_mark = max(9, min(26, int(520 // max(1, nn * nn))))
            ax.scatter(
                t_gh,
                q_gh,
                s=s_mark,
                facecolors="#d4e8ff",
                edgecolors="tab:blue",
                linewidths=0.75,
                alpha=0.93,
                zorder=5.2,
                rasterized=True,
                label=rf"GH $N\times N$ @ $\zeta$=0 (${nn}^2$)",
            )
            ax.fill_between(
                t_lin,
                q_sat_gpkg(t_lin),
                q_lin[-1] + 5.0,
                color="#ffe0e0",
                alpha=0.5,
                zorder=1,
            )
            ax.plot(t_lin, q_sat_gpkg(t_lin), "r-", lw=1.5, zorder=3)
            ax.axvline(mu_t, color="k", lw=1.4, zorder=4)
            ax.axhline(mu_q, color="k", lw=1.4, zorder=4)
            ax.set_xlim(t_lin[0], t_lin[-1])
            ax.set_ylim(q_lin[0], q_lin[-1])
            ax.set_xlabel("T [K]  (layer-mean phase plane — not height z)", fontsize=9)
            ax.set_ylabel("q [g/kg]", fontsize=10)
            ax.set_title(title_sub, fontsize=8.5)
            ax.text(
                0.02,
                0.02,
                "Purple = layer-mean column-tensor p(T,q) (GL average in depth ζ; not profile q_c law). "
                f"Black = mean (T,q) along depth (same column model). "
                f"Blue = N×N GH at ζ=0 (N={nn}).",
                transform=ax.transAxes,
                fontsize=6.0,
                va="bottom",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    alpha=0.92,
                    edgecolor="0.75",
                    linewidth=0.8,
                ),
                zorder=15,
            )

        sub_title = (
            f"Purple = layer-mean p(T,q), column-tensor, GL ζ ({n_t_vis}², nζ={Z_GL_ORDER}) | "
            f"GH {n_q}² @ center | black μ(ζ)"
        )
        _draw_tq_panel(ax_p1, sub_title)
        _lh = _layer_L_half(par)
        mu_dn = column_mean_Tq(-_lh, par)
        mu_ct = column_mean_Tq(0.0, par)
        mu_up = column_mean_Tq(_lh, par)
        t_dnf, q_dnf = float(mu_dn[0]), float(mu_dn[1])
        mu_tc, mu_qc = float(mu_ct[0]), float(mu_ct[1])
        t_upf, q_upf = float(mu_up[0]), float(mu_up[1])
        ax_p1.plot(
            [t_dnf, mu_tc, t_upf],
            [q_dnf, mu_qc, q_upf],
            color="k",
            lw=2.15,
            ls="-",
            zorder=5.7,
            solid_capstyle="round",
            label="Mean (T,q) vs depth (linear column; not profile p)",
        )
        ax_p1.scatter(
            [t_dnf, t_upf],
            [q_dnf, q_upf],
            s=[85, 85],
            facecolors="white",
            edgecolors="k",
            linewidths=1.35,
            zorder=5.75,
        )
        # High-contrast marker: crosshairs alone are easy to mis-read against Purples + saturation fill.
        ax_p1.plot(
            mu_tc,
            mu_qc,
            linestyle="none",
            marker="*",
            color="#ffd200",
            markeredgecolor="k",
            markeredgewidth=0.55,
            markersize=12,
            zorder=8,
            label=r"cell-center $\mu(0)$",
        )
        ax_p1.legend(loc="upper left", fontsize=7, framealpha=0.92)

        ax_conv.set_title(
            rf"Ground truth={_fmt_qc(truth)} · profile@N=16={_fmt_qc(pr_n16)} ($|\Delta|$={_fmt_qc(gh_residual)}) [g/kg]",
            fontsize=8,
        )
        gap_note = ""
        if truth > 1e-6 and gh_residual / max(truth, 1e-20) > 0.12:
            gap_note = " — Two-half profile cubature at N=16 still >12% from fine grid; raise N truth (ORACLE_SGS.md)."
        fig.suptitle(
            "Ground-truth q_c (L): profile–Rosen convergence + layer-mean column p(T,q) + GH N² @ ζ=0"
            + gap_note,
            fontsize=10,
            y=0.97,
            color="#b35900" if gap_note else "#111111",
        )
        ax_conv.legend(loc="best", fontsize=8, framealpha=0.92)
        fig.canvas.draw_idle()

    for wobj in wd.values():
        wobj.observe(lambda _c: update(), names="value")
    update()
    if ip is not None:
        display(
            W.VBox(
                [fig.canvas, controls],
                layout=W.Layout(align_items="flex-start", width="100%"),
            )
        )
    else:
        plt.show()


if __name__ == "__main__":
    # Allow `python variance_dashboard_interactive.py` if run in IPython-free env
    try:
        run_dashboard()
    except Exception as e:
        print("Run this from Jupyter/IPython with %matplotlib widget:", e)
