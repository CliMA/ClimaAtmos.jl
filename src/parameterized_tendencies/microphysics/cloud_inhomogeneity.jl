# ============================================================================
# Subgrid cloud condensate inhomogeneity for shortwave radiation
# ============================================================================
#
# Plane-parallel shortwave radiative transfer evaluated at the mean in-cloud
# condensate overestimates cloud reflectance, because reflectance is concave
# in optical depth. The effective-thickness approximation of Cahalan et al.
# (1994) replaces the mean optical depth `E[τ]` by its geometric mean
# `exp(E[ln τ])`, i.e. scales the in-cloud condensate by
#
#     χ = exp(E[ln τ]) / E[τ] ∈ (0, 1]
#
# (Jensen's inequality). For a Gamma distribution of shape `ν` (Barker, 1996),
# `ln χ = ψ(ν) − ln ν` (`log_gamma_effective_thickness`). The factor applies
# to the shortwave solve only (`RRTMGPInterface.solve_fluxes!`): longwave
# emissivity is linear in optical depth for thin clouds, so the same scaling
# would understate the emissivity of thin cirrus and raise the outgoing
# longwave radiation. The closures below evaluate `χ` for the cloudy part of
# a cell as PROPHET represents it: the truncated-Gaussian cloud of the
# subdomain with the SGS closure (the environment under PrognosticEDMFX, the
# grid mean otherwise) plus the homogeneous cloudy updrafts, with an
# unresolved multiplicative inhomogeneity that applies to every cloudy
# component.

"""
    log_gamma_effective_thickness(ν)

Logarithm `ln χ = ψ(ν) − ln ν` [-] of the effective-thickness factor
`χ = exp(E[ln τ]) / E[τ]` of a Gamma-distributed optical depth with shape
`ν > 0` (inverse relative variance), with `ψ` the digamma function.

Shifts `ν` to `y = ν + 6` with `ψ(ν) = ψ(y) − Σ_{k=0}^{5} 1/(ν + k)` and
evaluates the asymptotic expansion
`ψ(y) − ln y = −1/(2y) − 1/(12y²) + 1/(120y⁴) − 1/(252y⁶)` at `y ≥ 6`
(absolute error below `3 × 10⁻⁹`). Combining `ln(y/ν) = log1p(6/ν)` with
the shift keeps the result finite for `ν = Inf`, where it is `0` (a
homogeneous cloud), and it tends to `−Inf` as `ν → 0⁺`.
"""
@inline function log_gamma_effective_thickness(ν)
    FT = typeof(ν)
    shift = -(
        inv(ν) + inv(ν + FT(1)) + inv(ν + FT(2)) +
        inv(ν + FT(3)) + inv(ν + FT(4)) + inv(ν + FT(5))
    )
    inv_y = inv(ν + FT(6))
    inv_y2 = inv_y * inv_y
    asymptotic =
        -FT(0.5) * inv_y -
        inv_y2 * (FT(1 / 12) - inv_y2 * (FT(1 / 120) - inv_y2 * FT(1 / 252)))
    return asymptotic + log1p(FT(6) / ν) + shift
end

"""
    truncated_gaussian_relative_variance(cf)

Relative variance `v / m² ∈ (0, 1]` [-] of the positive tail `S > 0` of a
Gaussian saturation excess `S ~ N(z, 1)` whose cloud fraction is
`cf = Φ(z)`, i.e. of the in-cloud condensate of the truncated-Gaussian cloud
closure (`_compute_cloud_fraction`). With the inverse Mills ratio
`r = φ(z) / Φ(z)`, the conditional mean and variance are `m = z + r` and
`v = 1 − z r − r²`. The relative variance tends to `1` (an exponential tail)
as `cf → 0⁺` and to `1/z² → 0` as `cf → 1⁻`.

`z` is recovered from `cf` by `normal_cdf_inv` and one Newton step on
`Φ(z) = normal_tail_prob(−z)`; `cf` is clamped to `[10⁻⁶, 1 − 10⁻⁴]`, so the
relative variance is constant (≈ 0.072) above `1 − 10⁻⁴` and continuous
there. For `z < −2.4` (`cf ≲ 0.008`), where `m = z + r` and `v = 1 − z r − r²`
suffer from subtractive cancellation, `v / m²` is evaluated from the
continued-fraction convergents `(d, e) = _mills_continued_fraction(−z)` of
`r(−x) = x + 1/(x + d)`, `d = 2/(x + e)`, for which `v / m² = 1 − d e + d²`
identically (relative error below `10⁻⁷` there). The maximum relative error
across the clamped range is `≈ 2 × 10⁻⁴`, just above the switch, where the
absolute accuracy of `normal_cdf` in `Φ(z)` is amplified by the cancellation
in `m` and `v`. `cf ≤ 0` returns the limit `1`.
"""
@inline function truncated_gaussian_relative_variance(cf_in)
    FT = typeof(cf_in)
    cf = clamp(cf_in, FT(1e-6), one(FT) - FT(1e-4))
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    z0 = normal_cdf_inv(cf)
    φ0 = inv_sqrt2π * exp(-FT(0.5) * z0 * z0)
    z = clamp(
        z0 - (normal_tail_prob(-z0) - cf) / max(φ0, FT(1e-12)),
        FT(-6),
        FT(6),
    )
    # Direct moments of the truncated Gaussian.
    Φz = clamp(normal_tail_prob(-z), FT(1e-12), one(FT))
    r = inv_sqrt2π * exp(-FT(0.5) * z * z) / Φz
    m = max(z + r, FT(1e-4))
    v = clamp(one(FT) - z * r - r * r, FT(1e-4), one(FT))
    rv_direct = v / (m * m)
    # Lower tail: continued fraction of the Mills ratio.
    d, e = _mills_continued_fraction(abs(z))
    rv_tail = one(FT) - d * e + d * d
    rv = ifelse(z < FT(-2.4), rv_tail, rv_direct)
    return ifelse(cf_in <= zero(FT), one(FT), clamp(rv, zero(FT), one(FT)))
end

"""
    cloud_condensate_relative_variance(cf, rv_floor)

Relative variance [-] of the in-cloud condensate of the truncated-Gaussian
cloud with cloud fraction `cf` [-], multiplied by an independent unit-mean
inhomogeneity of relative variance `rv_floor` [-], set by
`cloud_condensate_relative_variance_floor`:

    1 + rv = (1 + rv_gauss(cf)) (1 + rv_floor).

The factor represents in-cloud variability beyond the saturation-excess PDF:
the truncated Gaussian of an overcast layer (`cf → 1`) is nearly homogeneous,
whereas observed overcast stratocumulus has a relative standard deviation of
liquid water path of about 0.4–0.8.

The factor applies to every cloud type. Its default is set from overcast
stratocumulus; ice and convective clouds are more inhomogeneous, and the
inhomogeneity grows with grid spacing, so a scale-aware or phase-dependent
value is a possible refinement. For ice under non-equilibrium microphysics,
the truncated-Gaussian shape is a weaker model and the floor accounts for more
of the variability.
"""
@inline cloud_condensate_relative_variance(cf, rv_floor) =
    (one(cf) + truncated_gaussian_relative_variance(cf)) * (one(cf) + rv_floor) -
    one(cf)

"""
    cloud_optical_depth_factor(
        cf, a_env, q_env, a_up, aq_up, aq2_up, alnq_up, rv_floor,
    )

Effective-thickness factor `χ ∈ (0, 1]` [-] of the in-cloud condensate over
the cloud cover `cf` [-] of a cell, by which the mean in-cloud water path
passed to radiation is scaled.

The cloud is a mixture of components `k` with cloudy area `A_k`, mean in-cloud
condensate `μ_k`, and Gamma-distributed in-cloud condensate of shape `ν_k`
(relative variance `1/ν_k`). `χ` is the larger of two estimates of
`exp(E[ln q]) / E[q]`, which coincide for a single component:

  - the geometric mean of the mixture, exact for the effective-thickness
    approximation because `ln` is additive over the components,
    `E[ln q] = Σ A_k (ln μ_k + ψ(ν_k) − ln ν_k) / Σ A_k`;
  - the Gamma factor `exp(ψ(ν) − ln ν)` of the mixture's own mean and
    variance, `1/ν = Σ A_k μ_k² (1 + 1/ν_k) / (Σ A_k E[q]²) − 1`.

Each underestimates `χ` where the other holds. The geometric mean diverges to
zero when a component holds almost no condensate, such as an updraft that the
binary cover test (`_edmf_weight_cloud_fraction!`) counts as cloudy with trace
condensate, although such a component is optically clear and reflects linearly
in its optical depth. The moment-matched Gamma puts the excess of a small,
dense component into a heavy tail near zero, whereas the geometric mean weighs
that component by its area.

The components are

  - the subdomain with the SGS cloud closure, with area fraction
    `a_env` [-] and mean cloud condensate `q_env` [kg/kg]; its cloud fraction
    `cf_env = (cf − a_up) / a_env` inverts the EDMF cover weighting, its
    cloudy area is `a_env cf_env`, its in-cloud mean `q_env / cf_env`, and its
    relative variance `cloud_condensate_relative_variance(cf_env, rv_floor)`;
  - the cloudy updrafts, homogeneous apart from the unresolved inhomogeneity
    `rv_floor` [-], entering through their summed area `a_up = Σ a_j` [-],
    condensate `aq_up = Σ a_j q_j` [kg/kg], squared condensate
    `aq2_up = Σ a_j q_j²` [kg²/kg²], and log-condensate
    `alnq_up = Σ a_j ln q_j` [-].

For a single grid-mean domain (`a_env = 1` and vanishing updraft sums),
`χ = exp(ψ(ν) − ln ν)` depends only on `cf`. Returns `1` where the cell has
no cloudy component.
"""
@inline function cloud_optical_depth_factor(
    cf, a_env, q_env, a_up, aq_up, aq2_up, alnq_up, rv_floor,
)
    FT = typeof(cf)
    cf_env = clamp((cf - a_up) / max(a_env, eps(FT)), zero(FT), one(FT))
    has_env = (cf_env > zero(FT)) & (q_env > zero(FT))
    A_env = ifelse(has_env, a_env * cf_env, zero(FT))
    μ_env = max(q_env / max(cf_env, eps(FT)), floatmin(FT))
    rv_env = cloud_condensate_relative_variance(cf_env, rv_floor)
    rv_up = max(rv_floor, zero(FT))
    A = A_env + a_up
    A_safe = max(A, eps(FT))
    mean_q = max((A_env * μ_env + aq_up) / A_safe, floatmin(FT))
    # Geometric mean of the mixture.
    mean_lnq =
        (
            A_env * (log(μ_env) + log_gamma_effective_thickness(inv(rv_env))) +
            alnq_up +
            a_up * log_gamma_effective_thickness(inv(rv_up))
        ) / A_safe
    χ_geometric = exp(mean_lnq - log(mean_q))
    # Gamma matched to the mixture's mean and variance.
    mean_q2 =
        (A_env * μ_env^2 * (one(FT) + rv_env) + aq2_up * (one(FT) + rv_up)) /
        A_safe
    rv_mixture = max(mean_q2 / mean_q / mean_q - one(FT), zero(FT))
    χ_moments = exp(log_gamma_effective_thickness(inv(rv_mixture)))
    χ = max(χ_geometric, χ_moments)
    return ifelse(
        (A > zero(FT)) & (mean_q > floatmin(FT)),
        clamp(χ, zero(FT), one(FT)),
        one(FT),
    )
end

"""
    _updraft_cloud_moments_j(Y, p, j, thermo_params, microphysics_model)

Lazy contributions of updraft `j` to the cloudy-updraft sums of
`cloud_optical_depth_factor`: its cloudy area `a_j` [-], condensate
`a_j q_j` [kg/kg], squared condensate `a_j q_j²` [kg²/kg²], and
log-condensate `a_j ln q_j` [-], the last `0` where the updraft is clear.
"""
function _updraft_cloud_moments_j(Y, p, j, thermo_params, microphysics_model)
    FT = eltype(p.params)
    ᶜqʲ, ᶜaʲ = _updraft_cloud_condensate_and_area(
        Y, p, j, thermo_params, microphysics_model,
    )
    ᶜaqʲ = @. lazy(ᶜaʲ * ᶜqʲ)
    ᶜaq2ʲ = @. lazy(ᶜaʲ * ᶜqʲ * ᶜqʲ)
    ᶜalnqʲ = @. lazy(
        ifelse(ᶜaʲ > zero(FT), ᶜaʲ * log(max(ᶜqʲ, floatmin(FT))), zero(FT)),
    )
    return ᶜaʲ, ᶜaqʲ, ᶜaq2ʲ, ᶜalnqʲ
end

"""
    _updraft_cloud_moments(Y, p, thermo_params, turbconv_model)

Lazy sums over the mass-flux updrafts of the moments of
`_updraft_cloud_moments_j`, `(a_up, aq_up, aq2_up, alnq_up)`, which enter
the single broadcast of `set_sw_cloud_optical_depth_factor!`; four zeros
without `PrognosticEDMFX`. The broadcast repeats the cloudy-area test of
each updraft in all four moments, so its expression, and with it the GPU
kernel's parameter size, grows with the number of updrafts.
"""
function _updraft_cloud_moments(Y, p, thermo_params, turbconv_model)
    FT = eltype(p.params)
    z = zero(FT)
    return z, z, z, z
end

function _updraft_cloud_moments(
    Y, p, thermo_params, ::PrognosticEDMFX{N},
) where {N}
    microphysics_model = p.atmos.microphysics_model
    return mapreduce(
        j -> _updraft_cloud_moments_j(
            Y, p, j, thermo_params, microphysics_model,
        ),
        (m1, m2) -> map((x, y) -> (@. lazy(x + y)), m1, m2),
        ntuple(identity, Val(N)),
    )
end

"""
    set_sw_cloud_optical_depth_factor!(Y, p)

Fill `p.radiation.ᶜsw_cloud_optical_depth_factor`, the factor by which the
shortwave solve (`RRTMGPInterface.solve_fluxes!`) scales the in-cloud liquid
and ice water paths, from the current cloud fraction and condensate; return
`nothing`. No-op where the radiation cache has no such field: radiation
without interactive clouds, idealized clouds, non-quadrature cloud models, or
`cloud_inhomogeneity_in_sw_radiation: false`, which leaves the shortwave water
paths unscaled (`χ = 1`) for comparison runs.

Dispatches on the cloud model: `QuadratureCloud` uses the truncated-Gaussian
closure (`cloud_optical_depth_factor`), whose subgrid PDF also sets the cloud
fraction; every other cloud model (`GridScaleCloud`, `MLCloud`) has no
in-cloud distribution and gets `χ = 1`. Under `PrognosticEDMFX`, the
environment and the cloudy updrafts enter as separate components, as in the
cloud cover.
"""
function set_sw_cloud_optical_depth_factor!(Y, p)
    hasproperty(p.radiation, :ᶜsw_cloud_optical_depth_factor) || return nothing
    set_sw_cloud_optical_depth_factor!(
        p.radiation.ᶜsw_cloud_optical_depth_factor,
        Y,
        p,
        p.atmos.cloud_model,
        p.atmos.turbconv_model,
    )
    return nothing
end

set_sw_cloud_optical_depth_factor!(ᶜχ, Y, p, cloud_model, turbconv_model) =
    fill!(ᶜχ, one(eltype(ᶜχ)))

function set_sw_cloud_optical_depth_factor!(
    ᶜχ, Y, p, ::QuadratureCloud, turbconv_model,
)
    FT = eltype(p.params)
    rv_floor = FT(CAP.cloud_condensate_relative_variance_floor(p.params))
    thermo_params = CAP.thermodynamics_params(p.params)
    microphysics_model = p.atmos.microphysics_model
    (; ᶜcloud_fraction) = p.precomputed
    ᶜa⁰ = _env_area_fraction(Y, p, thermo_params, turbconv_model)
    ᶜq_lcl, ᶜq_icl =
        _get_condensate_means(Y, p, turbconv_model, microphysics_model)
    ᶜa_up, ᶜaq_up, ᶜaq2_up, ᶜalnq_up =
        _updraft_cloud_moments(Y, p, thermo_params, turbconv_model)
    @. ᶜχ = cloud_optical_depth_factor(
        ᶜcloud_fraction, ᶜa⁰, ᶜq_lcl + ᶜq_icl, ᶜa_up, ᶜaq_up, ᶜaq2_up,
        ᶜalnq_up, rv_floor,
    )
    return nothing
end
