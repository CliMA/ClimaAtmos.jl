#####
##### Subgrid / subcell variance statistics (pure scalar kernels)
#####
##### Used by SGS quadrature and precomputed quantities (two-slope, face-anchored
##### reconstruction of subcell geometric contributions to variances / covariance).

import SpecialFunctions: erf

"""
    ϵ_variance_statistics(FT)

Numerical floor for variance–correlation algebra (same idea as `ϵ_numerics` in
`microphysics_cache.jl`).
"""
@inline ϵ_variance_statistics(FT) = cbrt(floatmin(FT))

"""
    _two_slope_d_D(s_dn, s_up)

Centered slope ``d = (s_+ + s_-) / 2`` and slope asymmetry ``D = s_+ - s_-`` from
backward / forward half-slopes ``s_-, s_+`` of a center-defined field on a two-slope
piecewise-linear reconstruction.
"""
@inline function _two_slope_d_D(s_dn, s_up)
    half = one(typeof(s_dn)) / 2
    d = half * (s_up + s_dn)
    D = s_up - s_dn
    return d, D
end

"""
    subcell_geometric_variance_increment(Δz, s_q_dn, s_q_up, s_T_dn, s_T_up)

Layer-mean **geometric** variance increment of a center-defined field reconstructed
two-slope piecewise-linearly across one cell of thickness `Δz` (eq.
`two-slope-geomvar` of the methods section):

```math
\\overline{(\\phi - \\bar\\phi)^2}
  \\;=\\; \\frac{d^2 \\, \\Delta z^2}{12}
       \\;+\\; \\frac{D^2 \\, \\Delta z^2}{192},
```

with centered slope ``d = (s_+ + s_-)/2`` and asymmetry ``D = s_+ - s_-``.
Returns `(Δq′q′, ΔT′T′)`. Reduces to the classical ``\\tfrac{1}{12}\\Delta z^2 d^2``
form when `s_dn == s_up`.

Arguments are **half-slopes in `z`** (i.e. forward and backward differences of
adjacent center values divided by `Δz`); the caller is responsible for converting
``\\theta_{li}``-space half-slopes to ``T``-space via the local Jacobian
``\\partial T / \\partial \\theta_{li}`` before passing `s_T_dn, s_T_up`.
"""
@inline function subcell_geometric_variance_increment(
    Δz,
    s_q_dn,
    s_q_up,
    s_T_dn,
    s_T_up,
)
    FT = typeof(Δz)
    twelfth = one(FT) / 12
    one192 = one(FT) / 192
    d_q, D_q = _two_slope_d_D(s_q_dn, s_q_up)
    d_T, D_T = _two_slope_d_D(s_T_dn, s_T_up)
    Δz2 = Δz^2
    Δq = twelfth * Δz2 * d_q^2 + one192 * Δz2 * D_q^2
    ΔT = twelfth * Δz2 * d_T^2 + one192 * Δz2 * D_T^2
    return Δq, ΔT
end

"""
    subcell_geometric_covariance_Tq(Δz, s_q_dn, s_q_up, s_T_dn, s_T_up)

Layer-mean **geometric** ``\\mathrm{Cov}(T', q')`` increment for two two-slope
piecewise-linear fields (eq. `two-slope-geomcov` of the methods section):

```math
\\overline{(T-\\bar T)(q-\\bar q)}
  \\;=\\; \\frac{d^T d^q \\, \\Delta z^2}{12}
       \\;+\\; \\frac{D^T D^q \\, \\Delta z^2}{192}.
```

Half-slopes are in `z` and in **`T`-space** (caller supplies the ``\\partial T /
\\partial \\theta_{li}`` rotation).
"""
@inline function subcell_geometric_covariance_Tq(
    Δz,
    s_q_dn,
    s_q_up,
    s_T_dn,
    s_T_up,
)
    FT = typeof(Δz)
    twelfth = one(FT) / 12
    one192 = one(FT) / 192
    d_q, D_q = _two_slope_d_D(s_q_dn, s_q_up)
    d_T, D_T = _two_slope_d_D(s_T_dn, s_T_up)
    Δz2 = Δz^2
    return twelfth * Δz2 * d_q * d_T + one192 * Δz2 * D_q * D_T
end

"""
    subcell_layer_mean_excursion(Δz, s_dn, s_up)

Cell-mean offset ``\\bar\\phi - \\phi_c = \\tfrac{\\Delta z}{8}\\,D`` induced by a
two-slope piecewise-linear reconstruction with asymmetry `D = s_up - s_dn`
(eq. `two-slope-layer-mean` of the methods section). Zero when `s_dn == s_up`.
"""
@inline function subcell_layer_mean_excursion(Δz, s_dn, s_up)
    return (Δz / oftype(Δz, 8)) * (s_up - s_dn)
end

"""
    uniform_normal_convolution_pdf(x, a, b, σ)

PDF at `x` of the convolution of `Uniform(a,b)` with `Normal(0,σ)` (univariate).
For `a == b`, reduces to `Normal(a, σ)`.
"""
function uniform_normal_convolution_pdf(x, a, b, σ)
    FT = typeof(x)
    πf = FT(π)
    σp = max(σ, ϵ_variance_statistics(FT))
    s2 = σp * sqrt(FT(2))
    if a == b
        return (one(FT) / (σp * sqrt(FT(2) * πf))) *
               exp(-((x - a) / σp)^2 / 2)
    end
    return (one(FT) / (2 * (b - a))) * (erf((x - a) / s2) - erf((x - b) / s2))
end

"""
    bivariate_uniform_normal_isotropic_pdf(q, T, μ_q, μ_T, δq_half, δT_half, σ)

Bivariate PDF at `(q, T)` in **fully factorized form** ``f(q,T) = f_q(q) f_T(T)``: each
marginal is a univariate uniform–normal convolution on
`[μ_q − δq_half, μ_q + δq_half]` and `[μ_T − δT_half, μ_T + δT_half]` respectively, with
**the same** Gaussian noise scale `σ` on both axes. That is exactly **`Σ_turb = σ² I`**
(diagonal turbulent covariance — this kernel assigns **no** turbulent `q`–`T` cross-covariance).

That is **not** a statement that moisture and temperature are uncoupled in the real SGS
closure: operationally we still carry **`q′q′`**, **`T′T′`**, and **`corr_Tq`**
(`correlation_Tq`), with geometry in `subcell_geometric_covariance_Tq` and the
production layer-profile quadrature in `subgrid_layer_profile_quadrature.jl`. This
function is only the **diagonal-`Σ_turb`, product-density** special case for tests and
pencil-and-paper checks; it does **not** accept `corr_Tq` or distinct `σ_q`, `σ_T`.
"""
function bivariate_uniform_normal_isotropic_pdf(
    q,
    T,
    μ_q,
    μ_T,
    δq_half,
    δT_half,
    σ,
)
    FT = typeof(q)
    σp = max(σ, ϵ_variance_statistics(FT))
    return uniform_normal_convolution_pdf(q, μ_q - δq_half, μ_q + δq_half, σp) *
           uniform_normal_convolution_pdf(T, μ_T - δT_half, μ_T + δT_half, σp)
end
