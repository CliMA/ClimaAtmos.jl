# Scalar algebra mirroring `sgs_quadrature_moments_from_gradients` / `variance_statistics.jl`
# (moment-matched Gaussian subcell geometry) for a **single vertical column**:
# `dot(WVector(∇q), WVector(∇q)) → (∂q/∂z)²`, same for θ_li, and
# `dot(WVector(∇q), WVector(∇θ)) → (∂q/∂z)(∂θ_li/∂z)`.

"""
    mathsanity_sgs_quad_moments_with_geometry(qq_turb, TT_turb, ρ_param, dz, ∂q∂z, ∂θ∂z, ∂T∂θ_li; ε=1e-10)

Reproduce ClimaAtmos quadrature moment **materialization** (varfix path):

- `var_q = q′q′ + (1/12) Δz² (∂q/∂z)²`
- `var_T = T′T′ + (1/12) Δz² (∂T/∂θ_li)² (∂θ_li/∂z)²`
- Effective correlation: turbulent part `ρ_param·σ_q,turb·σ_T,turb` plus geometric cross
  `(1/12)Δz² (∂T/∂θ_li)(∂q/∂z)(∂θ_li/∂z)`, divided by `σ_q,tot·σ_T,tot`, clamped to [-1,1].

Returns `(var_q, var_T, ρ_eff, (; Δvar_q, Δvar_T, cov_geom))`.
"""
function mathsanity_sgs_quad_moments_with_geometry(
    qq_turb::FT,
    TT_turb::FT,
    ρ_param::FT,
    dz::FT,
    dq_dz::FT,
    dtheta_dz::FT,
    dT_dtheta::FT;
    ε::FT = FT(1e-10),
) where {FT <: AbstractFloat}
    twelfth = one(FT) / 12
    dqdz_sq = dq_dz^2
    dTdz_sq = dT_dtheta^2 * dtheta_dz^2
    Δvar_q = twelfth * dz^2 * dqdz_sq
    Δvar_T = twelfth * dz^2 * dTdz_sq
    var_q = qq_turb + Δvar_q
    var_T = TT_turb + Δvar_T
    dot_wq_wθ = dq_dz * dtheta_dz
    cov_geom = twelfth * dz^2 * dT_dtheta * dot_wq_wθ
    cov_turb = ρ_param * sqrt(max(zero(FT), qq_turb)) * sqrt(max(zero(FT), TT_turb))
    numer = cov_turb + cov_geom
    denom = max(ε, sqrt(max(zero(FT), var_q)) * sqrt(max(zero(FT), var_T)))
    ρ_eff = clamp(numer / denom, -one(FT), one(FT))
    return var_q, var_T, ρ_eff, (; Δvar_q = Δvar_q, Δvar_T = Δvar_T, cov_geom = cov_geom)
end

"""
    mathsanity_Pi_q_Pi_T(qq_turb, TT_turb, dz, dq_dz, dtheta_dz, dT_dtheta)

Dimensionless **geometry / turbulence** ratios `Π_q = Δvar_q / qq_turb`, `Π_T = Δvar_T / TT_turb` with
`Δvar_* = (1/12) Δz² (∂·/∂z)²` as in `mathsanity_sgs_quad_moments_with_geometry`.

In standardized `(z_q,z_T)` with linear profiles, a layer `δz ∈ [-Δz/2,Δz/2]` maps to the chord from
`−½(R_q,R_T)` to `+½(R_q,R_T)` with `R_q = (Δz·∂q/∂z)/σ_q`, `R_T = (Δz·∂T/∂z)/σ_T` (same `Δz` and
`σ` as the turbulent SGS panel). Optional scale `s = √(Π_q² + Π_T²)` summarizes both channels.
"""
function mathsanity_Pi_q_Pi_T(
    qq_turb::FT,
    TT_turb::FT,
    dz::FT,
    dq_dz::FT,
    dtheta_dz::FT,
    dT_dtheta::FT,
) where {FT <: AbstractFloat}
    twelfth = one(FT) / 12
    dqdz_sq = dq_dz^2
    dTdz_sq = dT_dtheta^2 * dtheta_dz^2
    Δvar_q = twelfth * dz^2 * dqdz_sq
    Δvar_T = twelfth * dz^2 * dTdz_sq
    Π_q = qq_turb > zero(FT) ? Δvar_q / qq_turb : FT(Inf)
    Π_T = TT_turb > zero(FT) ? Δvar_T / TT_turb : FT(Inf)
    s = sqrt(Π_q^2 + Π_T^2)
    return (; Π_q = Π_q, Π_T = Π_T, s = s, Δvar_q = Δvar_q, Δvar_T = Δvar_T)
end

"""
Effective correlation as a function of turbulent `ρ_param` and `(Π_q,Π_T)` is already returned by
`mathsanity_sgs_quad_moments_with_geometry`; this alias documents the **split** (turbulent vs geometric
numerator, total σ in the denominator). Prefer calling `mathsanity_sgs_quad_moments_with_geometry` directly.
"""
function mathsanity_rho_eff_from_geometry(
    qq_turb::FT,
    TT_turb::FT,
    ρ_param::FT,
    dz::FT,
    dq_dz::FT,
    dtheta_dz::FT,
    dT_dtheta::FT;
    ε::FT = FT(1e-10),
) where {FT <: AbstractFloat}
    _, _, ρ_eff, _ = mathsanity_sgs_quad_moments_with_geometry(
        qq_turb,
        TT_turb,
        ρ_param,
        dz,
        dq_dz,
        dtheta_dz,
        dT_dtheta;
        ε = ε,
    )
    return ρ_eff
end

"""
    scale_sgs_moments_geometry(qt_var_ref, thetali_var_ref, cov_ref, dz_ref, dz_target, dqt_dz, dthetali_dz; ref_includes_gradient=true)

Transfer variances / covariance between resolutions using **only** the subcell linear
`(1/12) Δz² (∂c/∂z)²` geometry (same structure as the user’s draft `scale_sgs_moments`).

- `ref_includes_gradient == true`: reference moments already include geometry at `dz_ref`; swap to `dz_target` via `(1/12)(dz_target² - dz_ref²)` factor on gradient terms.
- `false`: reference is “pure turbulence”; add `(1/12) dz_target²` geometry only.

Covariance increment uses `geom_factor * dqt_dz * dthetali_dz`. Variances floored at 0;
covariance clamped by Cauchy–Schwarz to keep PSD.
"""
function scale_sgs_moments_geometry(
    qt_var_ref::FT,
    thetali_var_ref::FT,
    cov_ref::FT,
    dz_ref::FT,
    dz_target::FT,
    dqt_dz::FT,
    dthetali_dz::FT;
    ref_includes_gradient::Bool = true,
) where {FT <: AbstractFloat}
    geom_factor = if ref_includes_gradient
        (one(FT) / FT(12.0)) * (dz_target^2 - dz_ref^2)
    else
        (one(FT) / FT(12.0)) * dz_target^2
    end
    qt_var_target = max(zero(FT), qt_var_ref + geom_factor * dqt_dz^2)
    thetali_var_target = max(zero(FT), thetali_var_ref + geom_factor * dthetali_dz^2)
    cov_target = cov_ref + geom_factor * (dqt_dz * dthetali_dz)
    max_cov = sqrt(qt_var_target * thetali_var_target)
    cov_target = clamp(cov_target, -max_cov, max_cov)
    return qt_var_target, thetali_var_target, cov_target
end
