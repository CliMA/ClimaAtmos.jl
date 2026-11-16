Base.broadcastable(flag::DerivativeFlag) = tuple(flag)

"""
    use_moist_wave_jacobian(p)

Whether the `f.u₃` pressure-gradient Jacobian blocks blend frozen-condensate
derivatives with saturation-equilibrium derivatives: active only for
`EquilibriumMicrophysics0M`. Non-equilibrium microphysics (`1M`, `2M`, `2M+P3`)
uses the residual-consistent frozen-condensate Jacobian.
"""
@inline use_moist_wave_jacobian(p) =
    p.atmos.microphysics_model isa EquilibriumMicrophysics0M

@inline _sgs_cf_flag(::MLCloud, _) = UseDerivative()
@inline _sgs_cf_flag(::QuadratureCloud, ::Nothing) = IgnoreDerivative()
@inline _sgs_cf_flag(::QuadratureCloud, ::GridMeanSGS) = IgnoreDerivative()
@inline _sgs_cf_flag(
    ::QuadratureCloud,
    ::SGSQuadrature{<:Any, <:Any, <:Any, <:GridMeanSGS},
) = IgnoreDerivative()
@inline _sgs_cf_flag(::QuadratureCloud, ::SGSQuadrature) = UseDerivative()
@inline _sgs_cf_flag(_, _) = IgnoreDerivative()

@generated _has_cloud_fraction(::P) where {P} =
    hasfield(P, :ᶜcloud_fraction) ? :(true) : :(false)

@inline _sgs_cf_flag_for_cache(::IgnoreDerivative, _, _) = IgnoreDerivative()
@inline _sgs_cf_flag_for_cache(::UseDerivative, atmos, precomputed) =
    _has_cloud_fraction(precomputed) ?
    _sgs_cf_flag(atmos.cloud_model, atmos.sgs_quadrature) : IgnoreDerivative()

"""
    moist_jacobian_cloud_fraction(thermo_params, T, ρ, q_tot, cf, sgs_cf_flag)

Return the saturated fraction `f ∈ [0, 1]` used to blend frozen and
saturation-equilibrium pressure derivatives. Uses `clamp(cf, 0, 1)` when
`use_derivative(sgs_cf_flag)` holds and `cf > 0`, and falls back to the
grid-mean saturation step `ifelse(q_tot > q_sat, 1, 0)` otherwise.
"""
@inline function moist_jacobian_cloud_fraction(
    thermo_params,
    T,
    ρ,
    q_tot,
    cf,
    sgs_cf_flag::DerivativeFlag,
)
    FT = typeof(T)
    if use_derivative(sgs_cf_flag) && cf > zero(FT)
        return clamp(FT(cf), zero(FT), one(FT))
    end
    q_sat = TD.q_vap_saturation(thermo_params, T, ρ)
    return ifelse(q_tot > q_sat, one(FT), zero(FT))
end

"""
    in_cloud_q_tot(thermo_params, T, ρ, q_tot)

Compute the in-cloud total water specific humidity `max(max(0, q_tot), q_sat)`
at which `TD.∂p_∂T_sat_ρ` and `TD.∂e_int_∂T_sat_ρ` are evaluated with
`Val(false)` under `EquilibriumMicrophysics0M`. Setting `q_tot_cld ≥ q_sat`
keeps `q_c ≥ 0` in the mixed-phase latent-heat term `L_f q_c dλ/dT` of
`C_v,sat` in subsaturated grid-mean cells with `f > 0`.
"""
@inline function in_cloud_q_tot(thermo_params, T, ρ, q_tot)
    q_sat = TD.q_vap_saturation(thermo_params, T, ρ)
    return max(max(zero(q_sat), oftype(q_sat, q_tot)), q_sat)
end

"""
    saturated_∂p_∂ρq_tot(thermo_params, T, κ_sat)

Closed-form saturated derivative `(∂p / ∂(ρq_tot))|_{ρ, ρe_tot}` along the
`Thermodynamics` phase-equilibrium manifold:

    ∂p_∂ρq_tot_sat = -R_d T - κ_sat (e_c(T, λ) - e_d(T)),

where `λ = TD.liquid_fraction_ramp(thermo_params, T)` and
`e_c(T, λ) = λ e_l(T) + (1 - λ) e_i(T)`. At fixed `(ρ, T)`, vapor density
`ρ_v,sat(T) = p_v,sat(T) / (R_v T)` is independent of `ρq_tot`, so added water
enters condensate while reducing dry-air density `ρ_d = ρ - ρq_tot`.
"""
@inline function saturated_∂p_∂ρq_tot(thermo_params, T, κ_sat)
    R_d = TD.Parameters.R_d(thermo_params)
    λ = TD.liquid_fraction_ramp(thermo_params, T)
    e_c =
        λ * TD.internal_energy_liquid(thermo_params, T) +
        (one(λ) - λ) * TD.internal_energy_ice(thermo_params, T)
    e_d = TD.internal_energy_dry(thermo_params, T)
    return -R_d * T - κ_sat * (e_c - e_d)
end

# TODO: remove or move into the test suite. The model does not call this
# function; only `moist_jacobian_tests.jl` uses it, as a closed-form reference.
"""
    saturated_pressure_derivatives(thermo_params, T, ρ, q_tot_cld, K = 0, Φ = 0)

Closed-form fixed-`(ρ, ρe_tot, ρq_tot)` pressure derivatives along the
saturation-equilibrium manifold, evaluated at `q_tot_cld ≥ q_sat(T, ρ)` via
`TD.∂p_∂T_sat_ρ(..., Val(false))` and `TD.∂e_int_∂T_sat_ρ(..., Val(false))`.

Returns `(; κ_sat, ∂p_∂ρq_tot_sat, ∂p_∂ρ_sat, C_v_sat)`, where
`κ_sat = (∂p / ∂(ρe_tot))|_{ρ, ρq_tot}`.
"""
@inline function saturated_pressure_derivatives(
    thermo_params,
    T,
    ρ,
    q_tot_cld,
    K = zero(T),
    Φ = zero(T),
)
    R_d = TD.Parameters.R_d(thermo_params)
    cv_d = TD.Parameters.cv_d(thermo_params)
    cp_d = TD.Parameters.cp_d(thermo_params)
    T_0 = TD.Parameters.T_0(thermo_params)

    ∂p_∂T_sat = TD.∂p_∂T_sat_ρ(thermo_params, T, ρ, q_tot_cld, Val(false))
    C_v_sat = TD.∂e_int_∂T_sat_ρ(thermo_params, T, ρ, q_tot_cld, Val(false))
    κ_sat = ∂p_∂T_sat / (ρ * C_v_sat)

    ∂p_∂ρq_tot_sat = saturated_∂p_∂ρq_tot(thermo_params, T, κ_sat)
    ∂p_∂ρ_sat = κ_sat * (cp_d * T_0 - K - Φ) + (R_d - κ_sat * cv_d) * T

    return (; κ_sat, ∂p_∂ρq_tot_sat, ∂p_∂ρ_sat, C_v_sat)
end

@inline function saturated_kappa(thermo_params, T, ρ, q_tot, f)
    FT = typeof(T)
    iszero(f) && return zero(FT)
    q_tot_cld = in_cloud_q_tot(thermo_params, T, ρ, q_tot)
    ∂p_∂T_sat = TD.∂p_∂T_sat_ρ(thermo_params, T, ρ, q_tot_cld, Val(false))
    C_v_sat = TD.∂e_int_∂T_sat_ρ(thermo_params, T, ρ, q_tot_cld, Val(false))
    return ∂p_∂T_sat / (ρ * C_v_sat)
end

@inline blend_linear(x_frozen, x_sat, f) =
    iszero(f) ? x_frozen :
    (isone(f) ? x_sat : (one(f) - f) * x_frozen + f * x_sat)

@inline function blended_∂p_∂ρq_tot(
    thermo_params,
    T,
    ∂p_∂ρq_tot_frozen,
    κ_sat,
    f,
)
    iszero(f) && return ∂p_∂ρq_tot_frozen
    ∂p_∂ρq_tot_sat = saturated_∂p_∂ρq_tot(thermo_params, T, κ_sat)
    return blend_linear(∂p_∂ρq_tot_frozen, ∂p_∂ρq_tot_sat, f)
end

struct MoistJacobianParams{TP, SF <: DerivativeFlag}
    thermo_params::TP
    sgs_cf_flag::SF
end
Base.broadcastable(m::MoistJacobianParams) = tuple(m)

@inline _moist_cf(mp::MoistJacobianParams, T, ρ, q_tot, cf) =
    moist_jacobian_cloud_fraction(
        mp.thermo_params,
        T,
        ρ,
        q_tot,
        cf,
        mp.sgs_cf_flag,
    )

@inline _sat_kappa_0m(mp::MoistJacobianParams, T, ρ, q_tot, f) =
    saturated_kappa(mp.thermo_params, T, ρ, q_tot, f)

@inline _blended_dp_dρq(mp::MoistJacobianParams, T, dp_frozen, κ_sat, f) =
    blended_∂p_∂ρq_tot(mp.thermo_params, T, dp_frozen, κ_sat, f)

function _compute_moist_wave_jacobian_scratch!(
    p,
    Y,
    thermo_params,
    sgs_cf_flag::DerivativeFlag,
)
    (; params) = p
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    FT = Spaces.undertype(axes(Y.c))
    cv_d = FT(CAP.cv_d(params))
    Δcv_v = FT(CAP.cv_v(params)) - cv_d
    T_0 = FT(CAP.T_0(params))
    R_d = FT(CAP.R_d(params))
    R_v = FT(CAP.R_v(params))
    ΔR_v = R_v - R_d
    e_int_v0 = FT(CAP.e_int_v0(params))
    zero_FT = zero(FT)

    moist_params = MoistJacobianParams(thermo_params, sgs_cf_flag)
    ᶜρ = Y.c.ρ
    (ᶜkappa_eff, ᶜf, ᶜdp_dρq_tot) =
        (p.scratch.ᶜtemp_scalar, p.scratch.ᶜtemp_scalar_2, p.scratch.ᶜtemp_scalar_3)

    if use_derivative(sgs_cf_flag) && _has_cloud_fraction(p.precomputed)
        ᶜcf = p.precomputed.ᶜcloud_fraction
        @. ᶜf = _moist_cf(moist_params, ᶜT, ᶜρ, ᶜq_tot_nonneg, ᶜcf)
    else
        @. ᶜf = _moist_cf(moist_params, ᶜT, ᶜρ, ᶜq_tot_nonneg, zero_FT)
    end

    @. ᶜdp_dρq_tot =
        _sat_kappa_0m(moist_params, ᶜT, ᶜρ, ᶜq_tot_nonneg, ᶜf)

    ᶜkappa_m = @. lazy(
        TD.gas_constant_air(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) /
        TD.cv_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice),
    )
    @. ᶜkappa_eff = blend_linear(ᶜkappa_m, ᶜdp_dρq_tot, ᶜf)
    ᶜdp_frozen = @. lazy(
        ᶜkappa_m * (-e_int_v0 - R_d * T_0 - Δcv_v * (ᶜT - T_0)) + ΔR_v * ᶜT,
    )
    @. ᶜdp_dρq_tot = _blended_dp_dρq(
        moist_params,
        ᶜT,
        ᶜdp_frozen,
        ᶜdp_dρq_tot,
        ᶜf,
    )
    return nothing
end

@inline _moist_scratch_fields(::UseDerivative, scratch) =
    (scratch.ᶜtemp_scalar, scratch.ᶜtemp_scalar_2, scratch.ᶜtemp_scalar_3)
@inline _moist_scratch_fields(::IgnoreDerivative, _) = ()
