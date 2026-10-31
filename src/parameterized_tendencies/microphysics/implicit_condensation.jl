import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Parameters as CMP

# Implicit 1M `q_lcl` condensation / evaporation.
#
# `microphysics_1m_params` carries `cloud_liquid_formation = nothing`, so
# CM's `BulkMicrophysicsTendencies` never averages condensation. The
# implicit source here evaluates CM's instantaneous rate
# `conv_q_vap_to_q_lcl` at each SGS node (same condensate placement as
# the explicit `Microphysics1MEvaluator`), so Newton can linearize about
# the current `Y`. Updrafts are single-point. Dispatch is on
# `microphysics_1m_implicit_params.processes.cloud_liquid_formation`.

"""
    _cm_condensation_rate(mp, thp, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno)

CM's `conv_q_vap_to_q_lcl` at this state, packed for broadcast. Dispatches
on `mp.processes.cloud_liquid_formation`, so `::Nothing` returns 0 and
`::CloudLiquidFormation` returns the vapor-relaxation rate with `τ_relax`
and `T_hom` from `mp.process_params.cloud_liquid_formation`.
"""
@inline function _cm_condensation_rate(
    mp,
    thp,
    ρ,
    T,
    q_tot,
    q_lcl,
    q_icl,
    q_rai,
    q_sno,
)
    FT = typeof(ρ)
    micro = (;
        q_tot,
        q_lcl = max(zero(FT), q_lcl),
        q_icl = max(zero(FT), q_icl),
        q_rai = max(zero(FT), q_rai),
        q_sno = max(zero(FT), q_sno),
    )
    thermo = (; ρ, T)
    return CMNe.conv_q_vap_to_q_lcl(
        mp.processes.cloud_liquid_formation, mp, thp, micro, thermo,
    )
end

"""
    LiquidCondensationEvaluator{MP, TPS, FT}

Per-cell node evaluator for the SGS-averaged implicit condensation rate.
Reuses the Lagrange-multiplier placement of `Microphysics1MEvaluator`
(`sgs_local_condensate`) plus the `λ_eq`-based equilibrium reconstruction
of the node total water (`q_tot_eff`), and the precipitation-fraction
placement (`β_precip`, `shaft_frac`) that scales `q_rai`, `q_sno` by
`φ_p` and shifts the node total water so the vapour (and the condensate
reconstruction) is unchanged by the placement. Driven by
`_sum_over_conditional_sgs` under the conditional-moment node rule.
"""
struct LiquidCondensationEvaluator{MP, TPS, FT}
    mp::MP
    tps::TPS
    ρ::FT
    λ_lagrange::FT
    λ_eq::FT
    mu_S::FT
    α::FT
    λ::FT
    ξ_liq::FT
    ξ_ice::FT
    q_lcl::FT
    q_icl::FT
    q_rai::FT
    q_sno::FT
    β_precip::FT
    shaft_frac::FT
end

"""
    (ev::LiquidCondensationEvaluator)(T_hat, q_tot_hat, flag_p)

Place condensate at `(T_hat, q_tot_hat)` with the Lagrange closure, build
the equilibrium-reconstructed `q_tot_eff` from `λ_eq`, apply the
precip-fraction placement (`φ_p` from `sgs_placement_factor(β_precip, flag_p, shaft_frac)`) to scale `q_rai, q_sno` and shift `q_tot_eff` by
`(φ_p − 1)·q_precip`, then return CM's instantaneous condensation rate
at the placed node. On cloudy equilibrium nodes (`q_c_eq_hat > 0`) this
makes `S → 0` when `q_c → q_c_eq`; on subsaturated nodes the raw cloud
excess is preserved so below-cloud evaporation is unchanged. `flag_p`
is the shaft indicator passed by `sum_over_conditional_nodes` (1 on
shaft nodes, 0 off-shaft); `β_precip = 0` short-circuits the placement.
"""
@inline function (ev::LiquidCondensationEvaluator)(T_hat, q_tot_hat, flag_p)
    FT = typeof(ev.ρ)
    q_tot_hat = max(zero(FT), q_tot_hat)
    q_sat_hat = TD.q_vap_saturation(_unwrapped(ev.tps), T_hat, ev.ρ)
    S′_hat = q_tot_hat - q_sat_hat - ev.mu_S
    shifted_excess = max(zero(FT), ev.λ_lagrange + ev.α * S′_hat)
    q_lcl_hat, q_icl_hat = sgs_local_condensate(
        ev.λ, shifted_excess, ev.ξ_liq, ev.ξ_ice, ev.q_lcl, ev.q_icl,
    )
    # Equilibrium-reconstructed q_tot: on cloudy equilibrium nodes use
    # q_c_eq (unattenuated — λ_eq has α = 1), on subsaturated nodes keep
    # the raw cloud excess so below-cloud evaporation of q_rai / q_sno
    # is driven by the real subsaturation.
    q_c_eq_hat = max(zero(FT), ev.λ_eq + S′_hat)
    q_precip = ev.q_rai + ev.q_sno
    S_raw_cloud = q_tot_hat - q_precip - q_sat_hat
    S_eff_cloud = ifelse(
        q_c_eq_hat > zero(FT),
        q_c_eq_hat,
        min(zero(FT), S_raw_cloud),
    )
    q_tot_eff = max(zero(FT), q_sat_hat + S_eff_cloud + q_precip)
    # Precipitation-fraction placement: scale q_rai, q_sno by φ_p and
    # shift q_tot by (φ_p − 1)·q_precip so the node vapour (and the
    # condensate reconstruction above) is unchanged by the placement.
    q_rai_node, q_sno_node, q_tot_node =
        if ev.β_precip > zero(FT)
            φ_p = sgs_placement_factor(ev.β_precip, flag_p, ev.shaft_frac)
            shift = (φ_p - one(FT)) * q_precip
            (ev.q_rai * φ_p, ev.q_sno * φ_p, max(zero(FT), q_tot_eff + shift))
        else
            (ev.q_rai, ev.q_sno, q_tot_eff)
        end
    return _cm_condensation_rate(
        ev.mp, ev.tps, ev.ρ, T_hat, q_tot_node,
        q_lcl_hat, q_icl_hat, q_rai_node, q_sno_node,
    )
end

"""
    _sum_cond_over_sgs(mp, thp, ρ, T, q_lcl, q_icl, q_rai, q_sno,
                      T′T′, q′q′, corr_Tq, λ_lagrange, λ_eq, mu_S, σ_S, α,
                      λ, ξ_liq, ξ_ice, β_precip, precip_frac,
                      precip_frac_floor, sgs_quad)

SGS average of CM's condensation rate through `_sum_over_conditional_sgs`:
same conditional-moment rule as `_sum_1m_over_sgs`. The shaft edge
`S_star = sgs_precip_shaft_threshold(max(precip_frac, precip_frac_floor), σ_S)` is active when `β_precip > 0`; the `LiquidCondensationEvaluator`
is built with the shared helper's rescaled `(λ_lagrange, α)`, the raw
`λ_eq` (unattenuated, `α = 1`), and the `shaft_frac` returned by the
driver. `λ_eq` is used both to position the equilibrium cloud edge in
the conditional rule and inside the evaluator to reconstruct the node
`q_tot_eff` toward the SGS truncated-Gaussian equilibrium (`S → 0` when
`q_c → q_c_eq`); `β_precip` enables the precip-fraction placement in
the evaluator.
"""
@inline function _sum_cond_over_sgs(
    mp,
    thp,
    ρ,
    T,
    q_lcl,
    q_icl,
    q_rai,
    q_sno,
    T′T′,
    q′q′,
    corr_Tq,
    λ_lagrange,
    λ_eq,
    mu_S,
    σ_S,
    α,
    λ,
    ξ_liq,
    ξ_ice,
    β_precip,
    precip_frac,
    precip_frac_floor,
    sgs_quad,
)
    FT = typeof(ρ)
    q_lcl_nn = max(zero(FT), q_lcl)
    q_icl_nn = max(zero(FT), q_icl)
    q_rai_nn = max(zero(FT), q_rai)
    q_sno_nn = max(zero(FT), q_sno)
    # Shaft threshold on the centred excess: the moistest `a_p` of the PDF,
    # with `a_p` bounded below by `precip_frac_floor`.
    a_p = max(FT(precip_frac), FT(precip_frac_floor))
    σ_S_ft = FT(σ_S)
    S_star = sgs_precip_shaft_threshold(a_p, σ_S_ft)
    β_p = FT(β_precip)
    λ_eq_ft = FT(λ_eq)
    mu_S_ft = FT(mu_S)
    λ_ft = FT(λ)
    ξ_liq_ft = FT(ξ_liq)
    ξ_ice_ft = FT(ξ_ice)
    return _sum_over_conditional_sgs(
        (λ_scaled, α_scaled, shaft_frac) -> LiquidCondensationEvaluator(
            mp, thp, ρ, λ_scaled, λ_eq_ft, mu_S_ft, α_scaled,
            λ_ft, ξ_liq_ft, ξ_ice_ft, q_lcl_nn, q_icl_nn, q_rai_nn, q_sno_nn,
            β_p, shaft_frac,
        ),
        thp, ρ, T, mu_S_ft, σ_S_ft, FT(λ_lagrange), λ_eq_ft, FT(α),
        q_lcl_nn + q_icl_nn, T′T′, q′q′, corr_Tq, β_p, S_star, sgs_quad,
    )
end

"""
    _sum_cond_over_sgs(
        mp, thp, sgs_quad, ρ, T, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno,
        T′T′, q′q′, corr_Tq, sgs_moments, α, precip_frac,
        opts::SGSMicrophysicsOptions,
    )

Packed form of `_sum_cond_over_sgs`: computes `λ` and `mu_S` from the mean
state and forwards `sgs_moments` and `opts` to the positional form so the
tendency broadcast stays within Julia's specialized `Broadcast._getindex`
argument limit.
"""
@inline function _sum_cond_over_sgs(
    mp,
    thp,
    sgs_quad,
    ρ,
    T,
    q_tot_nonneg,
    q_lcl,
    q_icl,
    q_rai,
    q_sno,
    T′T′,
    q′q′,
    corr_Tq,
    sgs_moments,
    α,
    precip_frac,
    opts::SGSMicrophysicsOptions,
)
    FT = typeof(ρ)
    λ = TD.liquid_fraction(thp, T, max(q_lcl, zero(FT)), max(q_icl, zero(FT)))
    # `mu_S = q_tot − q_sat(T, ρ)` centres the SGS `S′` perturbation.
    mu_S = q_tot_nonneg - TD.q_vap_saturation(thp, T, ρ)
    return _sum_cond_over_sgs(
        mp,
        thp,
        ρ,
        T,
        q_lcl,
        q_icl,
        q_rai,
        q_sno,
        T′T′,
        q′q′,
        corr_Tq,
        sgs_moments.λ_lagrange,
        sgs_moments.λ_eq,
        mu_S,
        sgs_moments.sigma_S,
        α,
        λ,
        opts.ξ_liq,
        opts.ξ_ice,
        opts.precip_incloud_fraction,
        precip_frac,
        opts.precip_frac_floor,
        sgs_quad,
    )
end

@inline function _grid_mean_sgs_cond_tendency(
    mp,
    thp,
    sgs_quad,
    opts::SGSMicrophysicsOptions,
    ρ,
    T,
    q_tot_nonneg,
    ρq_lcl,
    ρq_icl,
    ρq_rai,
    ρq_sno,
    T′T′,
    q′q′,
    corr_Tq,
    sgs_moments,
    α,
    precip_frac,
)
    return ρ * _sum_cond_over_sgs(
        mp,
        thp,
        sgs_quad,
        ρ,
        T,
        q_tot_nonneg,
        specific(ρq_lcl, ρ),
        specific(ρq_icl, ρ),
        specific(ρq_rai, ρ),
        specific(ρq_sno, ρ),
        T′T′,
        q′q′,
        corr_Tq,
        sgs_moments,
        α,
        precip_frac,
        opts,
    )
end

@inline function _env_tracers_for_cond(
    ρ,
    ρa_env,
    sgsʲs,
    ρq_lcl,
    ρq_icl,
    ρq_rai,
    ρq_sno,
    turbconv_model,
)
    q_lcl⁰ = specific(
        env_value(ρq_lcl, sgsʲ -> sgsʲ.ρa * sgsʲ.q_lcl, sgsʲs),
        ρa_env,
        ρq_lcl,
        ρ,
        turbconv_model,
    )
    q_icl⁰ = specific(
        env_value(ρq_icl, sgsʲ -> sgsʲ.ρa * sgsʲ.q_icl, sgsʲs),
        ρa_env,
        ρq_icl,
        ρ,
        turbconv_model,
    )
    q_rai⁰ = specific(
        env_value(ρq_rai, sgsʲ -> sgsʲ.ρa * sgsʲ.q_rai, sgsʲs),
        ρa_env,
        ρq_rai,
        ρ,
        turbconv_model,
    )
    q_sno⁰ = specific(
        env_value(ρq_sno, sgsʲ -> sgsʲ.ρa * sgsʲ.q_sno, sgsʲs),
        ρa_env,
        ρq_sno,
        ρ,
        turbconv_model,
    )
    return (q_lcl⁰, q_icl⁰, q_rai⁰, q_sno⁰)
end

@inline function _env_point_cond_tendency(
    mp,
    thp,
    turbconv_model,
    ρ,
    sgsʲs,
    p_air,
    T⁰,
    q_tot_nonneg⁰,
    q_liq⁰,
    q_ice⁰,
    ρq_lcl,
    ρq_icl,
    ρq_rai,
    ρq_sno,
)
    ρa_env = ρa⁰(ρ, sgsʲs, turbconv_model)
    ρ⁰ = TD.air_density(thp, T⁰, p_air, q_tot_nonneg⁰, q_liq⁰, q_ice⁰)
    q_lcl⁰, q_icl⁰, q_rai⁰, q_sno⁰ = _env_tracers_for_cond(
        ρ, ρa_env, sgsʲs, ρq_lcl, ρq_icl, ρq_rai, ρq_sno, turbconv_model,
    )
    return ρa_env * _cm_condensation_rate(
        mp, thp, ρ⁰, T⁰, q_tot_nonneg⁰, q_lcl⁰, q_icl⁰, q_rai⁰, q_sno⁰,
    )
end

@inline function _env_sgs_cond_tendency(
    mp,
    thp,
    sgs_quad,
    turbconv_model,
    opts::SGSMicrophysicsOptions,
    ρ,
    sgsʲs,
    p_air,
    T⁰,
    q_tot_nonneg⁰,
    q_liq⁰,
    q_ice⁰,
    ρq_lcl,
    ρq_icl,
    ρq_rai,
    ρq_sno,
    T′T′,
    q′q′,
    corr_Tq,
    sgs_moments,
    α,
    precip_frac,
)
    ρa_env = ρa⁰(ρ, sgsʲs, turbconv_model)
    ρ⁰ = TD.air_density(thp, T⁰, p_air, q_tot_nonneg⁰, q_liq⁰, q_ice⁰)
    q_lcl⁰, q_icl⁰, q_rai⁰, q_sno⁰ = _env_tracers_for_cond(
        ρ, ρa_env, sgsʲs, ρq_lcl, ρq_icl, ρq_rai, ρq_sno, turbconv_model,
    )
    return ρa_env * _sum_cond_over_sgs(
        mp,
        thp,
        sgs_quad,
        ρ⁰,
        T⁰,
        q_tot_nonneg⁰,
        q_lcl⁰,
        q_icl⁰,
        q_rai⁰,
        q_sno⁰,
        T′T′,
        q′q′,
        corr_Tq,
        sgs_moments,
        α,
        precip_frac,
        opts,
    )
end

"""
    _add_implicit_lcl_condensation_grid_mean!(Yₜ, Y, p)

Grid-mean 1M (no `PrognosticEDMFX`): add `ρ · <S_cond>_SGS` to
`Yₜ.c.ρq_lcl`. Dispatches on
`CAP.microphysics_1m_implicit_params(p.params).processes.cloud_liquid_formation`
— `::Nothing` is a no-op; `::CMP.CloudLiquidFormation` runs the
SGS-integrated source.
"""
@inline _add_implicit_lcl_condensation_grid_mean!(Yₜ, Y, p) =
    _add_implicit_lcl_condensation_grid_mean!(
        Yₜ,
        Y,
        p,
        CAP.microphysics_1m_implicit_params(p.params).processes.cloud_liquid_formation,
    )
@inline _add_implicit_lcl_condensation_grid_mean!(Yₜ, Y, p, ::Nothing) = nothing
function _add_implicit_lcl_condensation_grid_mean!(
    Yₜ,
    Y,
    p,
    ::CMP.CloudLiquidFormation,
)
    MatrixFields.has_field(Y, @name(c.ρq_lcl)) || return nothing
    thp = CAP.thermodynamics_params(p.params)
    mp = CAP.microphysics_1m_implicit_params(p.params)
    sgs_quad = p.atmos.sgs_quadrature
    if not_quadrature(sgs_quad)
        (; ᶜT, ᶜq_tot_nonneg) = p.precomputed
        ᶜq_lcl = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
        ᶜq_icl = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ))
        ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
        ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))
        @. Yₜ.c.ρq_lcl +=
            Y.c.ρ * _cm_condensation_rate(
                mp, thp, Y.c.ρ, ᶜT, ᶜq_tot_nonneg,
                ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
            )
    else
        (;
            ᶜT, ᶜq_tot_nonneg, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜsgs_moments,
            ᶜprecip_frac,
        ) = p.precomputed
        opts = sgs_microphysics_options(p.params)
        @. Yₜ.c.ρq_lcl += _grid_mean_sgs_cond_tendency(
            mp, thp, sgs_quad, opts, Y.c.ρ, ᶜT, ᶜq_tot_nonneg,
            Y.c.ρq_lcl, Y.c.ρq_icl, Y.c.ρq_rai, Y.c.ρq_sno,
            ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜsgs_moments, ᶜα, ᶜprecip_frac,
        )
    end
    return nothing
end

"""
    _add_implicit_lcl_condensation_prognostic_edmfx!(Yₜ, Y, p, turbconv_model)

`PrognosticEDMFX` branch:

  - **Env**: `ρa⁰ · <S_cond⁰>_SGS` via `_sum_cond_over_sgs` with env
    `(T⁰, q_tot⁰)` and env variances.
  - **Updraft `j`**: single-point `_cm_condensation_rate` at
    `(Tʲ, ρʲ, qʲ)`.

Condensate inputs are clamped non-negative inside `_cm_condensation_rate`
and `_sum_cond_over_sgs`; call sites do not need to pre-clip.

Dispatches on
`CAP.microphysics_1m_implicit_params(p.params).processes.cloud_liquid_formation`.
"""
@inline _add_implicit_lcl_condensation_prognostic_edmfx!(
    Yₜ,
    Y,
    p,
    turbconv_model,
) = _add_implicit_lcl_condensation_prognostic_edmfx!(
    Yₜ,
    Y,
    p,
    turbconv_model,
    CAP.microphysics_1m_implicit_params(p.params).processes.cloud_liquid_formation,
)
@inline _add_implicit_lcl_condensation_prognostic_edmfx!(
    Yₜ,
    Y,
    p,
    turbconv_model,
    ::Nothing,
) = nothing
function _add_implicit_lcl_condensation_prognostic_edmfx!(
    Yₜ,
    Y,
    p,
    turbconv_model,
    ::CMP.CloudLiquidFormation,
)
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    (; ᶜTʲs, ᶜρʲs, ᶜq_tot_nonnegʲs) = p.precomputed
    thp = CAP.thermodynamics_params(p.params)
    mp = CAP.microphysics_1m_implicit_params(p.params)
    sgs_quad = p.atmos.sgs_quadrature

    # Env contribution to grid-mean `ρq_lcl`.
    if not_quadrature(sgs_quad)
        @. Yₜ.c.ρq_lcl += _env_point_cond_tendency(
            mp, thp, turbconv_model, Y.c.ρ, Y.c.sgsʲs,
            ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰,
            Y.c.ρq_lcl, Y.c.ρq_icl, Y.c.ρq_rai, Y.c.ρq_sno,
        )
    else
        (; ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜsgs_moments, ᶜprecip_frac) =
            p.precomputed
        opts = sgs_microphysics_options(p.params)
        @. Yₜ.c.ρq_lcl += _env_sgs_cond_tendency(
            mp, thp, sgs_quad, turbconv_model, opts, Y.c.ρ, Y.c.sgsʲs,
            ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰,
            Y.c.ρq_lcl, Y.c.ρq_icl, Y.c.ρq_rai, Y.c.ρq_sno,
            ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜsgs_moments, ᶜα, ᶜprecip_frac,
        )
    end

    # Updrafts: single-point (no SGS distribution). Tracers are clipped inside
    # `_cm_condensation_rate`.
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        ᶜTʲ = ᶜTʲs.:($j)
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜq_tot_nonnegʲ = ᶜq_tot_nonnegʲs.:($j)
        sgsʲ = Y.c.sgsʲs.:($j)
        ᶜρaʲ = sgsʲ.ρa
        ᶜS_condʲ = @. lazy(
            _cm_condensation_rate(
                mp, thp, ᶜρʲ, ᶜTʲ, ᶜq_tot_nonnegʲ,
                sgsʲ.q_lcl, sgsʲ.q_icl, sgsʲ.q_rai, sgsʲ.q_sno,
            ),
        )
        @. Yₜ.c.ρq_lcl += ᶜρaʲ * ᶜS_condʲ
        @. Yₜ.c.sgsʲs.:($$j).q_lcl += ᶜS_condʲ
    end
    return nothing
end
