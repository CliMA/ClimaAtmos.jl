"""
    microphysics_tendency!(Yₜ, Y, p, t, microphysics_model, turbconv_model)

Single entry point for all microphysics tendency calculations.
The tendency is based on `mp_tendency` values stored in microphysics cache.
Assumes that all limiting was done in the cache, and that
`mp_tendency` is defined to be positive when representing a source.

When running without EDMF, the tendency is computed based on
the grid mean properties, optionally including the SGS fluctuations
as an integral over quardature points.

In EDMF modes, grid mean tendency is equal to the area weighted sum of
sub-domain contributions. The environment contribution can be optionally
computed including SGS fluctuations as an integral over quadrature points.

In `PrognosticEDMFX` mode, both grid-mean and EDMF tendencies
are modified in place.
In `DiagnosticEDMFX` mode, updraft sources are already computed and applied
to updrafts inside the diagnostic vertical integral loop, and the
`microphysics_tendency` only modifies the grid-mean tendency.

Arguments:
- `Yₜ`: The tendency state vector.
- `Y`: The current state vector.
- `p`: The cache, containing precomputed quantities and parameters.
- `t`: The current simulation time.
- `microphysics_model` (e.g., `EquilibriumMicrophysics0M`,
  `NonEquilibriumMicrophysics1M`, `NonEquilibriumMicrophysics2M`).
- `turbconv_model`: (e.g., `PrognosticEDMFX`, `DiagnosticEDMFX`).

Returns: `nothing`, modifies `Yₜ` in place.
"""
microphysics_tendency!(Yₜ, Y, p, t, ::DryModel, _) = nothing

#####
##### 0-Moment Microphysics
#####

function microphysics_tendency!(Yₜ, Y, p, t,
    ::EquilibriumMicrophysics0M, _,
)
    (; ᶜmp_tendency) = p.precomputed
    ρ_dq_tot_dt = @. lazy(Y.c.ρ * ᶜmp_tendency.dq_tot_dt)

    @. Yₜ.c.ρq_tot += ρ_dq_tot_dt
    @. Yₜ.c.ρ += ρ_dq_tot_dt
    @. Yₜ.c.ρe_tot += ρ_dq_tot_dt * ᶜmp_tendency.e_tot_hlpr
    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::EquilibriumMicrophysics0M, ::DiagnosticEDMFX,
)
    # TODO ᶜmp_tendency⁰ not needed in diagnostic EDMF configuration
    (; ᶜmp_tendency, ᶜmp_tendencyʲ, ᶜρaʲs) = p.precomputed
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    # Environment contibution to grid mean tendency
    ρ_dq_tot_dt = @. lazy(
        ᶜmp_tendency.dq_tot_dt * ρa⁰(Y.c.ρ, ᶜρaʲs, p.atmos.turbconv_model)
    )
    @. ᶜYₜ.c.ρq_tot += ρ_dq_tot_dt
    @. ᶜYₜ.c.ρ += ρ_dq_tot_dt
    @. ᶜYₜ.c.ρe_tot += ρ_dq_tot_dt * ᶜmp_tendency.e_tot_hlpr
    # Updraft contribution to grid mean tendency
    # (Sources in updrafts are applied in the diagnostic EDMF integral loop)
    for j in 1:n
        ρ_dq_tot_dt = @. lazy(* ᶜρaʲs.:($$j) * ᶜmp_tendencyʲ.:($$j).dq_tot_dt)
        @. ᶜYₜ.c.ρq_tot += ρ_dq_tot_dt
        @. ᶜYₜ.c.ρ += ρ_dq_tot_dt
        @. ᶜYₜ.c.ρe_tot += ρ_dq_tot_dt * ᶜmp_tendencyʲ.:($$j).e_tot_hlpr
    end
    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::EquilibriumMicrophysics0M, ::PrognosticEDMFX,
)
    # TODO ᶜmp_tendency not needed in prognostic EDMF configuration
    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰, ᶜTʲs) = p.precomputed
    thp = CAP.thermodynamics_params(p.params)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    # Environment contribution to grid mean tendency
    ρ_dq_tot_dt⁰ = @. lazy(
        ᶜmp_tendency⁰.dq_tot_dt * ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model)
    )
    @. Yₜ.c.ρq_tot += ρ_dq_tot_dt⁰
    @. Yₜ.c.ρ += ρ_dρ_tot_dt⁰
    @. Yₜ.c.ρe_tot += ρ_dq_tot_dt⁰ * ᶜmp_tendency⁰.e_tot_hlpr
    # Updraft contribution to...
    for j in 1:n
        ρ_dq_tot_dtʲ = @. lazy(
            ᶜmp_tendencyʲs.:($$j).dq_tot_dt * Y.c.sgsʲs.:($$j).ρa
        )
        # ... grid mean tendency ...
        @. ᶜYₜ.c.ρq_tot += ρ_dq_tot_dtʲ
        @. ᶜYₜ.c.ρ += ρ_dq_tot_dtʲ
        @. ᶜYₜ.c.ρe_tot += ρ_dq_tot_dtʲ * ᶜmp_tendencyʲs.:($$j).e_tot_hlpr
        # ... and updraft tendency
        @. Yₜ.c.sgsʲs.:($$j).ρa += ρ_dq_tot_dt
        @. Yₜ.c.sgsʲs.:($$j).q_tot += ᶜmp_tendencyʲs.:($$j).dq_tot_dt *
            (1 - Y.c.sgsʲs.:($$j).q_tot)
        @. Yₜ.c.sgsʲs.:($$j).mse += ᶜmp_tendencyʲs.:($$j).dq_tot_dt * (
            ᶜmp_tendencyʲs.:($$j).e_tot_hlpr -
            TD.internal_energy(thermo_params, ᶜTʲs.:($$j))
        )
    end
    return nothing
end

#####
##### 1-Moment Microphysics
#####

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics1M, _,
)
    (; ᶜmp_tendency) = p.precomputed
    @. Yₜ.c.ρq_liq += Y.c.ρ * ᶜmp_tendency.dq_lcl_dt
    @. Yₜ.c.ρq_ice += Y.c.ρ * ᶜmp_tendency.dq_icl_dt
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜmp_tendency.dq_rai_dt
    @. Yₜ.c.ρq_sno += Y.c.ρ * ᶜmp_tendency.dq_sno_dt
    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics1M, ::DiagnosticEDMFX,
)
    # TODO ᶜmp_tendency⁰ not needed in diagnostic EDMF configuration
    (; ᶜmp_tendencyʲs, ᶜmp_tendency) = p.precomputed
    (; ᶜρaʲs) = p.precomputed

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, p.precomputed.ᶜρaʲs, p.atmos.turbconv_model))

    # Environment contribution to grid mean tendency
    @. Yₜ.c.ρq_liq += ᶜρa⁰ * ᶜmp_tendency.dq_lcl_dt
    @. Yₜ.c.ρq_ice += ᶜρa⁰ * ᶜmp_tendency.dq_icl_dt
    @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜmp_tendency.dq_rai_dt
    @. Yₜ.c.ρq_sno += ᶜρa⁰ * ᶜmp_tendency.dq_sno_dt

    # Updraft contribution to grid mean tendency
    # (Sources in updrafts are applied in the diagnostic EDMF integral loop)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_liq += ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_lcl_dt
        @. Yₜ.c.ρq_ice += ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_icl_dt
        @. Yₜ.c.ρq_rai += ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.ρq_sno += ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_sno_dt
    end
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics1M, turbconv_model::PrognosticEDMFX,
)
    # TODO ᶜmp_tendency not needed in prognostic EDMF configuration
    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰) = p.precomputed

    # Contribution to grid mean tendency from environment
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    @. Yₜ.c.ρq_liq += ᶜρa⁰ * ᶜmp_tendency⁰.dq_lcl_dt
    @. Yₜ.c.ρq_ice += ᶜρa⁰ * ᶜmp_tendency⁰.dq_icl_dt
    @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜmp_tendency⁰.dq_rai_dt
    @. Yₜ.c.ρq_sno += ᶜρa⁰ * ᶜmp_tendency⁰.dq_sno_dt

    # Contribution from updraft microphysics to grid mean and updraft tendency
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_liq += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_liq_dt
        @. Yₜ.c.ρq_ice += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_ice_dt
        @. Yₜ.c.ρq_rai += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.ρq_sno += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_sno_dt

        @. Yₜ.c.sgsʲs.:($$j).q_liq += ᶜmp_tendencyʲs.:($$j).dq_liq_dt
        @. Yₜ.c.sgsʲs.:($$j).q_ice += ᶜmp_tendencyʲs.:($$j).dq_ice_dt
        @. Yₜ.c.sgsʲs.:($$j).q_rai += ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.sgsʲs.:($$j).q_sno += ᶜmp_tendencyʲs.:($$j).dq_sno_dt
    end
    return nothing
end

#####
##### 2-Moment and P3 Ice Scheme Microphysics
#####

function microphysics_tendency!(
    Yₜ, Y, p, t, ::NonEquilibriumMicrophysics2M, _,
)
    (; ᶜmp_tendency) = p.precomputed
    @. Yₜ.c.ρq_liq += Y.c.ρ * ᶜmp_tendency.dq_lcl_dt
    @. Yₜ.c.ρn_liq += Y.c.ρ * ᶜmp_tendency.dn_lcl_dt
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜmp_tendency.dq_rai_dt
    @. Yₜ.c.ρn_rai += Y.c.ρ * ᶜmp_tendency.dn_rai_dt
    @. Yₜ.c.ρq_ice += Y.c.ρ * ᶜmp_tendency.dq_ice_dt
    @. Yₜ.c.ρq_sno += Y.c.ρ * ᶜmp_tendency.dq_sno_dt
    return nothing
end

# 2M with DiagnosticEDMFX
function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics2M, ::DiagnosticEDMFX,
)
    error("NonEquilibriumMicrophysics2M is not implemented for DiagnosticEDMFX")
end

# 2M with PrognosticEDMFX
function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics2M, turbconv_model::PrognosticEDMFX,
)
    # TODO ᶜmp_tendency not needed in prognostic EDMF configuration
    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰) = p.precomputed

    # Contribution to grid mean tendency from environment
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model))
    @. Yₜ.c.ρq_liq += ᶜρa⁰ * ᶜmp_tendency⁰.dq_lcl_dt
    @. Yₜ.c.ρn_liq += ᶜρa⁰ * ᶜmp_tendency⁰.dn_lcl_dt
    @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜmp_tendency⁰.dq_rai_dt
    @. Yₜ.c.ρn_rai += ᶜρa⁰ * ᶜmp_tendency⁰.dn_rai_dt
    @. Yₜ.c.ρq_ice += ᶜρa⁰ * ᶜmp_tendency⁰.dq_ice_dt
    @. Yₜ.c.ρq_sno += ᶜρa⁰ * ᶜmp_tendency⁰.dq_sno_dt

    # Contribution from updraft microphysics to grid mean and updraft tendency
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_liq += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_lcl_dt
        @. Yₜ.c.ρn_liq += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dn_lcl_dt
        @. Yₜ.c.ρq_rai += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.ρn_rai += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dn_rai_dt
        @. Yₜ.c.ρq_ice += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_ice_dt
        @. Yₜ.c.ρq_sno += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_sno_dt

        @. Yₜ.c.sgsʲs.:($$j).q_liq += ᶜmp_tendencyʲs.:($$j).dq_lcl_dt
        @. Yₜ.c.sgsʲs.:($$j).n_liq += ᶜmp_tendencyʲs.:($$j).dn_lcl_dt
        @. Yₜ.c.sgsʲs.:($$j).q_rai += ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.sgsʲs.:($$j).n_rai += ᶜmp_tendencyʲs.:($$j).dn_rai_dt
        @. Yₜ.c.sgsʲs.:($$j).q_ice += ᶜmp_tendencyʲs.:($$j).dq_ice_dt
        @. Yₜ.c.sgsʲs.:($$j).q_sno += ᶜmp_tendencyʲs.:($$j).dq_sno_dt
    end
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics2MP3, ::Nothing,
)
    (; ᶜScoll) = p.precomputed

    # 2 moment scheme (warm)
    microphysics_tendency!(Yₜ, Y, p, t, NonEquilibriumMicrophysics2M(), nothing)

    # P3 scheme (cold) - collisions
    @. Yₜ.c.ρq_liq += Y.c.ρ * ᶜScoll.∂ₜq_c
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜScoll.∂ₜq_r
    @. Yₜ.c.ρn_liq += ᶜScoll.∂ₜN_c
    @. Yₜ.c.ρn_rai += ᶜScoll.∂ₜN_r
    @. Yₜ.c.ρq_rim += ᶜScoll.∂ₜL_rim
    @. Yₜ.c.ρq_ice += ᶜScoll.∂ₜL_ice
    @. Yₜ.c.ρb_rim += ᶜScoll.∂ₜB_rim

    return nothing
end
