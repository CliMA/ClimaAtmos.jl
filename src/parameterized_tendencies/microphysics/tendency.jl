# ============================================================================
# Unified Microphysics Tendencies
# ============================================================================
#
# Single entry point for all microphysics tendency calculations.  All expensive
# computations (SGS quadrature, BMT calls, limiters) are performed in
# `set_microphysics_tendency_cache!` and stored in `p.precomputed`.  The
# functions here apply those cached tendencies with the appropriate density
# and EDMF area weighting.
#
# Dispatch matrix (microphysics_model × turbconv_model):
#
#   Model        | Nothing | DiagnosticEDMFX | PrognosticEDMFX
#   -------------|---------|-----------------|----------------
#   DryModel     | no-op   | no-op (fallback)| no-op (fallback)
#   0M           | ✓       | ✓               | ✓
#   1M           | ✓       | ✓               | ✓
#   2M           | ✓       | error           | ✓
#   2MP3         | ✓       | —               | —
#
# For 1M/2M in EDMF modes, separate source terms for the environment (⁰ suffix)
# and each updraft (ʲs suffix) are area-weighted and accumulated.

import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

"""
    microphysics_tendency!(Yₜ, Y, p, t, microphysics_model, turbconv_model)

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
    (; ᶜmp_tendency, ᶜ∂tendency_∂q_tot) = p.precomputed
    ρ_dq_tot_dt = @. lazy(
        Y.c.ρ * microphysics_tendency_model(
            ᶜmp_tendency.dq_tot_dt,
            ᶜ∂tendency_∂q_tot,
            Y.c.ρq_tot,
            Y.c.ρ,
        ),
    )

    @. Yₜ.c.ρq_tot += ρ_dq_tot_dt
    @. Yₜ.c.ρ += ρ_dq_tot_dt
    @. Yₜ.c.ρe_tot += ρ_dq_tot_dt * ᶜmp_tendency.e_tot_hlpr
    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::EquilibriumMicrophysics0M, turbconv_model::DiagnosticEDMFX,
)
    (; ᶜmp_tendency, ᶜmp_tendencyʲs, ᶜρaʲs) = p.precomputed
    (; ᶜ∂tendency_∂q_tot) = p.precomputed
    n = n_mass_flux_subdomains(turbconv_model)

    # Environment contibution to grid mean tendency
    ρ_dq_tot_dt = @. lazy(
        microphysics_tendency_model(
            ᶜmp_tendency.dq_tot_dt,
            ᶜ∂tendency_∂q_tot,
            Y.c.ρq_tot,
            Y.c.ρ,
        ) *
        ρa⁰(Y.c.ρ, ᶜρaʲs, turbconv_model),
    )
    @. Yₜ.c.ρq_tot += ρ_dq_tot_dt
    @. Yₜ.c.ρ += ρ_dq_tot_dt
    @. Yₜ.c.ρe_tot += ρ_dq_tot_dt * ᶜmp_tendency.e_tot_hlpr
    # Updraft contribution to grid mean tendency
    # (Sources in updrafts are applied in the diagnostic EDMF integral loop)
    for j in 1:n
        ρ_dq_tot_dtʲ = @. lazy(ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_tot_dt)
        @. Yₜ.c.ρq_tot += ρ_dq_tot_dtʲ
        @. Yₜ.c.ρ += ρ_dq_tot_dtʲ
        @. Yₜ.c.ρe_tot += ρ_dq_tot_dtʲ * ᶜmp_tendencyʲs.:($$j).e_tot_hlpr
    end
    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::EquilibriumMicrophysics0M, turbconv_model::PrognosticEDMFX,
)
    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰, ᶜTʲs) = p.precomputed
    (; ᶜ∂tendency_∂q_totʲs, ᶜ∂tendency_∂q_tot, ᶜq_tot_safe⁰) = p.precomputed
    thp = CAP.thermodynamics_params(p.params)
    n = n_mass_flux_subdomains(turbconv_model)

    # Environment contribution to grid mean tendency
    ρ_dq_tot_dt⁰ = @. lazy(
        microphysics_tendency_model(
            ᶜmp_tendency⁰.dq_tot_dt,
            ᶜ∂tendency_∂q_tot,
            ᶜq_tot_safe⁰,
        ) *
        ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model),
    )
    @. Yₜ.c.ρq_tot += ρ_dq_tot_dt⁰
    @. Yₜ.c.ρ += ρ_dq_tot_dt⁰
    @. Yₜ.c.ρe_tot += ρ_dq_tot_dt⁰ * ᶜmp_tendency⁰.e_tot_hlpr
    # Updraft contribution to...
    for j in 1:n
        dq_tot_dtʲ = @. lazy(
            microphysics_tendency_model(
                ᶜmp_tendencyʲs.:($$j).dq_tot_dt,
                ᶜ∂tendency_∂q_totʲs.:($$j),
                Y.c.sgsʲs.:($$j).q_tot,
            ),
        )
        # ... grid mean tendency ...
        @. Yₜ.c.ρq_tot += Y.c.sgsʲs.:($$j).ρa * dq_tot_dtʲ
        @. Yₜ.c.ρ += Y.c.sgsʲs.:($$j).ρa * dq_tot_dtʲ
        @. Yₜ.c.ρe_tot +=
            Y.c.sgsʲs.:($$j).ρa * dq_tot_dtʲ * ᶜmp_tendencyʲs.:($$j).e_tot_hlpr
        # ... and updraft tendency
        @. Yₜ.c.sgsʲs.:($$j).ρa += Y.c.sgsʲs.:($$j).ρa * dq_tot_dtʲ
        @. Yₜ.c.sgsʲs.:($$j).q_tot +=
            dq_tot_dtʲ *
            (1 - Y.c.sgsʲs.:($$j).q_tot)
        @. Yₜ.c.sgsʲs.:($$j).mse +=
            dq_tot_dtʲ * (
                ᶜmp_tendencyʲs.:($$j).e_tot_hlpr -
                TD.internal_energy(thp, ᶜTʲs.:($$j))
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
    (; ᶜmp_tendency, ᶜmp_derivative) = p.precomputed
    @. Yₜ.c.ρq_liq +=
        Y.c.ρ * microphysics_tendency_model(
            ᶜmp_tendency.dq_lcl_dt,
            ᶜmp_derivative.∂tendency_∂q_lcl,
            Y.c.ρq_liq,
            Y.c.ρ,
        )
    @. Yₜ.c.ρq_ice +=
        Y.c.ρ * microphysics_tendency_model(
            ᶜmp_tendency.dq_icl_dt,
            ᶜmp_derivative.∂tendency_∂q_icl,
            Y.c.ρq_ice,
            Y.c.ρ,
        )
    @. Yₜ.c.ρq_rai +=
        Y.c.ρ * microphysics_tendency_model(
            ᶜmp_tendency.dq_rai_dt,
            ᶜmp_derivative.∂tendency_∂q_rai,
            Y.c.ρq_rai,
            Y.c.ρ,
        )
    @. Yₜ.c.ρq_sno +=
        Y.c.ρ * microphysics_tendency_model(
            ᶜmp_tendency.dq_sno_dt,
            ᶜmp_derivative.∂tendency_∂q_sno,
            Y.c.ρq_sno,
            Y.c.ρ,
        )
    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics1M, turbconv_model::DiagnosticEDMFX,
)
    (; ᶜmp_tendencyʲs, ᶜmp_tendency, ᶜmp_derivative) = p.precomputed
    (; ᶜρaʲs) = p.precomputed

    n = n_mass_flux_subdomains(turbconv_model)
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, p.precomputed.ᶜρaʲs, turbconv_model))

    # Environment contribution to grid mean tendency
    @. Yₜ.c.ρq_liq +=
        ᶜρa⁰ * microphysics_tendency_model(
            ᶜmp_tendency.dq_lcl_dt,
            ᶜmp_derivative.∂tendency_∂q_lcl,
            Y.c.ρq_liq,
            Y.c.ρ,
        )
    @. Yₜ.c.ρq_ice +=
        ᶜρa⁰ * microphysics_tendency_model(
            ᶜmp_tendency.dq_icl_dt,
            ᶜmp_derivative.∂tendency_∂q_icl,
            Y.c.ρq_ice,
            Y.c.ρ,
        )
    @. Yₜ.c.ρq_rai +=
        ᶜρa⁰ * microphysics_tendency_model(
            ᶜmp_tendency.dq_rai_dt,
            ᶜmp_derivative.∂tendency_∂q_rai,
            Y.c.ρq_rai,
            Y.c.ρ,
        )
    @. Yₜ.c.ρq_sno +=
        ᶜρa⁰ * microphysics_tendency_model(
            ᶜmp_tendency.dq_sno_dt,
            ᶜmp_derivative.∂tendency_∂q_sno,
            Y.c.ρq_sno,
            Y.c.ρ,
        )

    # Updraft contribution to grid mean tendency
    # (Sources in updrafts are applied in the diagnostic EDMF integral loop)
    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_liq += ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_lcl_dt
        @. Yₜ.c.ρq_ice += ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_icl_dt
        @. Yₜ.c.ρq_rai += ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.ρq_sno += ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_sno_dt
    end
    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics1M, turbconv_model::PrognosticEDMFX,
)
    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰) = p.precomputed
    (; ᶜmp_derivativeʲs, ᶜmp_derivative) = p.precomputed

    # Contribution to grid mean tendency from environment
    # ᶜmp_derivative is computed based on environmental microphysics variables
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    ᶜq_liq⁰ = ᶜspecific_env_value(@name(q_liq), Y, p)
    ᶜq_ice⁰ = ᶜspecific_env_value(@name(q_ice), Y, p)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    @. Yₜ.c.ρq_liq +=
        ᶜρa⁰ * microphysics_tendency_model(
            ᶜmp_tendency⁰.dq_lcl_dt,
            ᶜmp_derivative.∂tendency_∂q_lcl,
            ᶜq_liq⁰,
        )
    @. Yₜ.c.ρq_ice +=
        ᶜρa⁰ * microphysics_tendency_model(
            ᶜmp_tendency⁰.dq_icl_dt,
            ᶜmp_derivative.∂tendency_∂q_icl,
            ᶜq_ice⁰,
        )
    @. Yₜ.c.ρq_rai +=
        ᶜρa⁰ * microphysics_tendency_model(
            ᶜmp_tendency⁰.dq_rai_dt,
            ᶜmp_derivative.∂tendency_∂q_rai,
            ᶜq_rai⁰,
        )
    @. Yₜ.c.ρq_sno +=
        ᶜρa⁰ * microphysics_tendency_model(
            ᶜmp_tendency⁰.dq_sno_dt,
            ᶜmp_derivative.∂tendency_∂q_sno,
            ᶜq_sno⁰,
        )

    # Contribution from updraft microphysics to grid mean and updraft tendency
    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_liq +=
            Y.c.sgsʲs.:($$j).ρa * microphysics_tendency_model(
                ᶜmp_tendencyʲs.:($$j).dq_lcl_dt,
                ᶜmp_derivativeʲs.:($$j).∂tendency_∂q_lcl,
                Y.c.sgsʲs.:($$j).q_liq,
            )
        @. Yₜ.c.ρq_ice +=
            Y.c.sgsʲs.:($$j).ρa * microphysics_tendency_model(
                ᶜmp_tendencyʲs.:($$j).dq_icl_dt,
                ᶜmp_derivativeʲs.:($$j).∂tendency_∂q_icl,
                Y.c.sgsʲs.:($$j).q_ice,
            )
        @. Yₜ.c.ρq_rai +=
            Y.c.sgsʲs.:($$j).ρa * microphysics_tendency_model(
                ᶜmp_tendencyʲs.:($$j).dq_rai_dt,
                ᶜmp_derivativeʲs.:($$j).∂tendency_∂q_rai,
                Y.c.sgsʲs.:($$j).q_rai,
            )
        @. Yₜ.c.ρq_sno +=
            Y.c.sgsʲs.:($$j).ρa * microphysics_tendency_model(
                ᶜmp_tendencyʲs.:($$j).dq_sno_dt,
                ᶜmp_derivativeʲs.:($$j).∂tendency_∂q_sno,
                Y.c.sgsʲs.:($$j).q_sno,
            )

        @. Yₜ.c.sgsʲs.:($$j).q_liq += microphysics_tendency_model(
            ᶜmp_tendencyʲs.:($$j).dq_lcl_dt,
            ᶜmp_derivativeʲs.:($$j).∂tendency_∂q_lcl,
            Y.c.sgsʲs.:($$j).q_liq,
        )
        @. Yₜ.c.sgsʲs.:($$j).q_ice += microphysics_tendency_model(
            ᶜmp_tendencyʲs.:($$j).dq_icl_dt,
            ᶜmp_derivativeʲs.:($$j).∂tendency_∂q_icl,
            Y.c.sgsʲs.:($$j).q_ice,
        )
        @. Yₜ.c.sgsʲs.:($$j).q_rai += microphysics_tendency_model(
            ᶜmp_tendencyʲs.:($$j).dq_rai_dt,
            ᶜmp_derivativeʲs.:($$j).∂tendency_∂q_rai,
            Y.c.sgsʲs.:($$j).q_rai,
        )
        @. Yₜ.c.sgsʲs.:($$j).q_sno += microphysics_tendency_model(
            ᶜmp_tendencyʲs.:($$j).dq_sno_dt,
            ᶜmp_derivativeʲs.:($$j).∂tendency_∂q_sno,
            Y.c.sgsʲs.:($$j).q_sno,
        )
    end
    return nothing
end

#####
##### 2-Moment Microphysics
#####

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics2M, _,
)
    (; ᶜmp_tendency) = p.precomputed
    @. Yₜ.c.ρq_liq += Y.c.ρ * ᶜmp_tendency.dq_lcl_dt
    @. Yₜ.c.ρn_liq += Y.c.ρ * ᶜmp_tendency.dn_lcl_dt
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜmp_tendency.dq_rai_dt
    @. Yₜ.c.ρn_rai += Y.c.ρ * ᶜmp_tendency.dn_rai_dt
    @. Yₜ.c.ρq_ice += Y.c.ρ * ᶜmp_tendency.dq_ice_dt
    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics2M, ::DiagnosticEDMFX,
)
    error("NonEquilibriumMicrophysics2M is not implemented for DiagnosticEDMFX")
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics2M, turbconv_model::PrognosticEDMFX,
)
    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰) = p.precomputed

    # Contribution to grid mean tendency from environment
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    @. Yₜ.c.ρq_liq += ᶜρa⁰ * ᶜmp_tendency⁰.dq_lcl_dt
    @. Yₜ.c.ρn_liq += ᶜρa⁰ * ᶜmp_tendency⁰.dn_lcl_dt
    @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜmp_tendency⁰.dq_rai_dt
    @. Yₜ.c.ρn_rai += ᶜρa⁰ * ᶜmp_tendency⁰.dn_rai_dt
    @. Yₜ.c.ρq_ice += ᶜρa⁰ * ᶜmp_tendency⁰.dq_ice_dt

    # Contribution from updraft microphysics to grid mean and updraft tendency
    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_liq += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_lcl_dt
        @. Yₜ.c.ρn_liq += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dn_lcl_dt
        @. Yₜ.c.ρq_rai += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.ρn_rai += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dn_rai_dt
        @. Yₜ.c.ρq_ice += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_ice_dt

        @. Yₜ.c.sgsʲs.:($$j).q_liq += ᶜmp_tendencyʲs.:($$j).dq_lcl_dt
        @. Yₜ.c.sgsʲs.:($$j).n_liq += ᶜmp_tendencyʲs.:($$j).dn_lcl_dt
        @. Yₜ.c.sgsʲs.:($$j).q_rai += ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.sgsʲs.:($$j).n_rai += ᶜmp_tendencyʲs.:($$j).dn_rai_dt
        @. Yₜ.c.sgsʲs.:($$j).q_ice += ᶜmp_tendencyʲs.:($$j).dq_ice_dt
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
