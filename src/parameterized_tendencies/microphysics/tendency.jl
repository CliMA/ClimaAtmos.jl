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
#   Model        | Nothing | PrognosticEDMFX
#   -------------|---------|----------------
#   DryModel     | no-op   | no-op (fallback)
#   0M           | ✓       | ✓
#   1M           | ✓       | ✓
#   2M           | ✓       | ✓
#   2MP3         | ✓       | —
#
# For 1M/2M in EDMF modes, separate source terms for the environment (⁰ suffix)
# and each updraft (ʲs suffix) are area-weighted and accumulated.

import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

"""
    microphysics_tendency!(Yₜ, Y, p, t, microphysics_model, turbconv_model)

Apply cached microphysics tendencies to the grid-mean (and EDMF subdomain) state.

All expensive work (SGS quadrature, CloudMicrophysics process rates, limiters) is
done in `set_microphysics_tendency_cache!`; this function only applies the cached
specific tendencies with the appropriate density and EDMF area weighting. The cached
tendencies are assumed to be already limited and are positive when representing a
source of the corresponding tracer.

Dispatches on `(microphysics_model, turbconv_model)`:

  - `DryModel`: no-op.
  - `EquilibriumMicrophysics0M`: precipitation removal; sinks of `ρq_tot`, `ρ`,
    and `ρe_tot`.
  - `NonEquilibriumMicrophysics1M`: sources for `ρq_lcl`, `ρq_icl`, `ρq_rai`, `ρq_sno`.
  - `NonEquilibriumMicrophysics2M`: sources for `ρq_lcl`, `ρn_lcl`, `ρq_rai`,
    `ρn_rai`, and `ρq_icl`.
  - `NonEquilibriumMicrophysics2MP3` (non-EDMF only): 2M sources plus P3 collision
    sources.

Without EDMF, the cached grid-mean tendency `p.precomputed.ᶜmp_tendency` (which
may already include SGS fluctuations via quadrature) is weighted by `Y.c.ρ`. With
`PrognosticEDMFX`, the grid-mean tendency is the area-weighted sum of the
environment (`ᶜmp_tendency⁰`, weighted by `ρa⁰`) and updraft (`ᶜmp_tendencyʲs`,
weighted by `ρaʲ`) contributions, and each updraft's own prognostic variables are
also updated with its unweighted specific tendencies.

Cache fields read: `p.precomputed.ᶜmp_tendency` (non-EDMF), `ᶜmp_tendency⁰`,
`ᶜmp_tendencyʲs`, `ᶜTʲs` (PrognosticEDMFX), `ᶜScoll` (2M+P3).

Returns `nothing`; mutates `Yₜ` in place. See `docs/src/microphysics.md` for the
scheme formulations.
"""
microphysics_tendency!(Yₜ, Y, p, t, ::DryModel, _) = nothing

#####
##### 0-Moment Microphysics
#####

"""
    microphysics_tendency!(Yₜ, Y, p, t, ::EquilibriumMicrophysics0M, _)

Apply the 0-moment precipitation-removal tendency to the grid mean.

The cached `dq_tot_dt` (a sink, ≤ 0 after limiting) removes water from `ρq_tot`
together with a matching sink of total air mass `ρ`. The energy sink of `ρe_tot`
is the water mass sink times the cached `e_tot_hlpr`, the specific energy of the
precipitating condensate (see `e_tot_0M_precipitation_sources_helper`).
"""
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

"""
    microphysics_tendency!(Yₜ, Y, p, t, ::EquilibriumMicrophysics0M, ::PrognosticEDMFX)

Apply 0-moment precipitation removal for PrognosticEDMFX.

The environment tendency `ᶜmp_tendency⁰` (weighted by `ρa⁰`) and each updraft
tendency `ᶜmp_tendencyʲs` (weighted by `ρaʲ`) accumulate into the grid-mean
`ρq_tot`, `ρ`, and `ρe_tot` sinks. Each updraft's prognostic state is also
updated: `ρaʲ` loses the removed mass, `qₜʲ` receives the `(1 - qₜʲ)` conversion
from the mass-removal rate to a specific-humidity tendency, and `mseʲ` changes by
the removed specific energy relative to the updraft internal energy at `ᶜTʲs`.
"""
function microphysics_tendency!(Yₜ, Y, p, t,
    ::EquilibriumMicrophysics0M, turbconv_model::PrognosticEDMFX,
)
    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰, ᶜTʲs) = p.precomputed
    thp = CAP.thermodynamics_params(p.params)
    n = n_mass_flux_subdomains(turbconv_model)

    # Environment contribution to grid mean tendency
    ρ_dq_tot_dt⁰ = @. lazy(ᶜmp_tendency⁰.dq_tot_dt * ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    @. Yₜ.c.ρq_tot += ρ_dq_tot_dt⁰
    @. Yₜ.c.ρ += ρ_dq_tot_dt⁰
    @. Yₜ.c.ρe_tot += ρ_dq_tot_dt⁰ * ᶜmp_tendency⁰.e_tot_hlpr
    # Updraft contribution to...
    for j in 1:n
        # ... grid mean tendency ...
        @. Yₜ.c.ρq_tot += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_tot_dt
        @. Yₜ.c.ρ += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_tot_dt
        @. Yₜ.c.ρe_tot +=
            Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_tot_dt *
            ᶜmp_tendencyʲs.:($$j).e_tot_hlpr
        # ... and updraft tendency
        @. Yₜ.c.sgsʲs.:($$j).ρa += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_tot_dt
        @. Yₜ.c.sgsʲs.:($$j).q_tot +=
            ᶜmp_tendencyʲs.:($$j).dq_tot_dt *
            (1 - Y.c.sgsʲs.:($$j).q_tot)
        @. Yₜ.c.sgsʲs.:($$j).mse +=
            ᶜmp_tendencyʲs.:($$j).dq_tot_dt * (
                ᶜmp_tendencyʲs.:($$j).e_tot_hlpr -
                TD.internal_energy(thp, ᶜTʲs.:($$j))
            )
    end
    return nothing
end

#####
##### 1-Moment Microphysics
#####

"""
    microphysics_tendency!(Yₜ, Y, p, t, ::NonEquilibriumMicrophysics1M, _)
    microphysics_tendency!(Yₜ, Y, p, t, ::NonEquilibriumMicrophysics1M, ::PrognosticEDMFX)

Apply 1-moment tendencies to the four condensate/precipitation mass tracers
`ρq_lcl`, `ρq_icl`, `ρq_rai`, `ρq_sno`.

The 1M process rates exchange water between vapor, cloud condensate, and
precipitation within the total; no direct sources are applied to `ρq_tot`, `ρ`,
or `ρe_tot` here. With PrognosticEDMFX, the environment contribution (weighted by
`ρa⁰`) and updraft contributions (weighted by `ρaʲ`) accumulate into the
grid-mean tracers, and each updraft's prognostic `q_lclʲ`, `q_iclʲ`, `q_raiʲ`,
`q_snoʲ` receive the unweighted specific tendencies.
"""
function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics1M, _,
)
    (; ᶜmp_tendency) = p.precomputed
    @. Yₜ.c.ρq_lcl += Y.c.ρ * ᶜmp_tendency.dq_lcl_dt
    @. Yₜ.c.ρq_icl += Y.c.ρ * ᶜmp_tendency.dq_icl_dt
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜmp_tendency.dq_rai_dt
    @. Yₜ.c.ρq_sno += Y.c.ρ * ᶜmp_tendency.dq_sno_dt
    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics1M, turbconv_model::PrognosticEDMFX,
)
    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰) = p.precomputed

    # Contribution to grid mean tendency from environment
    # ᶜmp_tendency⁰ is computed from environmental microphysics variables
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    @. Yₜ.c.ρq_lcl += ᶜρa⁰ * ᶜmp_tendency⁰.dq_lcl_dt
    @. Yₜ.c.ρq_icl += ᶜρa⁰ * ᶜmp_tendency⁰.dq_icl_dt
    @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜmp_tendency⁰.dq_rai_dt
    @. Yₜ.c.ρq_sno += ᶜρa⁰ * ᶜmp_tendency⁰.dq_sno_dt

    # Contribution from updraft microphysics to grid mean and updraft tendency
    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_lcl += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_lcl_dt
        @. Yₜ.c.ρq_icl += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_icl_dt
        @. Yₜ.c.ρq_rai += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.ρq_sno += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_sno_dt

        @. Yₜ.c.sgsʲs.:($$j).q_lcl += ᶜmp_tendencyʲs.:($$j).dq_lcl_dt
        @. Yₜ.c.sgsʲs.:($$j).q_icl += ᶜmp_tendencyʲs.:($$j).dq_icl_dt
        @. Yₜ.c.sgsʲs.:($$j).q_rai += ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.sgsʲs.:($$j).q_sno += ᶜmp_tendencyʲs.:($$j).dq_sno_dt
    end
    return nothing
end

#####
##### 2-Moment Microphysics
#####

"""
    microphysics_tendency!(Yₜ, Y, p, t, ::NonEquilibriumMicrophysics2M, _)
    microphysics_tendency!(Yₜ, Y, p, t, ::NonEquilibriumMicrophysics2M, ::PrognosticEDMFX)

Apply 2-moment warm-rain tendencies: mass and number sources for cloud liquid
(`ρq_lcl`, `ρn_lcl`) and rain (`ρq_rai`, `ρn_rai`), plus the ice mass source
`ρq_icl` from `dq_ice_dt`.

The number tendencies `dn_*_dt` are specific (per unit air mass), so they are
density-weighted like the mass tendencies. With warm-rain-only 2M parameters
`dq_ice_dt` is identically zero (SB2006 has no ice processes); it is nonzero only
when the 2M parameter set includes an ice (P3) scheme. There are no snow sources.
With PrognosticEDMFX, environment and updraft contributions are area-weighted
into the grid mean as for the 1M scheme, and each updraft's prognostic `q_lclʲ`,
`n_lclʲ`, `q_raiʲ`, `n_raiʲ`, `q_iclʲ` receive the unweighted specific tendencies.
"""
function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics2M, _,
)
    (; ᶜmp_tendency) = p.precomputed
    @. Yₜ.c.ρq_lcl += Y.c.ρ * ᶜmp_tendency.dq_lcl_dt
    @. Yₜ.c.ρn_lcl += Y.c.ρ * ᶜmp_tendency.dn_lcl_dt
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜmp_tendency.dq_rai_dt
    @. Yₜ.c.ρn_rai += Y.c.ρ * ᶜmp_tendency.dn_rai_dt
    @. Yₜ.c.ρq_icl += Y.c.ρ * ᶜmp_tendency.dq_ice_dt
    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics2M, turbconv_model::PrognosticEDMFX,
)
    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰) = p.precomputed

    # Contribution to grid mean tendency from environment
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    @. Yₜ.c.ρq_lcl += ᶜρa⁰ * ᶜmp_tendency⁰.dq_lcl_dt
    @. Yₜ.c.ρn_lcl += ᶜρa⁰ * ᶜmp_tendency⁰.dn_lcl_dt
    @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜmp_tendency⁰.dq_rai_dt
    @. Yₜ.c.ρn_rai += ᶜρa⁰ * ᶜmp_tendency⁰.dn_rai_dt
    @. Yₜ.c.ρq_icl += ᶜρa⁰ * ᶜmp_tendency⁰.dq_ice_dt

    # Contribution from updraft microphysics to grid mean and updraft tendency
    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_lcl += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_lcl_dt
        @. Yₜ.c.ρn_lcl += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dn_lcl_dt
        @. Yₜ.c.ρq_rai += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.ρn_rai += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dn_rai_dt
        @. Yₜ.c.ρq_icl += Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_ice_dt

        @. Yₜ.c.sgsʲs.:($$j).q_lcl += ᶜmp_tendencyʲs.:($$j).dq_lcl_dt
        @. Yₜ.c.sgsʲs.:($$j).n_lcl += ᶜmp_tendencyʲs.:($$j).dn_lcl_dt
        @. Yₜ.c.sgsʲs.:($$j).q_rai += ᶜmp_tendencyʲs.:($$j).dq_rai_dt
        @. Yₜ.c.sgsʲs.:($$j).n_rai += ᶜmp_tendencyʲs.:($$j).dn_rai_dt
        @. Yₜ.c.sgsʲs.:($$j).q_icl += ᶜmp_tendencyʲs.:($$j).dq_ice_dt
    end
end

"""
    microphysics_tendency!(Yₜ, Y, p, t, ::NonEquilibriumMicrophysics2MP3, ::Nothing)

Apply 2M warm-rain tendencies, then add the P3 (cold-phase) collision sources.

First delegates to the `NonEquilibriumMicrophysics2M` method, then adds the
cached P3 collision rates `p.precomputed.ᶜScoll` to `ρq_lcl`, `ρq_rai`, `ρn_lcl`,
`ρn_rai`, `ρq_rim`, `ρq_icl`, and `ρb_rim`. The specific rates `∂ₜq_c` and
`∂ₜq_r` are density-weighted; the remaining `ᶜScoll` rates are volumetric and
added directly. Not available with EDMF.
"""
function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilibriumMicrophysics2MP3, ::Nothing,
)
    (; ᶜScoll) = p.precomputed

    # 2 moment scheme (warm)
    microphysics_tendency!(Yₜ, Y, p, t, NonEquilibriumMicrophysics2M(), nothing)

    # P3 scheme (cold) - collisions
    @. Yₜ.c.ρq_lcl += Y.c.ρ * ᶜScoll.∂ₜq_c
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜScoll.∂ₜq_r
    @. Yₜ.c.ρn_lcl += ᶜScoll.∂ₜN_c
    @. Yₜ.c.ρn_rai += ᶜScoll.∂ₜN_r
    @. Yₜ.c.ρq_rim += ᶜScoll.∂ₜL_rim
    @. Yₜ.c.ρq_icl += ᶜScoll.∂ₜL_ice
    @. Yₜ.c.ρb_rim += ᶜScoll.∂ₜB_rim
    return nothing
end
