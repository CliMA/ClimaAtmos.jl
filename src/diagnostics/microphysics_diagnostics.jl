#=
This file is included in Diagnostics.jl

Microphysics 1-moment process and species tendency diagnostics.

Exposes three families of zero-allocation 3D diagnostics for
`NonEquilibriumMicrophysics1M`:
  1. Instantaneous potential process rates
     (`mp1m_S_*`, `mp1mup_S_*`, `mp1men_S_*`):
     `t = 0` un-attenuated rates from
     `BMT.bulk_microphysics_tendencies(InstantaneousVerbose(), ...)`.
  2. Substep-averaged process rates
     (`mp1m_la_S_*`, `mp1mup_la_S_*`, `mp1men_la_S_*`):
     time-averaged rates over `(dt, n_substeps)` from
     `BMT.bulk_microphysics_tendencies(LinearizedAverageVerbose(), ...)` for
     bulk processes plus `InstantaneousVerbose()` for implicit phase changes.
     In the updraft (without SGS quadrature), the species sum of `mp1mup_la_S_*`
     equals `mp1mup_dq_*_dt` to machine precision.
  3. Applied net species tendencies
     (`mp1m_dq_*_dt`, `mp1mup_dq_*_dt`, `mp1men_dq_*_dt`):
     per-species tendencies (`dq_lcl_dt`, `dq_icl_dt`, `dq_rai_dt`, `dq_sno_dt`)
     applied to `Yₜ` in `microphysics_tendency!` from
     `cache.precomputed.ᶜmp_tendency*` and
     `cache.precomputed.ᶜmp_tendency_implicit*` (with 8-node SGS conditional
     quadrature and precipitation-shaft placement in the environment).
=#

import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

# ---------------------------------------------------------------------------
# Helpers: evaluate InstantaneousVerbose or LinearizedAverageVerbose and extract
# one field by name. Because everything is @inline, the compiler dead-code-
# eliminates unused source terms inside the broadcast.
# ---------------------------------------------------------------------------
"""
    _mp1m_source_term(
        ::Val{F}, mp_bulk, mp_imp, tps, ρ, T, w, q_tot,
        q_lcl, q_icl, q_rai, q_sno,
    )

Evaluate the instantaneous (`t = 0` potential rate) 1-moment verbose tendencies
on both the bulk and implicit 1M parameter sets and return the sum of the source
term `F` from each.

The bulk and implicit sets are process-disjoint by construction
(`_split_formation`):

  - `implicit_microphysics: true` → bulk evaluates every process except
    `cloud_liquid_formation`, `cloud_ice_formation`,
    `rain_condensation_evaporation`, and `snow_deposition_sublimation`; implicit
    evaluates only those four.
  - `implicit_microphysics: false` → bulk evaluates every process; implicit has
    no active process, so its contribution to every source term is zero.

Either way, each process appears in exactly one parameter set, with no double
counting. Floors each condensate species (`q_lcl`, `q_icl`, `q_rai`, `q_sno`) at
zero and raises `q_tot` to at least their sum before calling `BMT`, matching the
non-quadrature `microphysics_tendencies_1m`.
"""
@inline function _mp1m_source_term(
    ::Val{F}, mp_bulk, mp_imp, tps, ρ, T, w, q_tot, q_lcl, q_icl, q_rai, q_sno,
) where {F}
    FT = typeof(ρ)
    q_lcl_pos = max(zero(FT), q_lcl)
    q_icl_pos = max(zero(FT), q_icl)
    q_rai_pos = max(zero(FT), q_rai)
    q_sno_pos = max(zero(FT), q_sno)
    q_tot_safe = max(
        zero(FT),
        q_tot,
        q_lcl_pos + q_icl_pos + q_rai_pos + q_sno_pos,
    )
    src_bulk = BMT.bulk_microphysics_tendencies(
        BMT.InstantaneousVerbose(),
        BMT.Microphysics1Moment(),
        mp_bulk,
        tps,
        ρ,
        T,
        w,
        q_tot_safe,
        q_lcl_pos,
        q_icl_pos,
        q_rai_pos,
        q_sno_pos,
    )
    src_imp = BMT.bulk_microphysics_tendencies(
        BMT.InstantaneousVerbose(),
        BMT.Microphysics1Moment(),
        mp_imp,
        tps,
        ρ,
        T,
        w,
        q_tot_safe,
        q_lcl_pos,
        q_icl_pos,
        q_rai_pos,
        q_sno_pos,
    )
    return getfield(src_bulk, F) + getfield(src_imp, F)
end

"""
    _mp1m_linearized_source_term(
        ::Val{F}, mp_bulk, mp_imp, tps, ρ, T, w, q_tot,
        q_lcl, q_icl, q_rai, q_sno, dt, nsubs,
    )

Evaluate the substep-averaged 1-moment verbose tendency on the bulk 1M parameter
set with `BMT.LinearizedAverageVerbose()` over `(dt, nsubs)` (clamping
condensate inputs non-negative as in `microphysics_tendencies_1m`) and add the
implicit 1M parameter set evaluated with `BMT.InstantaneousVerbose()`
(matching `update_implicit_1m_tendency!`).
"""
@inline function _mp1m_linearized_source_term(
    ::Val{F},
    mp_bulk,
    mp_imp,
    tps,
    ρ,
    T,
    w,
    q_tot,
    q_lcl,
    q_icl,
    q_rai,
    q_sno,
    dt,
    nsubs,
) where {F}
    FT = typeof(ρ)
    q_lcl_pos = max(zero(FT), q_lcl)
    q_icl_pos = max(zero(FT), q_icl)
    q_rai_pos = max(zero(FT), q_rai)
    q_sno_pos = max(zero(FT), q_sno)
    q_tot_safe = max(
        zero(FT),
        q_tot,
        q_lcl_pos + q_icl_pos + q_rai_pos + q_sno_pos,
    )
    src_bulk = BMT.bulk_microphysics_tendencies(
        BMT.LinearizedAverageVerbose(), BMT.Microphysics1Moment(),
        mp_bulk, tps, ρ, T, w,
        q_tot_safe, q_lcl_pos, q_icl_pos, q_rai_pos, q_sno_pos,
        dt, nsubs,
    )
    src_imp = BMT.bulk_microphysics_tendencies(
        BMT.InstantaneousVerbose(), BMT.Microphysics1Moment(),
        mp_imp, tps, ρ, T, w,
        q_tot_safe, q_lcl_pos, q_icl_pos, q_rai_pos, q_sno_pos,
    )
    return getfield(src_bulk, F) + getfield(src_imp, F)
end

@inline _mp1m_species_sum(::Val{F}, mp_bulk, mp_imp) where {F} =
    getfield(mp_bulk, F) + getfield(mp_imp, F)

# ---------------------------------------------------------------------------
# The 19 source terms produced by _microphysics_source_terms and the 4
# per-species net tendencies in `ᶜmp_tendency*`.
# ---------------------------------------------------------------------------
"""
    MP1M_SOURCE_TERMS

The nineteen 1-moment source terms exposed as diagnostics, as
`(field_symbol, description)` pairs.

Each entry is registered for the grid mean, the first updraft, and the
environment under both the instantaneous potential-rate prefixes (`mp1m_`,
`mp1mup_`, `mp1men_`) and the substep-averaged `LinearizedAverageVerbose`
prefixes (`mp1m_la_`, `mp1mup_la_`, `mp1men_la_`).
"""
const MP1M_SOURCE_TERMS = [
    (:S_phase_change_vap_lcl, "Vapor ↔ cloud liquid phase change"),
    (:S_phase_change_vap_icl, "Vapor ↔ cloud ice phase change"),
    (:S_acnv_lcl_rai, "Cloud liquid → rain autoconversion"),
    (:S_acnv_icl_sno, "Cloud ice → snow autoconversion"),
    (:S_accr_lcl_rai, "Cloud liquid + rain accretion"),
    (:S_accr_lcl_sno_cold, "Cloud liquid + snow accretion (cold)"),
    (:S_accr_lcl_sno_warm, "Cloud liquid + snow accretion (warm)"),
    (:S_accr_melt_lcl_sno, "Thermal melt of snow by warm cloud liquid"),
    (:S_accr_icl_rai, "Cloud ice + rain accretion (cold, ice → snow)"),
    (:S_accr_melt_icl_rai, "Cloud ice + rain accretion (warm, ice → rain)"),
    (:S_accr_freeze_icl_rai, "Rain frozen in ice-rain collision"),
    (:S_accr_icl_sno, "Cloud ice + snow accretion"),
    (:S_accr_rai_sno_cold, "Rain-snow accretion (cold, rain → snow)"),
    (:S_accr_melt_rai_sno, "Thermal melt of snow by warm rain"),
    (:S_phase_change_vap_rai, "Rain condensation/evaporation"),
    (:S_phase_change_vap_sno, "Snow deposition/sublimation"),
    (:S_melt_icl_lcl, "Cloud ice melt → cloud liquid"),
    (:S_freeze_lcl_icl, "Cloud liquid freezing → cloud ice"),
    (:S_melt_sno_rai, "Snow melt → rain"),
]

"""
    MP1M_SPECIES_TENDENCIES

The four applied per-species 1-moment microphysics net tendencies
(`dq_lcl_dt`, `dq_icl_dt`, `dq_rai_dt`, `dq_sno_dt`), registered under the
`mp1m_`, `mp1mup_`, and `mp1men_` prefixes.
"""
const MP1M_SPECIES_TENDENCIES = [
    (:dq_lcl_dt, "Applied net cloud liquid tendency"),
    (:dq_icl_dt, "Applied net cloud ice tendency"),
    (:dq_rai_dt, "Applied net rain tendency"),
    (:dq_sno_dt, "Applied net snow tendency"),
]

# ---------------------------------------------------------------------------
# Grid-mean compute functions (instantaneous & LinearizedAverageVerbose)
# ---------------------------------------------------------------------------
"""
    compute_mp1m_source(state, cache, ::Val{F})
    compute_mp1m_source(state, cache, time, F)

Evaluate the instantaneous (`t = 0` potential rate) 1-moment source term `F` on
the grid-mean state.
"""
function compute_mp1m_source(state, cache, ::Val{F}) where {F}
    cmp = CAP.microphysics_1m_params(cache.params)
    cmp_i = CAP.microphysics_1m_implicit_params(cache.params)
    thp = CAP.thermodynamics_params(cache.params)
    ᶜρ = state.c.ρ
    ᶜT = cache.precomputed.ᶜT
    ᶜw = @. lazy(w_component(Geometry.WVector(cache.precomputed.ᶜu)))
    ᶜq_tot = @. lazy(specific(state.c.ρq_tot, ᶜρ))
    ᶜq_lcl = @. lazy(specific(state.c.ρq_lcl, ᶜρ))
    ᶜq_icl = @. lazy(specific(state.c.ρq_icl, ᶜρ))
    ᶜq_rai = @. lazy(specific(state.c.ρq_rai, ᶜρ))
    ᶜq_sno = @. lazy(specific(state.c.ρq_sno, ᶜρ))
    return @. lazy(
        _mp1m_source_term(
            $Val(F), cmp, cmp_i, thp,
            ᶜρ, ᶜT, ᶜw, ᶜq_tot, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
        ),
    )
end

compute_mp1m_source(state, cache, time, F) =
    compute_mp1m_source(state, cache, time, F, cache.atmos.microphysics_model)
compute_mp1m_source(_, _, _, F, model) =
    error_diagnostic_variable("mp1m_$(F)", model)
compute_mp1m_source(state, cache, _, F, ::NonEquilibriumMicrophysics1M) =
    compute_mp1m_source(state, cache, F)

"""
    compute_mp1m_linearized_source(state, cache, ::Val{F})
    compute_mp1m_linearized_source(state, cache, time, F)

Evaluate the substep-averaged (`LinearizedAverageVerbose`) 1-moment source term
`F` on the grid-mean state over `(mp_averaging_dt, n_substeps)`.
"""
function compute_mp1m_linearized_source(state, cache, ::Val{F}) where {F}
    cmp = CAP.microphysics_1m_params(cache.params)
    cmp_i = CAP.microphysics_1m_implicit_params(cache.params)
    thp = CAP.thermodynamics_params(cache.params)
    dt = ClimaAtmos.mp_averaging_dt(cache, cache.mp_averaging)
    nsubs = cache.atmos.microphysics_model.n_substeps
    ᶜρ = state.c.ρ
    ᶜT = cache.precomputed.ᶜT
    ᶜw = @. lazy(w_component(Geometry.WVector(cache.precomputed.ᶜu)))
    ᶜq_tot = cache.precomputed.ᶜq_tot_nonneg
    ᶜq_lcl = @. lazy(specific(state.c.ρq_lcl, ᶜρ))
    ᶜq_icl = @. lazy(specific(state.c.ρq_icl, ᶜρ))
    ᶜq_rai = @. lazy(specific(state.c.ρq_rai, ᶜρ))
    ᶜq_sno = @. lazy(specific(state.c.ρq_sno, ᶜρ))
    return @. lazy(
        _mp1m_linearized_source_term(
            $Val(F), cmp, cmp_i, thp,
            ᶜρ, ᶜT, ᶜw, ᶜq_tot, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno, dt, nsubs,
        ),
    )
end

compute_mp1m_linearized_source(state, cache, time, F) =
    compute_mp1m_linearized_source(
        state, cache, time, F, cache.atmos.microphysics_model,
    )
compute_mp1m_linearized_source(_, _, _, F, model) =
    error_diagnostic_variable("mp1m_la_$(F)", model)
compute_mp1m_linearized_source(
    state, cache, _, F, ::NonEquilibriumMicrophysics1M,
) = compute_mp1m_linearized_source(state, cache, F)

# ---------------------------------------------------------------------------
# Updraft compute functions (PrognosticEDMFX, first subdomain)
# ---------------------------------------------------------------------------
"""
    compute_mp1m_source_updraft(state, cache, ::Val{F})
    compute_mp1m_source_updraft(state, cache, time, F)

Evaluate the instantaneous (`t = 0` potential rate) 1-moment source term `F` on
the first updraft subdomain (in-draft specific rate).
"""
function compute_mp1m_source_updraft(state, cache, ::Val{F}) where {F}
    cmp = CAP.microphysics_1m_params(cache.params)
    cmp_i = CAP.microphysics_1m_implicit_params(cache.params)
    thp = CAP.thermodynamics_params(cache.params)
    ᶜρʲ = cache.precomputed.ᶜρʲs.:1
    ᶜTʲ = cache.precomputed.ᶜTʲs.:1
    ᶜwʲ = @. lazy(w_component(Geometry.WVector(cache.precomputed.ᶜuʲs.:1)))
    ᶜq_totʲ = cache.precomputed.ᶜq_tot_nonnegʲs.:1
    ᶜq_lclʲ = (state.c.sgsʲs.:1).q_lcl
    ᶜq_iclʲ = (state.c.sgsʲs.:1).q_icl
    ᶜq_raiʲ = (state.c.sgsʲs.:1).q_rai
    ᶜq_snoʲ = (state.c.sgsʲs.:1).q_sno
    return @. lazy(
        _mp1m_source_term(
            $Val(F), cmp, cmp_i, thp,
            ᶜρʲ, ᶜTʲ, ᶜwʲ, ᶜq_totʲ, ᶜq_lclʲ, ᶜq_iclʲ, ᶜq_raiʲ, ᶜq_snoʲ,
        ),
    )
end

compute_mp1m_source_updraft(state, cache, time, F) =
    compute_mp1m_source_updraft(
        state, cache, time, F,
        cache.atmos.microphysics_model, cache.atmos.turbconv_model,
    )
compute_mp1m_source_updraft(_, _, _, F, mp_model, tc_model) =
    error_diagnostic_variable("mp1mup_$(F)", (mp_model, tc_model))
compute_mp1m_source_updraft(
    state, cache, _, F,
    ::NonEquilibriumMicrophysics1M, ::PrognosticEDMFX,
) = compute_mp1m_source_updraft(state, cache, F)

"""
    compute_mp1m_linearized_source_updraft(state, cache, ::Val{F})
    compute_mp1m_linearized_source_updraft(state, cache, time, F)

Evaluate the substep-averaged (`LinearizedAverageVerbose`) 1-moment source term
`F` on the first updraft subdomain over `(mp_averaging_dt, n_substeps)`. The
species sum of these process rates equals `mp1mup_dq_*_dt` to machine precision.
"""
function compute_mp1m_linearized_source_updraft(state, cache, ::Val{F}) where {F}
    cmp = CAP.microphysics_1m_params(cache.params)
    cmp_i = CAP.microphysics_1m_implicit_params(cache.params)
    thp = CAP.thermodynamics_params(cache.params)
    dt = ClimaAtmos.mp_averaging_dt(cache, cache.mp_averaging)
    nsubs = cache.atmos.microphysics_model.n_substeps
    ᶜρʲ = cache.precomputed.ᶜρʲs.:1
    ᶜTʲ = cache.precomputed.ᶜTʲs.:1
    ᶜwʲ = @. lazy(w_component(Geometry.WVector(cache.precomputed.ᶜuʲs.:1)))
    ᶜq_totʲ = cache.precomputed.ᶜq_tot_nonnegʲs.:1
    ᶜq_lclʲ = (state.c.sgsʲs.:1).q_lcl
    ᶜq_iclʲ = (state.c.sgsʲs.:1).q_icl
    ᶜq_raiʲ = (state.c.sgsʲs.:1).q_rai
    ᶜq_snoʲ = (state.c.sgsʲs.:1).q_sno
    return @. lazy(
        _mp1m_linearized_source_term(
            $Val(F), cmp, cmp_i, thp,
            ᶜρʲ, ᶜTʲ, ᶜwʲ, ᶜq_totʲ, ᶜq_lclʲ, ᶜq_iclʲ, ᶜq_raiʲ, ᶜq_snoʲ,
            dt, nsubs,
        ),
    )
end

compute_mp1m_linearized_source_updraft(state, cache, time, F) =
    compute_mp1m_linearized_source_updraft(
        state, cache, time, F,
        cache.atmos.microphysics_model, cache.atmos.turbconv_model,
    )
compute_mp1m_linearized_source_updraft(_, _, _, F, mp_model, tc_model) =
    error_diagnostic_variable("mp1mup_la_$(F)", (mp_model, tc_model))
compute_mp1m_linearized_source_updraft(
    state, cache, _, F,
    ::NonEquilibriumMicrophysics1M, ::PrognosticEDMFX,
) = compute_mp1m_linearized_source_updraft(state, cache, F)

# ---------------------------------------------------------------------------
# Environment compute functions (PrognosticEDMFX)
# ---------------------------------------------------------------------------
"""
    compute_mp1m_source_env(state, cache, ::Val{F})
    compute_mp1m_source_env(state, cache, time, F)

Evaluate the instantaneous (`t = 0` potential rate) 1-moment source term `F` on
the environment subdomain.

The four-argument form dispatches jointly on `cache.atmos.microphysics_model` and
`cache.atmos.turbconv_model`, and errors unless both `NonEquilibriumMicrophysics1M` and
`PrognosticEDMFX` are active. The environment density is diagnosed from the environment
temperature and the grid-mean pressure; the condensate and precipitation mixing ratios
come from `ᶜspecific_env_value`. The vertical velocity is that of the environment
tendency (`env_autoconversion_velocity` where it is active).
"""
function compute_mp1m_source_env(state, cache, ::Val{F}) where {F}
    cmp = CAP.microphysics_1m_params(cache.params)
    cmp_i = CAP.microphysics_1m_implicit_params(cache.params)
    thp = CAP.thermodynamics_params(cache.params)
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = cache.precomputed
    eps_FT = eps(eltype(thp))
    ᶜρ⁰ = @. lazy(
        max(
            eps_FT,
            TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
        ),
    )
    ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), state, cache)
    ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), state, cache)
    # The velocity of the environment tendency (`set_environment_1m_tendency!`).
    ᶜw⁰ = if uses_env_autoconversion_velocity(cmp, cache.params)
        ᶜenv_autoconversion_velocity(state, cache, ᶜq_lcl⁰, ᶜq_icl⁰)
    else
        @. lazy(w_component(Geometry.WVector(cache.precomputed.ᶜu⁰)))
    end
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), state, cache)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), state, cache)
    return @. lazy(
        _mp1m_source_term(
            $Val(F), cmp, cmp_i, thp,
            ᶜρ⁰, ᶜT⁰, ᶜw⁰, ᶜq_tot_nonneg⁰, ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰,
        ),
    )
end

compute_mp1m_source_env(state, cache, time, F) =
    compute_mp1m_source_env(
        state, cache, time, F,
        cache.atmos.microphysics_model, cache.atmos.turbconv_model,
    )
compute_mp1m_source_env(_, _, _, F, mp_model, tc_model) =
    error_diagnostic_variable("mp1men_$(F)", (mp_model, tc_model))
compute_mp1m_source_env(
    state, cache, _, F,
    ::NonEquilibriumMicrophysics1M, ::PrognosticEDMFX,
) = compute_mp1m_source_env(state, cache, F)

"""
    compute_mp1m_linearized_source_env(state, cache, ::Val{F})
    compute_mp1m_linearized_source_env(state, cache, time, F)

Evaluate the substep-averaged (`LinearizedAverageVerbose`) 1-moment source term
`F` on the environment mean state over `(mp_averaging_dt, n_substeps_quad)`.
"""
function compute_mp1m_linearized_source_env(state, cache, ::Val{F}) where {F}
    cmp = CAP.microphysics_1m_params(cache.params)
    cmp_i = CAP.microphysics_1m_implicit_params(cache.params)
    thp = CAP.thermodynamics_params(cache.params)
    eps_FT = eps(eltype(thp))
    dt = ClimaAtmos.mp_averaging_dt(cache, cache.mp_averaging)
    nsubs = cache.atmos.microphysics_model.n_substeps_quad
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = cache.precomputed
    ᶜρ⁰ = @. lazy(
        max(
            eps_FT,
            TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
        ),
    )
    ᶜw⁰ = @. lazy(w_component(Geometry.WVector(cache.precomputed.ᶜu⁰)))
    ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), state, cache)
    ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), state, cache)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), state, cache)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), state, cache)
    return @. lazy(
        _mp1m_linearized_source_term(
            $Val(F), cmp, cmp_i, thp,
            ᶜρ⁰, ᶜT⁰, ᶜw⁰, ᶜq_tot_nonneg⁰, ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰,
            dt, nsubs,
        ),
    )
end

compute_mp1m_linearized_source_env(state, cache, time, F) =
    compute_mp1m_linearized_source_env(
        state, cache, time, F,
        cache.atmos.microphysics_model, cache.atmos.turbconv_model,
    )
compute_mp1m_linearized_source_env(_, _, _, F, mp_model, tc_model) =
    error_diagnostic_variable("mp1men_la_$(F)", (mp_model, tc_model))
compute_mp1m_linearized_source_env(
    state, cache, _, F,
    ::NonEquilibriumMicrophysics1M, ::PrognosticEDMFX,
) = compute_mp1m_linearized_source_env(state, cache, F)

# ---------------------------------------------------------------------------
# Applied per-species net tendency compute functions (from `cache.precomputed`)
# ---------------------------------------------------------------------------
"""
    compute_mp1m_species_tendency(state, cache, time, ::Val{F})

Return the exact grid-mean specific microphysics tendency `(1/ρ) ∂(ρq_*)/∂t|_mp`
applied to `Yₜ.c.ρq_*` in `microphysics_tendency!`, summing the cached bulk
(`ᶜmp_tendency*`) and implicit (`ᶜmp_tendency_implicit*`) contributions (and
area-weighting the environment and updraft subdomains when `PrognosticEDMFX` is
active).
"""
compute_mp1m_species_tendency(state, cache, time, F) =
    compute_mp1m_species_tendency(
        state, cache, time, F,
        cache.atmos.microphysics_model, cache.atmos.turbconv_model,
    )
compute_mp1m_species_tendency(_, _, _, ::Val{F}, mp_model, tc_model) where {F} =
    error_diagnostic_variable("mp1m_$(F)", (mp_model, tc_model))
function compute_mp1m_species_tendency(
    state, cache, _, ::Val{F}, ::NonEquilibriumMicrophysics1M, _,
) where {F}
    ClimaAtmos.ensure_1m_tendencies!(state, cache)
    (; ᶜmp_tendency, ᶜmp_tendency_implicit) = cache.precomputed
    return @. lazy(
        _mp1m_species_sum($Val(F), ᶜmp_tendency, ᶜmp_tendency_implicit),
    )
end
function compute_mp1m_species_tendency(
    state,
    cache,
    _,
    ::Val{F},
    ::NonEquilibriumMicrophysics1M,
    tm::PrognosticEDMFX,
) where {F}
    ClimaAtmos.ensure_1m_tendencies!(state, cache)
    (;
        ᶜmp_tendency⁰, ᶜmp_tendency_implicit⁰,
        ᶜmp_tendencyʲs, ᶜmp_tendency_implicitʲs,
    ) = cache.precomputed
    ᶜρ = state.c.ρ
    ᶜρa¹ = (state.c.sgsʲs.:1).ρa
    ᶜρa⁰ = @. lazy(ρa⁰(ᶜρ, state.c.sgsʲs, tm))
    return @. lazy(
        (
            ᶜρa⁰ * _mp1m_species_sum(
                $Val(F), ᶜmp_tendency⁰, ᶜmp_tendency_implicit⁰,
            ) +
            ᶜρa¹ * _mp1m_species_sum(
                $Val(F), ᶜmp_tendencyʲs.:1, ᶜmp_tendency_implicitʲs.:1,
            )
        ) / ᶜρ,
    )
end

"""
    compute_mp1m_species_tendency_updraft(state, cache, time, ::Val{F})

Return the in-draft specific microphysics tendency applied to the first updraft
(`Yₜ.c.sgsʲs.:1.q_*`) in `microphysics_tendency!`, summing
`cache.precomputed.ᶜmp_tendencyʲs.:1` and
`cache.precomputed.ᶜmp_tendency_implicitʲs.:1`.
"""
compute_mp1m_species_tendency_updraft(state, cache, time, F) =
    compute_mp1m_species_tendency_updraft(
        state, cache, time, F,
        cache.atmos.microphysics_model, cache.atmos.turbconv_model,
    )
compute_mp1m_species_tendency_updraft(
    _, _, _, ::Val{F}, mp_model, tc_model,
) where {F} = error_diagnostic_variable("mp1mup_$(F)", (mp_model, tc_model))
function compute_mp1m_species_tendency_updraft(
    state,
    cache,
    _,
    ::Val{F},
    ::NonEquilibriumMicrophysics1M,
    ::PrognosticEDMFX,
) where {F}
    ClimaAtmos.ensure_1m_tendencies!(state, cache)
    (; ᶜmp_tendencyʲs, ᶜmp_tendency_implicitʲs) = cache.precomputed
    return @. lazy(
        _mp1m_species_sum(
            $Val(F), ᶜmp_tendencyʲs.:1, ᶜmp_tendency_implicitʲs.:1,
        ),
    )
end

"""
    compute_mp1m_species_tendency_env(state, cache, time, ::Val{F})

Return the exact environment specific microphysics tendency applied in
`microphysics_tendency!`, summing `cache.precomputed.ᶜmp_tendency⁰` (including
8-node SGS conditional quadrature, precipitation-shaft placement, and
`LinearizedAverage` substepping) and `cache.precomputed.ᶜmp_tendency_implicit⁰`.
"""
compute_mp1m_species_tendency_env(state, cache, time, F) =
    compute_mp1m_species_tendency_env(
        state, cache, time, F,
        cache.atmos.microphysics_model, cache.atmos.turbconv_model,
    )
compute_mp1m_species_tendency_env(
    _, _, _, ::Val{F}, mp_model, tc_model,
) where {F} = error_diagnostic_variable("mp1men_$(F)", (mp_model, tc_model))
function compute_mp1m_species_tendency_env(
    state,
    cache,
    _,
    ::Val{F},
    ::NonEquilibriumMicrophysics1M,
    ::PrognosticEDMFX,
) where {F}
    ClimaAtmos.ensure_1m_tendencies!(state, cache)
    (; ᶜmp_tendency⁰, ᶜmp_tendency_implicit⁰) = cache.precomputed
    return @. lazy(
        _mp1m_species_sum($Val(F), ᶜmp_tendency⁰, ᶜmp_tendency_implicit⁰),
    )
end

# ---------------------------------------------------------------------------
# Register all diagnostics via loops
# ---------------------------------------------------------------------------

for (prefix, label, location, compute_fn) in (
    ("mp1m", "", "grid-mean state", compute_mp1m_source),
    (
        "mp1mup",
        " Updraft",
        "updraft state (first subdomain)",
        compute_mp1m_source_updraft,
    ),
    (
        "mp1men",
        " Environment",
        "environment state",
        compute_mp1m_source_env,
    ),
)
    for (field, description) in MP1M_SOURCE_TERMS
        add_diagnostic_variable!(;
            short_name = "$(prefix)_$(field)",
            long_name = "1M Microphysics$(label) Instantaneous Potential Rate: $(description)",
            units = "kg kg^-1 s^-1",
            comments = "Instantaneous potential rate (t = 0, un-attenuated by substep " *
                       "averaging or SGS quadrature) of source term $(field) from " *
                       "bulk_microphysics_tendencies(InstantaneousVerbose()). " *
                       "Evaluated at $(location).",
            compute = let F = Val(field), f = compute_fn
                (state, cache, time) -> f(state, cache, time, F)
            end,
        )
    end
end

for (prefix, label, location, compute_fn) in (
    ("mp1m_la", "", "grid-mean state", compute_mp1m_linearized_source),
    (
        "mp1mup_la",
        " Updraft",
        "updraft state (first subdomain)",
        compute_mp1m_linearized_source_updraft,
    ),
    (
        "mp1men_la",
        " Environment",
        "environment state",
        compute_mp1m_linearized_source_env,
    ),
)
    for (field, description) in MP1M_SOURCE_TERMS
        add_diagnostic_variable!(;
            short_name = "$(prefix)_$(field)",
            long_name = "1M Microphysics$(label) Substep-Averaged Rate: $(description)",
            units = "kg kg^-1 s^-1",
            comments = "Substep-averaged rate of source term $(field) from " *
                       "bulk_microphysics_tendencies(LinearizedAverageVerbose()) " *
                       "for bulk processes plus InstantaneousVerbose() for implicit " *
                       "condensation/deposition. Evaluated at $(location).",
            compute = let F = Val(field), f = compute_fn
                (state, cache, time) -> f(state, cache, time, F)
            end,
        )
    end
end

for (prefix, label, location, compute_fn) in (
    (
        "mp1m",
        "",
        "grid-mean state (cached ᶜmp_tendency + ᶜmp_tendency_implicit, or area-weighted EDMF sum)",
        compute_mp1m_species_tendency,
    ),
    (
        "mp1mup",
        " Updraft",
        "updraft state (cached ᶜmp_tendencyʲs.:1 + ᶜmp_tendency_implicitʲs.:1)",
        compute_mp1m_species_tendency_updraft,
    ),
    (
        "mp1men",
        " Environment",
        "environment state (cached ᶜmp_tendency⁰ + ᶜmp_tendency_implicit⁰, including SGS quadrature and shaft placement)",
        compute_mp1m_species_tendency_env,
    ),
)
    for (field, description) in MP1M_SPECIES_TENDENCIES
        add_diagnostic_variable!(;
            short_name = "$(prefix)_$(field)",
            long_name = "1M Microphysics$(label): $(description)",
            units = "kg kg^-1 s^-1",
            comments = "Applied net species tendency $(field) read directly from " *
                       "precomputed microphysics cache at $(location).",
            compute = let F = Val(field), f = compute_fn
                (state, cache, time) -> f(state, cache, time, F)
            end,
        )
    end
end
