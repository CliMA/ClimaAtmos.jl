#=
This file is included in Diagnostics.jl

Microphysics 1-moment process tendency diagnostics.

Outputs individual source terms from CloudMicrophysics
`bulk_microphysics_tendencies(InstantaneousVerbose(), ...)` for:
  - Grid-mean state (prefix: mp1m_)
  - Updraft state for PrognosticEDMFX (prefix: mp1mup_)
  - Environment state for PrognosticEDMFX (prefix: mp1men_)

Each diagnostic calls InstantaneousVerbose and extracts a single scalar field
inside the broadcast, so the result is a proper `Field{FT}` with zero
intermediate allocations.
=#

import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

# ---------------------------------------------------------------------------
# Helper: call InstantaneousVerbose and extract one field by name.
# Because everything is @inline, the compiler can dead-code-eliminate unused
# source terms inside the broadcast.
# ---------------------------------------------------------------------------
@inline function _mp1m_source_term(
    ::Val{F}, mp, tps, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
) where {F}
    src = BMT.bulk_microphysics_tendencies(
        BMT.InstantaneousVerbose(), BMT.Microphysics1Moment(),
        mp, tps, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
    )
    return getfield(src, F)
end

# ---------------------------------------------------------------------------
# The 18 source terms produced by _microphysics_source_terms.
# Each entry: (field_symbol, human-readable description)
# ---------------------------------------------------------------------------
const MP1M_SOURCE_TERMS = [
    (:S_phase_change_vap_lcl, "Vapor ↔ cloud liquid phase change"),
    (:S_phase_change_vap_icl, "Vapor ↔ cloud ice phase change"),
    (:S_acnv_lcl_rai, "Cloud liquid → rain autoconversion"),
    (:S_acnv_icl_sno, "Cloud ice → snow autoconversion"),
    (:S_accr_lcl_rai, "Cloud liquid + rain accretion"),
    (:S_accr_lcl_sno_cold, "Cloud liquid + snow accretion (cold)"),
    (:S_accr_lcl_sno_warm, "Cloud liquid + snow accretion (warm)"),
    (:S_accr_melt_lcl_sno, "Thermal melt of snow by warm cloud liquid"),
    (:S_accr_icl_rai, "Cloud ice + rain accretion"),
    (:S_accr_freeze_icl_rai, "Rain frozen in ice-rain collision"),
    (:S_accr_icl_sno, "Cloud ice + snow accretion"),
    (:S_accr_rai_sno_cold, "Rain-snow accretion (cold, rain → snow)"),
    (:S_accr_rai_sno_warm, "Rain-snow accretion (warm, snow → rain)"),
    (:S_accr_melt_rai_sno, "Thermal melt of snow by warm rain"),
    (:S_phase_change_vap_rai, "Rain condensation/evaporation"),
    (:S_phase_change_vap_sno, "Snow deposition/sublimation"),
    (:S_melt_icl_lcl, "Cloud ice melt → cloud liquid"),
    (:S_melt_sno_rai, "Snow melt → rain"),
]

# ---------------------------------------------------------------------------
# Grid-mean compute function
# ---------------------------------------------------------------------------
function compute_mp1m_source(state, cache, ::Val{F}) where {F}
    cmp = CAP.microphysics_1m_params(cache.params)
    thp = CAP.thermodynamics_params(cache.params)
    ᶜρ = state.c.ρ
    ᶜT = cache.precomputed.ᶜT
    ᶜq_tot = @. lazy(specific(state.c.ρq_tot, ᶜρ))
    ᶜq_lcl = @. lazy(specific(state.c.ρq_lcl, ᶜρ))
    ᶜq_icl = @. lazy(specific(state.c.ρq_icl, ᶜρ))
    ᶜq_rai = @. lazy(specific(state.c.ρq_rai, ᶜρ))
    ᶜq_sno = @. lazy(specific(state.c.ρq_sno, ᶜρ))
    return @. lazy(
        _mp1m_source_term(
            Val(F), cmp, thp, ᶜρ, ᶜT, ᶜq_tot, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
        ),
    )
end

# Dispatch: only defined for 1M microphysics
compute_mp1m_source(state, cache, time, F) =
    compute_mp1m_source(state, cache, time, F, cache.atmos.microphysics_model)
compute_mp1m_source(_, _, _, F, model) =
    error_diagnostic_variable("mp1m_$(F)", model)
compute_mp1m_source(state, cache, _, F, ::NonEquilibriumMicrophysics1M) =
    compute_mp1m_source(state, cache, F)

# ---------------------------------------------------------------------------
# Updraft compute function (PrognosticEDMFX, first subdomain)
# ---------------------------------------------------------------------------
function compute_mp1m_source_updraft(state, cache, ::Val{F}) where {F}
    cmp = CAP.microphysics_1m_params(cache.params)
    thp = CAP.thermodynamics_params(cache.params)
    ᶜρʲ = cache.precomputed.ᶜρʲs.:1
    ᶜTʲ = cache.precomputed.ᶜTʲs.:1
    ᶜq_totʲ = cache.precomputed.ᶜq_tot_nonnegʲs.:1
    ᶜq_lclʲ = (state.c.sgsʲs.:1).q_lcl
    ᶜq_iclʲ = (state.c.sgsʲs.:1).q_icl
    ᶜq_raiʲ = (state.c.sgsʲs.:1).q_rai
    ᶜq_snoʲ = (state.c.sgsʲs.:1).q_sno
    return @. lazy(
        _mp1m_source_term(
            Val(F), cmp, thp,
            ᶜρʲ, ᶜTʲ, ᶜq_totʲ, ᶜq_lclʲ, ᶜq_iclʲ, ᶜq_raiʲ, ᶜq_snoʲ,
        ),
    )
end

# Dispatch: only for 1M + PrognosticEDMFX
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

# ---------------------------------------------------------------------------
# Environment compute function (PrognosticEDMFX)
# ---------------------------------------------------------------------------
function compute_mp1m_source_env(state, cache, ::Val{F}) where {F}
    cmp = CAP.microphysics_1m_params(cache.params)
    thp = CAP.thermodynamics_params(cache.params)
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = cache.precomputed
    ᶜρ⁰ = @. lazy(
        TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
    )
    ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), state, cache)
    ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), state, cache)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), state, cache)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), state, cache)
    return @. lazy(
        _mp1m_source_term(
            Val(F), cmp, thp,
            ᶜρ⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰,
        ),
    )
end

# Dispatch: only for 1M + PrognosticEDMFX
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
            long_name = "1M Microphysics$(label): $(description)",
            units = "kg kg^-1 s^-1",
            comments = "Instantaneous source term $(field) from " *
                       "bulk_microphysics_tendencies(InstantaneousVerbose()). " *
                       "Evaluated at $(location).",
            compute = let F = Val(field), f = compute_fn
                (state, cache, time) -> f(state, cache, time, F)
            end,
        )
    end
end
