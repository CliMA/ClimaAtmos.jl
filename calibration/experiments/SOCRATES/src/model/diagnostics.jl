"""
Diagnostic output for a SOCRATES run.

The default set is exactly the hydrometeor profiles and water paths the calibration scores on,
written as running averages at the calibration cadence. Anything extra is per-member,
per-case NetCDF that nothing reads, so it is opt-in rather than default.
"""

using ClimaAtmos: ClimaAtmos as CA

"""Hydrometeor profile variables scored by the calibration."""
const PROFILE_VARS = ("clw", "cli", "husra", "hussn")

"""Column-integrated water path variables scored by the calibration."""
const PATH_VARS = ("lwp", "iwp", "rwp", "swp")

"""The default scored variables: hydrometeor profiles and water paths."""
const DEFAULT_DIAGNOSTIC_VARS = (PROFILE_VARS..., PATH_VARS...)

"""
State, updraft and turbulence fields for diagnosing a run's evolution, beyond what is scored.
`pfull` is the quantity whose sign a thermodynamic blow-up destroys first; `arup`/`waup` are the
updraft area and vertical velocity.
"""
const DEBUG_DIAGNOSTIC_VARS = (
    DEFAULT_DIAGNOSTIC_VARS...,
    "pfull",
    "rhoa",
    "ta",
    "hus",
    "wa",
    "tke",
    "arup",
    "waup",
    "entr",
    "detr",
)

"""
Default diagnostic period. 600 s is the cadence the scoring window is averaged over; writing
faster only costs I/O.
"""
const DEFAULT_DIAGNOSTIC_PERIOD = "10mins"

"""
The 1-moment microphysics process rates ClimaAtmos registers, in
`src/diagnostics/microphysics_diagnostics.jl`. Available only for `NonEquilibriumMicrophysics1M`,
which is what [`socrates_model`](@ref) builds.
"""
const MP1M_SOURCE_TERMS = (
    "S_phase_change_vap_lcl",
    "S_phase_change_vap_icl",
    "S_acnv_lcl_rai",
    "S_acnv_icl_sno",
    "S_accr_lcl_rai",
    "S_accr_lcl_sno_cold",
    "S_accr_lcl_sno_warm",
    "S_accr_melt_lcl_sno",
    "S_accr_icl_rai",
    "S_accr_freeze_icl_rai",
    "S_accr_icl_sno",
    "S_accr_rai_sno_cold",
    "S_accr_rai_sno_warm",
    "S_accr_melt_rai_sno",
    "S_phase_change_vap_rai",
    "S_phase_change_vap_sno",
    "S_melt_icl_lcl",
    "S_melt_sno_rai",
)

"""
Where a process rate is evaluated: grid mean, updraft, or environment. The updraft and environment
sets additionally require `PrognosticEDMFX`.
"""
const MP1M_LOCATIONS = ("mp1m", "mp1mup", "mp1men")

"""
Which process rates enter each prognostic tendency, and with what sign.

Transcribed from `CloudMicrophysics.BulkMicrophysicsTendencies._aggregate_tendencies`, so summing a
variable's signed terms reproduces its `dq/dt` exactly. Cold/warm branching is already baked into the
source terms, so every sign here is fixed.
"""
const MP1M_BUDGETS = Dict{String, Vector{Tuple{String, Int}}}(
    "q_lcl" => [
        ("S_phase_change_vap_lcl", +1),
        ("S_acnv_lcl_rai", -1),
        ("S_accr_lcl_rai", -1),
        ("S_accr_lcl_sno_cold", -1),
        ("S_accr_lcl_sno_warm", -1),
        ("S_melt_icl_lcl", +1),
    ],
    "q_icl" => [
        ("S_phase_change_vap_icl", +1),
        ("S_acnv_icl_sno", -1),
        ("S_accr_icl_rai", -1),
        ("S_accr_icl_sno", -1),
        ("S_melt_icl_lcl", -1),
    ],
    "q_rai" => [
        ("S_acnv_lcl_rai", +1),
        ("S_accr_lcl_rai", +1),
        ("S_accr_lcl_sno_warm", +1),
        ("S_accr_melt_lcl_sno", +1),
        ("S_accr_freeze_icl_rai", -1),
        ("S_accr_rai_sno_cold", -1),
        ("S_accr_rai_sno_warm", +1),
        ("S_accr_melt_rai_sno", +1),
        ("S_phase_change_vap_rai", +1),
        ("S_melt_sno_rai", +1),
    ],
    "q_sno" => [
        ("S_acnv_icl_sno", +1),
        ("S_accr_lcl_sno_cold", +1),
        ("S_accr_melt_lcl_sno", -1),
        ("S_accr_icl_rai", +1),
        ("S_accr_freeze_icl_rai", +1),
        ("S_accr_icl_sno", +1),
        ("S_accr_rai_sno_cold", +1),
        ("S_accr_rai_sno_warm", -1),
        ("S_accr_melt_rai_sno", -1),
        ("S_phase_change_vap_sno", +1),
        ("S_melt_sno_rai", -1),
    ],
)

"""The prognostic variables with a budget, in a stable order for figures."""
const MP1M_BUDGET_VARS = ("q_lcl", "q_icl", "q_rai", "q_sno")

"""
Transport prefixes registered by [`register_transport_diagnostics!`](@ref), and the sign they enter a
budget with. Each is already a `∂q/∂t`, so every sign is `+1`.
"""
const TRANSPORT_PREFIXES = ("sed", "adv", "dif")

"""Transport terms of each budget variable, as `(diagnostic name, sign)` — the same shape as
[`MP1M_BUDGETS`](@ref) so the two concatenate."""
const TRANSPORT_BUDGETS = Dict{String, Vector{Tuple{String, Int}}}(
    var => [("$(prefix)_$(var)", +1) for prefix in TRANSPORT_PREFIXES] for
    var in MP1M_BUDGET_VARS
)

"""
Every microphysics process rate at every location, plus the state and updraft fields needed to
interpret them. This is the set a postprocessing run asks for — 54 rates is far too much output for
a calibration member, so it is never a default.
"""
const TENDENCY_DIAGNOSTIC_VARS = (
    DEBUG_DIAGNOSTIC_VARS...,
    ("$(loc)_$(term)" for loc in MP1M_LOCATIONS for term in MP1M_SOURCE_TERMS)...,
    ("$(prefix)_$(var)" for prefix in TRANSPORT_PREFIXES for var in MP1M_BUDGET_VARS)...,
)

"""
    socrates_diagnostics(short_names; period, reduction, n_levels, output_at_levels)

`DiagnosticsConfig` writing `short_names` at `period` with the given time `reduction`.

`output_at_levels = true` writes on the model's own levels, so the `z` coordinate of the output is
[`socrates_z`](@ref) — which is what lets observations be built on the same grid.
"""
function socrates_diagnostics(
    short_names = DEFAULT_DIAGNOSTIC_VARS;
    period::AbstractString = DEFAULT_DIAGNOSTIC_PERIOD,
    reduction::AbstractString = "average",
    n_levels::Int,
    output_at_levels::Bool = true,
)
    isempty(short_names) && error(
        "socrates_diagnostics needs at least one short name; a run with no diagnostics cannot \
         be scored.",
    )
    register_transport_diagnostics!()
    return CA.DiagnosticsConfig(;
        default = false,
        additional = ((; short_name = collect(String.(short_names)), period, reduction),),
        interpolation_num_points = (2, 2, n_levels),
        output_at_levels,
    )
end
