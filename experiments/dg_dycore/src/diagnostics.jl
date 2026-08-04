#=
ClimaDiagnostics output with ClimaAtmos-standard short names/metadata
(mirroring config/model_configs/baroclinic_wave.yml's set: pfull, ua, va,
wa, ta, ke + rhoa and rv), so the NetCDF files are drop-in compatible with
ClimaAnalysis and comparable 1:1 against CG ClimaAtmos runs.

Compute functions evaluate FROM THE STATE (ClimaAtmos's own compute
functions read `cache.precomputed`, which this sandbox does not carry) with
`cache` = the DGModel. Never stale, no AtmosModel types.

`rv` (relative vorticity) is DG-CONSISTENT: element-local strong curl of
uₕ completed by the central face lifting (`central_curl3_lift`) — the DG
analog of ClimaAtmos's curlₕ + weighted_dss! (which cannot be reused here:
no DSS in a DG discretization).
=#

import ClimaDiagnostics
import ClimaDiagnostics: DiagnosticVariable, ScheduledDiagnostic
import ClimaDiagnostics.Schedules: EveryDtSchedule

# --- pointwise state -> diagnostic fields (FDDG state layout) ---

function dg_velocities(state, m::DGModel{FT, <:BaroclinicWaveFDDG}) where {FT}
    (; eE1, eE2, eE3, eN1, eN2, eN3) = m.fields
    ρ = state.c.ρ
    uE = @. (state.c.ρu1 * eE1 + state.c.ρu2 * eE2 + state.c.ρu3 * eE3) / ρ
    uN = @. (state.c.ρu1 * eN1 + state.c.ρu2 * eN2 + state.c.ρu3 * eN3) / ρ
    w_c = @. m.ops.Ic(Geometry.WVector(state.f.ρw)).components.data.:1 / ρ
    return uE, uN, w_c
end

function dg_velocities(state, m::DGModel{FT, <:VIProblem}) where {FT}
    uv = @. Geometry.UVVector(state.c.uₕ)
    w_c = @. m.ops.Ic(Geometry.WVector(state.f.w)).components.data.:1
    return uv.components.data.:1, uv.components.data.:2, w_c
end

function dg_pressure(state, m::DGModel)
    uE, uN, w_c = dg_velocities(state, m)
    K = @. (uE^2 + uN^2 + w_c^2) / 2
    return @. pres_ρe(m.c, state.c.ρe, K, m.fields.ᶜΦ, state.c.ρ)
end

# DG-consistent relative vorticity (radial component, physical units):
# strong hcurl of the covariant horizontal wind + central face lifting.
function dg_vorticity(state, m::DGModel{FT}) where {FT}
    (; hcurl) = m.ops
    uE, uN, _ = dg_velocities(state, m)
    lgeom_c = Fields.local_geometry_field(m.spaces.hv_center_space)
    uₕ = @. C12(Geometry.UVVector(uE, uN), lgeom_c)
    ω³ = @. Geometry.WVector(hcurl(uₕ)).components.data.:1
    ω³ .+= Operators.lifting_correction(
        Operators.central_curl3_lift,
        FT,
        uE,
        uN,
    )
    return ω³
end

# ClimaDiagnostics compute! convention: return a new field when out ===
# nothing (first call), else fill in place.
out_or(out, bc) = isnothing(out) ? Base.materialize(bc) : Base.materialize!(out, bc)

function compute_rhoa!(out, state, cache, time)
    isnothing(out) ? copy(state.c.ρ) : (out .= state.c.ρ; out)
end
function compute_ua!(out, state, cache, time)
    uE, _, _ = dg_velocities(state, cache)
    isnothing(out) ? uE : (out .= uE; out)
end
function compute_va!(out, state, cache, time)
    _, uN, _ = dg_velocities(state, cache)
    isnothing(out) ? uN : (out .= uN; out)
end
function compute_wa!(out, state, cache, time)
    _, _, w_c = dg_velocities(state, cache)
    isnothing(out) ? w_c : (out .= w_c; out)
end
function compute_pfull!(out, state, cache, time)
    p = dg_pressure(state, cache)
    isnothing(out) ? p : (out .= p; out)
end
function compute_ta!(out, state, cache, time)
    m = cache
    p = dg_pressure(state, m)
    T = @. p / (m.c.R_d * state.c.ρ)
    isnothing(out) ? T : (out .= T; out)
end
function compute_ke!(out, state, cache, time)
    uE, uN, w_c = dg_velocities(state, cache)
    K = @. (uE^2 + uN^2 + w_c^2) / 2
    isnothing(out) ? K : (out .= K; out)
end
function compute_rv!(out, state, cache, time)
    rv = dg_vorticity(state, cache)
    isnothing(out) ? rv : (out .= rv; out)
end

# Metadata mirrors ClimaAtmos src/diagnostics/core_diagnostics.jl so output
# is CMIP-style and directly comparable with CG runs.
const DG_DIAGNOSTIC_VARIABLES = [
    DiagnosticVariable(;
        short_name = "rhoa",
        long_name = "Air Density",
        standard_name = "air_density",
        units = "kg m^-3",
        compute! = compute_rhoa!,
    ),
    DiagnosticVariable(;
        short_name = "ua",
        long_name = "Eastward Wind",
        standard_name = "eastward_wind",
        units = "m s^-1",
        compute! = compute_ua!,
    ),
    DiagnosticVariable(;
        short_name = "va",
        long_name = "Northward Wind",
        standard_name = "northward_wind",
        units = "m s^-1",
        compute! = compute_va!,
    ),
    DiagnosticVariable(;
        short_name = "wa",
        long_name = "Upward Air Velocity",
        standard_name = "upward_air_velocity",
        units = "m s^-1",
        comments = "Cell-center interpolated ρw/ρ",
        compute! = compute_wa!,
    ),
    DiagnosticVariable(;
        short_name = "pfull",
        long_name = "Pressure at Model Full-Levels",
        standard_name = "air_pressure",
        units = "Pa",
        compute! = compute_pfull!,
    ),
    DiagnosticVariable(;
        short_name = "ta",
        long_name = "Air Temperature",
        standard_name = "air_temperature",
        units = "K",
        compute! = compute_ta!,
    ),
    DiagnosticVariable(;
        short_name = "ke",
        long_name = "Specific Kinetic Energy",
        standard_name = "specific_kinetic_energy",
        units = "m^2 s^-2",
        compute! = compute_ke!,
    ),
    DiagnosticVariable(;
        short_name = "rv",
        long_name = "Relative Vorticity",
        standard_name = "relative_vorticity",
        units = "s^-1",
        comments = "DG-consistent: element-local strong curl + central " *
                   "face lifting (no DSS)",
        compute! = compute_rv!,
    ),
]

"""
    dg_diagnostics(m, Y; output_dir, period) -> (scheduled, writer)

ClimaAtmos-compatible instantaneous diagnostics every `period` seconds,
written to NetCDF in `output_dir`.
"""
function dg_diagnostics(m::DGModel{FT}, Y; output_dir, period) where {FT}
    mkpath(output_dir)
    writer = ClimaDiagnostics.Writers.NetCDFWriter(
        axes(Y.c),
        output_dir;
        start_date = nothing,
    )
    scheduled = [
        ScheduledDiagnostic(;
            variable = v,
            output_writer = writer,
            compute_schedule_func = EveryDtSchedule(FT(period)),
            output_schedule_func = EveryDtSchedule(FT(period)),
        ) for v in DG_DIAGNOSTIC_VARIABLES
    ]
    return scheduled, writer
end
