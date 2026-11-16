# Per-process tendency diagnostics (3D fields).
#
# For each (process, prognostic-field) pair listed in `_GRID_FIELDS` /
# `_SGS_FIELDS`, this file registers a diagnostic of one of two kinds.
#
# Instantaneous ARK tendencies (`_ARK_PROCESSES`): the diagnostic zeroes a
# scratch `Yₜ`, evaluates that one tendency function of the IMEX ARK step on
# `(Yₜ, Y, p, t)` at the current state, and returns `Yₜ.c.<field>` (or
# `Yₜ.c.sgsʲs.:(1).<scalar>`).
#
# Realized operator tendencies (`REALIZED_OPERATORS`): the diagnostic returns
# `(Y_after - Y_before) / Δt` of that operator over the most recent step, as
# recorded by `RealizedIncrements` around each operator of the split step
# (ARK step, post-ARK vertical transport, column turbulence, water repair,
# microphysics, and state constraints). The realized tendencies of all
# operators sum to the change of the state over the step divided by `Δt`. The
# recorder is allocated the first time such a diagnostic is computed, so the
# realized tendencies are zero until the first step after that.
#
# Grid-scale diagnostics have units of `field · m^-3 · s^-1`. SGS
# updraft-scalar diagnostics have per-mass rates (`J kg^-1 s^-1` for `mse`,
# `s^-1` for `q_*`).
#
# Diagnostic short-name convention: `tn_<field-short>_<process-short>`
# (CMIP-flavoured tendency prefix, e.g. `tn_rhoe_iva`, `tn_sgsqt_turb`).
#
# The process short-name strips underscores from the manifest key
# (`hor_dyn` → `hordyn`, `hor_tracer_adv` → `hortraceradv`) so each of the
# three underscore-delimited segments is a single alphanumeric token.
#
# These are debug-oriented and expensive relative to standard diagnostics:
# each instantaneous sample runs one tendency evaluation into a persistent
# scratch `FieldVector`, and the realized recorder holds seven state-sized
# `FieldVector`s and updates them every step. Enable only via the
# `debug_tendency_diagnostics` config flag and use a coarse output cadence.

import ClimaCore.MatrixFields
import ClimaCore.Operators

# Tendency functions and the realized-increment recorder live in the parent
# ClimaAtmos module.
import ..implicit_vertical_advection_tendency!
import ..horizontal_dynamics_tendency!
import ..horizontal_tracer_advection_tendency!
import ..explicit_vertical_advection_tendency!
import ..hyperdiffusion_tendency!
import ..additional_tendency!
import ..do_dss
import ..REALIZED_OPERATORS
import ..enable_realized_increments!

# -- Persistent scratches ---------------------------------------------------
# The instantaneous compute functions need a `Y`-sized tendency `FieldVector`
# per call. Allocating one via `similar(state)` on every diagnostic sample
# creates enormous GC pressure on CPU and can OOM on GPU. Instead we hold a
# module-level scratch that is allocated once on first use and reused across
# every subsequent call. Diagnostics execute sequentially inside the
# ClimaDiagnostics callback, so sharing a single scratch is safe.
const _SCRATCH_Yₜ = Ref{Any}(nothing)
const _SCRATCH_Yₜ_LIM = Ref{Any}(nothing)

function _acquire_scratch(scratch_ref::Ref{Any}, state)
    Yₜ = scratch_ref[]
    if !(Yₜ isa typeof(state)) || axes(Yₜ.c) !== axes(state.c)
        Yₜ = similar(state)
        scratch_ref[] = Yₜ
    end
    Yₜ .= zero(eltype(Yₜ))
    return Yₜ
end

# -- Instantaneous ARK process wrappers -------------------------------------
# Each wrapper takes `(Yₜ, Y, p, t)` and accumulates a single process's
# contribution into `Yₜ`. The caller is responsible for zeroing `Yₜ` first.

_iva_wrap!(Yₜ, Y, p, t) = implicit_vertical_advection_tendency!(Yₜ, Y, p, t)

_hor_dyn_wrap!(Yₜ, Y, p, t) = horizontal_dynamics_tendency!(Yₜ, Y, p, t)

_hor_tracer_adv_wrap!(Yₜ, Y, p, t) =
    horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)

_exp_vadv_wrap!(Yₜ, Y, p, t) =
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)

# Hyperdiffusion writes non-tracer contributions to the first arg and tracer
# contributions to the second. Uses a persistent Yₜ_lim scratch to merge.
function _hdiff_wrap!(Yₜ, Y, p, t)
    Yₜ_lim = _acquire_scratch(_SCRATCH_Yₜ_LIM, Y)
    hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ .+= Yₜ_lim
    return Yₜ
end

_additional_wrap!(Yₜ, Y, p, t) = additional_tendency!(Yₜ, Y, p, t)

const _ARK_PROCESSES = (
    iva = _iva_wrap!,
    hor_dyn = _hor_dyn_wrap!,
    hor_tracer_adv = _hor_tracer_adv_wrap!,
    exp_vadv = _exp_vadv_wrap!,
    hdiff = _hdiff_wrap!,
    additional = _additional_wrap!,
)

# -- Field extraction -------------------------------------------------------

# Return the tendency of process `pname` as a `Y`-sized `FieldVector`: the
# persistent scratch filled by the instantaneous wrapper for an ARK process,
# or the recorder's buffer for a realized operator.
#
# Horizontal tendencies (`hor_dyn`, `hor_tracer_adv`, `hdiff`, and some
# sub-terms of `additional`) leave the field discontinuous across spectral
# element boundaries; `weighted_dss!` makes the diagnostic output continuous.
# For vertical-only tendencies DSS is a no-op (co-located values on shared
# edges are already equal because the input state is continuous). Realized
# tendencies are differences of continuous states and need no DSS; the
# recorder's buffers are returned as they are.
function _process_tendency(pname, state, cache, time)
    if pname in REALIZED_OPERATORS
        recorder = enable_realized_increments!(cache, state)
        return getproperty(recorder.tendencies, pname), false
    end
    Yₜ = _acquire_scratch(_SCRATCH_Yₜ, state)
    _ARK_PROCESSES[pname](Yₜ, state, cache, time)
    return Yₜ, do_dss(axes(state.c))
end

# Return the tendency field at `field_path` as a 3D center-space Field, or a
# lazy zero field on center space if the field is not present in the current
# config. Downstream (e.g. NetCDFWriter) copies the values before the next
# diagnostic call overwrites the scratch.
function _process_tendency_field(pname, state, cache, time, field_path)
    if !MatrixFields.has_field(state, field_path)
        return @. lazy(state.c.ρ * 0)  # 3D center-space zero
    end
    Yₜ, needs_dss = _process_tendency(pname, state, cache, time)
    field = MatrixFields.get_field(Yₜ, field_path)
    needs_dss && Spaces.weighted_dss!(field)
    return field
end

# SGS variant: return the 3D `d(scalar)/dt` field for updraft `j = 1`.
function _sgs_process_tendency_field(pname, state, cache, time, scalar_name)
    if !hasproperty(state.c, :sgsʲs) ||
       !hasproperty(state.c.sgsʲs.:(1), scalar_name)
        return @. lazy(state.c.ρ * 0)
    end
    Yₜ, needs_dss = _process_tendency(pname, state, cache, time)
    field = getproperty(Yₜ.c.sgsʲs.:(1), scalar_name)
    needs_dss && Spaces.weighted_dss!(field)
    return field
end

# -- Field × process manifests ---------------------------------------------
# Each entry lists the instantaneous ARK processes known to contribute a
# non-zero tendency to that field; every field also gets one diagnostic per
# realized operator (`REALIZED_OPERATORS`), so that the realized tendencies
# close the budget of the step.

const _GRID_FIELDS = (
    (name = :ρ, short = "rho", units = "kg m^-3 s^-1",
        long = "air mass",
        processes = (:iva, :hor_dyn, :hdiff, :additional)),
    (name = :ρe_tot, short = "rhoe", units = "W m^-3",
        long = "total air energy",
        processes = (:iva, :hor_dyn, :exp_vadv, :hdiff, :additional)),
    (name = :ρq_tot, short = "rhoqt", units = "kg m^-3 s^-1",
        long = "total air water",
        processes = (:iva, :hor_tracer_adv, :exp_vadv, :hdiff, :additional)),
    (name = :ρq_lcl, short = "rhoqlcl", units = "kg m^-3 s^-1",
        long = "cloud liquid mass",
        processes = (:iva, :hor_tracer_adv, :hdiff, :additional)),
    (name = :ρq_icl, short = "rhoqicl", units = "kg m^-3 s^-1",
        long = "cloud ice mass",
        processes = (:iva, :hor_tracer_adv, :hdiff, :additional)),
    (name = :ρq_rai, short = "rhoqr", units = "kg m^-3 s^-1",
        long = "rain water mass",
        processes = (:iva, :hor_tracer_adv, :additional)),
    (name = :ρq_sno, short = "rhoqs", units = "kg m^-3 s^-1",
        long = "snow mass",
        processes = (:iva, :hor_tracer_adv, :additional)),
)

const _SGS_FIELDS = (
    (name = :mse, short = "sgsmse", units = "J kg^-1 s^-1",
        long = "updraft mse", processes = (:hdiff, :additional)),
    (name = :q_tot, short = "sgsqt", units = "s^-1",
        long = "updraft q_tot", processes = (:hdiff, :additional)),
    (name = :q_lcl, short = "sgsqlcl", units = "s^-1",
        long = "updraft q_lcl", processes = (:hdiff, :additional)),
    (name = :q_icl, short = "sgsqicl", units = "s^-1",
        long = "updraft q_icl", processes = (:hdiff, :additional)),
    (name = :q_rai, short = "sgsqr", units = "s^-1",
        long = "updraft q_rai", processes = (:additional,)),
    (name = :q_sno, short = "sgsqs", units = "s^-1",
        long = "updraft q_sno", processes = (:additional,)),
)

# Instantaneous ARK processes of a manifest entry followed by every realized
# operator.
_all_processes(f) = (f.processes..., REALIZED_OPERATORS...)

# CMIP-flavoured short name: `tn` (tendency) prefix, then the field short
# and the process short separated by underscores. The process short itself
# has its underscores stripped (`hor_dyn` → `hordyn`,
# `hor_tracer_adv` → `hortraceradv`) so each of the three segments is a
# single alphanumeric token: `tn_<field>_<process>`.
_process_short(pname) = replace(String(pname), "_" => "")
_tendency_short(field_short, pname) = "tn_$(field_short)_$(_process_short(pname))"

function _process_comment(field_name, pname; sgs = false)
    target = sgs ? "d($(field_name))/dt for updraft j=1" : "d($(field_name))/dt"
    return if pname in REALIZED_OPERATORS
        "$target realized by the $(pname) operator over the most recent \
         step, (Y_after - Y_before) / Δt (3D field). Debug diagnostic: \
         records the state around every operator of every step."
    else
        "$target for the $(pname) process alone, evaluated at the current \
         state (3D field). Debug diagnostic: expensive; runs one tendency \
         evaluation per sample."
    end
end

# -- Registration ----------------------------------------------------------

for f in _GRID_FIELDS
    field_path = MatrixFields.FieldName{(:c, f.name)}()
    for pname in _all_processes(f)
        add_diagnostic_variable!(
            short_name = _tendency_short(f.short, pname),
            long_name = "$(f.long) tendency from process $(pname)",
            units = f.units,
            comments = _process_comment(f.name, pname),
            compute = (state, cache, time) ->
                _process_tendency_field(pname, state, cache, time, field_path),
        )
    end
end

for f in _SGS_FIELDS
    scalar_name = f.name
    for pname in _all_processes(f)
        add_diagnostic_variable!(
            short_name = _tendency_short(f.short, pname),
            long_name = "$(f.long) tendency from process $(pname)",
            units = f.units,
            comments = _process_comment(f.name, pname; sgs = true),
            compute = (state, cache, time) -> _sgs_process_tendency_field(
                pname, state, cache, time, scalar_name,
            ),
        )
    end
end

"""
    tendency_debug_short_names()

Return a `Vector{String}` of all short names registered by
`tendency_diagnostics.jl`. Useful for enabling the full set as scheduled
diagnostics or for plotting.
"""
function tendency_debug_short_names()
    names = String[]
    for f in _GRID_FIELDS, pname in _all_processes(f)
        push!(names, _tendency_short(f.short, pname))
    end
    for f in _SGS_FIELDS, pname in _all_processes(f)
        push!(names, _tendency_short(f.short, pname))
    end
    return names
end

"""
    tendency_debug_field_manifest()

Return a `Vector{NamedTuple}` with one entry per prognostic field for which
tendency-debug diagnostics exist. Each entry has fields:

  - `short::String`: field short-name prefix (e.g. `"rhoe"`, `"sgsqt"`).
  - `long::String`: human-readable field description.
  - `units::String`: units of the tendency.
  - `processes::Vector{Symbol}`: instantaneous ARK processes contributing a
    non-zero tendency, followed by the realized operators
    (`REALIZED_OPERATORS`).
  - `kind::Symbol`: `:grid` or `:sgs`.
"""
function tendency_debug_field_manifest()
    entries = NamedTuple[]
    for (fields, kind) in ((_GRID_FIELDS, :grid), (_SGS_FIELDS, :sgs))
        for f in fields
            push!(
                entries,
                (;
                    short = String(f.short),
                    long = String(f.long),
                    units = String(f.units),
                    processes = Symbol[p for p in _all_processes(f)],
                    kind,
                ),
            )
        end
    end
    return entries
end

"""
    tendency_debug_default_diagnostics(output_writer, duration, start_date, t_start)

Return a `Vector{ScheduledDiagnostic}` covering every tendency-debug
diagnostic short name at the cadence chosen by `frequency_averages(duration)`.
Included in the default diagnostic set when `DiagnosticsConfig.debug_tendency`
is `true`.
"""
function tendency_debug_default_diagnostics(
    output_writer, duration, start_date, t_start,
)
    short_names = tendency_debug_short_names()
    average_func = frequency_averages(duration)
    scheduled = average_func(short_names...; output_writer, start_date, t_start)
    return collect(scheduled)
end
