# Per-process tendency diagnostics (3D fields).
#
# For each (process, prognostic-field) pair listed in `_GRID_FIELDS` /
# `_SGS_FIELDS`, this file registers a diagnostic that
#
#   1. allocates a scratch `Yₜ = zero(Y)`,
#   2. calls just that one process's tendency function on `(Yₜ, Y, p, t)`,
#   3. returns the tendency field `Yₜ.c.<field>` (or `Yₜ.c.sgsʲs.:(1).<scalar>`
#      for SGS updraft-scalar diagnostics) as a 3D center-space Field.
#
# Grid-scale diagnostics have units of `field · m^-3 · s^-1`. SGS
# updraft-scalar diagnostics have per-mass rates (`J kg^-1 s^-1` for `mse`,
# `s^-1` for `q_*`).
#
# Diagnostic short-name convention: `tn_<field-short>_<process-short>`
# (CMIP-flavoured tendency prefix, e.g. `tn_rhoe_iva`,
# `tn_sgsqt_entrdetr`). Updraft ρa's aggregate tendency uses the special
# `:aggregate` process wrapping `sgs_ρa_implicit_tendency!` (short name
# `tn_sgsrhoa_aggregate`).
#
# The process short-name strips underscores from the manifest key
# (`hor_dyn` → `hordyn`, `edmf_diff` → `edmfdiff`, `sgs_vadv` → `sgsvadv`,
# etc.) so each of the three underscore-delimited segments is a single
# alphanumeric token.
#
# These are debug-oriented and expensive relative to standard diagnostics:
# each sample runs one full tendency evaluation into a persistent scratch
# `FieldVector`. Enable only via the `debug_tendency_diagnostics` config
# flag and use a coarse output cadence.

import ClimaCore.MatrixFields
import ClimaCore.Operators

# Tendency functions live in the parent ClimaAtmos module.
import ..implicit_vertical_advection_tendency!
import ..correct_implicit_advection_tendency!
import ..horizontal_dynamics_tendency!
import ..horizontal_tracer_advection_tendency!
import ..explicit_vertical_advection_tendency!
import ..hyperdiffusion_tendency!
import ..additional_tendency!
import ..microphysics_tendency!
import ..edmfx_sgs_vertical_advection_tendency!
import ..vertical_diffusion_boundary_layer_tendency!
import ..edmfx_sgs_diffusive_flux_tendency!
import ..edmfx_entr_detr_tendency!
import ..edmfx_sgs_mass_flux_tendency!
import ..edmfx_boundary_condition_tendency!
import ..pressure_work_tendency!
# Used by the aggregate ρa diagnostic below: this function reads the
# analytically pre-aggregated `ᶜρa_tendencyʲs` from `p.precomputed`.
import ..sgs_ρa_implicit_tendency!
import ..do_dss

# -- Persistent scratches ---------------------------------------------------
# The compute functions below need a `Y`-sized tendency `FieldVector` per
# call. Allocating one via `similar(state)` on every diagnostic sample
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

# -- Process wrappers -------------------------------------------------------
# Each wrapper takes `(Yₜ, Y, p, t)` and accumulates a single process's
# contribution into `Yₜ`. The caller is responsible for zeroing `Yₜ` first.

_iva_wrap!(Yₜ, Y, p, t) = implicit_vertical_advection_tendency!(Yₜ, Y, p, t)

_corrimp_wrap!(Yₜ, Y, p, t) =
    correct_implicit_advection_tendency!(Yₜ, Y, p, t)

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

_microphys_wrap!(Yₜ, Y, p, t) = microphysics_tendency!(
    Yₜ, Y, p, t, p.atmos.microphysics_model, p.atmos.turbconv_model,
)

_sgs_vadv_wrap!(Yₜ, Y, p, t) = edmfx_sgs_vertical_advection_tendency!(
    Yₜ, Y, p, t, p.atmos.turbconv_model,
)

_vdiff_wrap!(Yₜ, Y, p, t) = vertical_diffusion_boundary_layer_tendency!(
    Yₜ, Y, p, t, p.atmos.vertical_diffusion,
)

_edmf_diff_wrap!(Yₜ, Y, p, t) = edmfx_sgs_diffusive_flux_tendency!(
    Yₜ, Y, p, t, p.atmos.turbconv_model,
)

_entrdetr_wrap!(Yₜ, Y, p, t) = edmfx_entr_detr_tendency!(
    Yₜ, Y, p, t, p.atmos.turbconv_model,
)

_edmf_mflux_wrap!(Yₜ, Y, p, t) = edmfx_sgs_mass_flux_tendency!(
    Yₜ, Y, p, t, p.atmos.turbconv_model,
)

_edmf_bc_wrap!(Yₜ, Y, p, t) = edmfx_boundary_condition_tendency!(
    Yₜ, Y, p, t, p.atmos.turbconv_model,
)

_pwork_wrap!(Yₜ, Y, p, t) = pressure_work_tendency!(
    Yₜ, Y, p, t, p.atmos.turbconv_model,
)

# Aggregate wrapper for updraft ρa: `sgs_ρa_implicit_tendency!` reads the
# analytically pre-aggregated `ᶜρa_tendencyʲs` from `p.precomputed`, which is
# not populated at `t = 0` (would give huge spurious values). We treat that
# initial sample as zero.
function _aggregate_wrap!(Yₜ, Y, p, t)
    iszero(t) && return Yₜ
    return sgs_ρa_implicit_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
end

const _PROCESS_WRAPPERS = (
    iva = _iva_wrap!,
    corrimp = _corrimp_wrap!,
    hor_dyn = _hor_dyn_wrap!,
    hor_tracer_adv = _hor_tracer_adv_wrap!,
    exp_vadv = _exp_vadv_wrap!,
    hdiff = _hdiff_wrap!,
    additional = _additional_wrap!,
    microphys = _microphys_wrap!,
    sgs_vadv = _sgs_vadv_wrap!,
    vdiff = _vdiff_wrap!,
    edmf_diff = _edmf_diff_wrap!,
    entrdetr = _entrdetr_wrap!,
    edmf_mflux = _edmf_mflux_wrap!,
    edmf_bc = _edmf_bc_wrap!,
    pwork = _pwork_wrap!,
    aggregate = _aggregate_wrap!,
)

# -- Column-integration helpers --------------------------------------------

# Run `process_wrapper!` on the persistent scratch Yₜ and return the
# tendency field at `field_path` as a 3D center-space Field. Returns a lazy
# zero field on center space if the field is not present in the current
# config. Downstream (e.g. NetCDFWriter) copies the values before the next
# diagnostic call overwrites the scratch.
#
# Horizontal tendencies (`hor_dyn`, `hor_tracer_adv`, `hdiff`, and some
# sub-terms of `additional`) leave the field discontinuous across spectral
# element boundaries; we apply `weighted_dss!` so the diagnostic output is
# a well-defined continuous field. For vertical-only tendencies DSS is a
# no-op (co-located values on shared edges are already equal because the
# input state is continuous), so applying it unconditionally is safe and
# saves plumbing a per-process flag.
function _process_tendency_field(
    process_wrapper!, state, cache, time, field_path,
)
    if !MatrixFields.has_field(state, field_path)
        return @. lazy(state.c.ρ * 0)  # 3D center-space zero
    end
    Yₜ = _acquire_scratch(_SCRATCH_Yₜ, state)
    process_wrapper!(Yₜ, state, cache, time)
    field = MatrixFields.get_field(Yₜ, field_path)
    if do_dss(axes(state.c))
        Spaces.weighted_dss!(field)
    end
    return field
end

# SGS variant: return the 3D `d(scalar)/dt` field for updraft `j = 1`.
function _sgs_process_tendency_field(
    process_wrapper!, state, cache, time, scalar_name,
)
    if !hasproperty(state.c, :sgsʲs) ||
       !hasproperty(state.c.sgsʲs.:(1), scalar_name)
        return @. lazy(state.c.ρ * 0)
    end
    Yₜ = _acquire_scratch(_SCRATCH_Yₜ, state)
    process_wrapper!(Yₜ, state, cache, time)
    field = getproperty(Yₜ.c.sgsʲs.:(1), scalar_name)
    if do_dss(axes(state.c))
        Spaces.weighted_dss!(field)
    end
    return field
end

# -- Field × process manifests ---------------------------------------------
# Each entry lists exactly the processes known to contribute non-zero
# tendency to that field. Processes that would be no-ops are omitted so we
# don't register useless diagnostic names.

const _GRID_FIELDS = (
    (name = :ρ, short = "rho", units = "kg m^-3 s^-1",
        long = "air mass",
        processes = (:iva, :hor_dyn, :vdiff, :edmf_diff, :edmf_mflux, :hdiff,
            :additional)),
    (name = :ρe_tot, short = "rhoe", units = "W m^-3",
        long = "total air energy",
        processes = (:iva, :corrimp, :hor_dyn, :exp_vadv, :vdiff, :edmf_diff,
            :edmf_mflux, :hdiff, :microphys, :additional)),
    (name = :ρq_tot, short = "rhoqt", units = "kg m^-3 s^-1",
        long = "total air water",
        processes = (:iva, :corrimp, :hor_tracer_adv, :exp_vadv, :vdiff,
            :edmf_diff, :edmf_mflux, :hdiff, :microphys, :additional)),
    (name = :ρq_lcl, short = "rhoqlcl", units = "kg m^-3 s^-1",
        long = "cloud liquid mass",
        processes = (:iva, :hor_tracer_adv, :vdiff, :edmf_diff, :edmf_mflux,
            :hdiff, :microphys, :additional)),
    (name = :ρq_icl, short = "rhoqicl", units = "kg m^-3 s^-1",
        long = "cloud ice mass",
        processes = (:iva, :hor_tracer_adv, :vdiff, :edmf_diff, :edmf_mflux,
            :hdiff, :microphys, :additional)),
    (name = :ρq_rai, short = "rhoqr", units = "kg m^-3 s^-1",
        long = "rain water mass",
        processes = (:iva, :hor_tracer_adv, :edmf_diff, :edmf_mflux,
            :microphys, :additional)),
    (name = :ρq_sno, short = "rhoqs", units = "kg m^-3 s^-1",
        long = "snow mass",
        processes = (:iva, :hor_tracer_adv, :edmf_diff, :edmf_mflux,
            :microphys, :additional)),
)

const _SGS_FIELDS = (
    (name = :mse, short = "sgsmse", units = "J kg^-1 s^-1",
        long = "updraft mse",
        processes = (:sgs_vadv, :entrdetr, :edmf_diff, :edmf_bc, :hdiff,
            :microphys, :additional)),
    (name = :q_tot, short = "sgsqt", units = "s^-1",
        long = "updraft q_tot",
        processes = (:sgs_vadv, :entrdetr, :edmf_diff, :edmf_bc, :hdiff,
            :microphys, :additional)),
    (name = :q_lcl, short = "sgsqlcl", units = "s^-1",
        long = "updraft q_lcl",
        processes = (:sgs_vadv, :entrdetr, :edmf_diff, :edmf_bc, :hdiff,
            :microphys, :additional)),
    (name = :q_icl, short = "sgsqicl", units = "s^-1",
        long = "updraft q_icl",
        processes = (:sgs_vadv, :entrdetr, :edmf_diff, :edmf_bc, :hdiff,
            :microphys, :additional)),
    (name = :q_rai, short = "sgsqr", units = "s^-1",
        long = "updraft q_rai",
        processes = (:sgs_vadv, :entrdetr, :edmf_diff, :edmf_bc, :microphys,
            :additional)),
    (name = :q_sno, short = "sgsqs", units = "s^-1",
        long = "updraft q_sno",
        processes = (:sgs_vadv, :entrdetr, :edmf_diff, :edmf_bc, :microphys,
            :additional)),
    # Aggregate updraft ρa tendency — the single `:aggregate` "process"
    # wraps `sgs_ρa_implicit_tendency!`, which reads the analytically
    # pre-aggregated `ᶜρa_tendencyʲs` from `p.precomputed`.
    (name = :ρa, short = "sgsrhoa", units = "kg m^-3 s^-1",
        long = "updraft ρa",
        processes = (:aggregate,)),
)

# CMIP-flavoured short name: `tn` (tendency) prefix, then the field short
# and the process short separated by underscores. The process short itself
# has its underscores stripped (`edmf_diff` → `edmfdiff`,
# `hor_tracer_adv` → `hortraceradv`) so each of the three segments is a
# single alphanumeric token: `tn_<field>_<process>`.
_process_short(pname) = replace(String(pname), "_" => "")
_tendency_short(field_short, pname) = "tn_$(field_short)_$(_process_short(pname))"

# -- Registration ----------------------------------------------------------

for f in _GRID_FIELDS
    field_path = MatrixFields.FieldName{(:c, f.name)}()
    for pname in f.processes
        wrap! = _PROCESS_WRAPPERS[pname]
        add_diagnostic_variable!(
            short_name = _tendency_short(f.short, pname),
            long_name = "$(f.long) tendency from process $(pname)",
            units = f.units,
            comments = "d($(f.name))/dt for the $(pname) process alone (3D field). \
                        Debug diagnostic: expensive; runs one full tendency evaluation per sample.",
            compute = (state, cache, time) ->
                _process_tendency_field(
                    wrap!, state, cache, time, field_path,
                ),
        )
    end
end

for f in _SGS_FIELDS
    scalar_name = f.name
    for pname in f.processes
        wrap! = _PROCESS_WRAPPERS[pname]
        add_diagnostic_variable!(
            short_name = _tendency_short(f.short, pname),
            long_name = "$(f.long) tendency from process $(pname)",
            units = f.units,
            comments = "d($(f.name))/dt for updraft j=1, $(pname) process alone (3D field). \
                        Debug diagnostic: expensive; runs one full tendency evaluation per sample.",
            compute = (state, cache, time) ->
                _sgs_process_tendency_field(
                    wrap!, state, cache, time, scalar_name,
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
    for f in _GRID_FIELDS, pname in f.processes
        push!(names, _tendency_short(f.short, pname))
    end
    for f in _SGS_FIELDS, pname in f.processes
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
  - `processes::Vector{Symbol}`: process names contributing non-zero tendency.
  - `kind::Symbol`: `:grid` or `:sgs`.
"""
function tendency_debug_field_manifest()
    entries = NamedTuple[]
    for f in _GRID_FIELDS
        push!(
            entries,
            (;
                short = String(f.short),
                long = String(f.long),
                units = String(f.units),
                processes = Symbol[p for p in f.processes],
                kind = :grid,
            ),
        )
    end
    for f in _SGS_FIELDS
        push!(
            entries,
            (;
                short = String(f.short),
                long = String(f.long),
                units = String(f.units),
                processes = Symbol[p for p in f.processes],
                kind = :sgs,
            ),
        )
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
