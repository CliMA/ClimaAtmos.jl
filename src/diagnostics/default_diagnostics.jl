# This file is included by Diagnostics.jl and defines all the defaults for various models. A
# model here is either a global AtmosModel, or small (sub)models (e.g., DryModel()).
#
# If you are developing new models, add your defaults here. If you want to add more high
# level interfaces, add them here. Feel free to include extra files.

"""
    default_diagnostics(model::AtmosModel, duration, start_date, t_start;
                        output_writer, topography)
    default_diagnostics(submodel, duration, start_date, t_start; output_writer)

Return the default `ScheduledDiagnostic`s for `model`, written through `output_writer`.

The `AtmosModel` method returns the core diagnostics (see `core_default_diagnostics`)
together with the defaults of every submodel field of `AtmosModel` that defines them.
Submodels dispatch on their own type; the fallback method returns `[]`, so submodels
without defaults contribute nothing. Note that `topography` is consumed by the
`AtmosModel` method only and is not forwarded to the submodel methods.

Dates are assigned to simulation times as

```julia
current_date = start_date + integrator.t
```

# Arguments

  - `model`: The `AtmosModel`, or one of its submodels (e.g. `EquilibriumMicrophysics0M`).
  - `duration`: Expected duration of the simulation [s]. Sets the output frequency, via
    `frequency_averages`.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation, not necessarily zero [s].

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `topography`: Whether the grid has topography. When `true`, the `AtmosModel` method
    prepends a single instantaneous `orog` diagnostic. Required by the `AtmosModel` method
    and not accepted by the submodel methods.

# Returns

A `Vector` of `ClimaDiagnostics.ScheduledDiagnostic`, possibly empty.

# Examples

```julia
import ClimaAtmos as CA
import Dates

diagnostics = CA.Diagnostics.default_diagnostics(
    CA.AtmosModel(),
    86400 * 10,                    # 10-day simulation
    Dates.DateTime(2010, 1, 1),
    0;
    output_writer = writer,
    topography = false,
)
```
"""
function default_diagnostics(
    model::AtmosModel,
    duration,
    start_date::DateTime,
    t_start;
    output_writer,
    topography,
)
    # Unfortunately, [] is not treated nicely in a map (we would like it to be "excluded"),
    # so we need to manually filter out the submodels that don't have defaults associated
    # to
    non_empty_fields = filter(
        x ->
            default_diagnostics(
                getfield(model, x),
                duration,
                start_date,
                t_start;
                output_writer,
            ) != [],
        fieldnames(AtmosModel),
    )

    # We use a map because we want to ensure that diagnostics is a well defined type, not
    # Any. This reduces latency.
    return vcat(
        core_default_diagnostics(output_writer, duration, start_date, t_start, topography),
        map(non_empty_fields) do field
            default_diagnostics(
                getfield(model, field),
                duration,
                start_date,
                t_start;
                output_writer,
            )
        end...,
    )
end

# Base case: if we call default_diagnostics on something that we don't have information
# about, we get nothing back (to be specific, we get an empty list, so that we can assume
# that all the default_diagnostics return the same type). This is used by
# default_diagnostics(model::AtmosModel; output_writer), so that we can ignore defaults for
# submodels that have no given defaults.
default_diagnostics(submodel, duration, start_date, t_start; output_writer) = []

"""
    common_diagnostics(period, reduction, output_writer, start_date, t_start,
                       short_names...; pre_output_hook! = nothing)

Build one `ScheduledDiagnostic` per short name, reduced over `period`.

Shared backend of the standard frequency helpers such as `daily_max` and
`monthly_averages`. The output schedule fires every `period`; the compute schedule comes
from `make_compute_schedule`, which computes every step for short periods and less often
for long ones. The reduction is seeded from `t_start` so that restarted runs accumulate
over the same calendar windows.

# Arguments

  - `period`: `Dates.Period` over which the reduction accumulates (e.g. `Dates.Day(1)`).
  - `reduction`: Binary reduction applied in time, e.g. `max`, `min`, or `(+)`.
  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `pre_output_hook! = nothing`: Hook applied to the accumulator before writing. Averages
    pass `average_pre_output_hook!` to normalize the accumulated sum by the sample count.

# Returns

A `Vector` of `ClimaDiagnostics.ScheduledDiagnostic`, one per short name.
"""
function common_diagnostics(
    period,
    reduction,
    output_writer,
    start_date,
    t_start,
    short_names...;
    pre_output_hook! = nothing,
)
    date_last =
        t_start isa ClimaUtilities.TimeManager.ITime ?
        ClimaUtilities.TimeManager.date(t_start) :
        start_date + Dates.Second(t_start)
    return vcat(
        map(short_names) do short_name
            variable = get_diagnostic_variable(short_name)
            return ScheduledDiagnostic(;
                variable,
                compute_schedule_func = make_compute_schedule(variable, period,
                    start_date,
                    date_last),
                output_schedule_func = EveryCalendarDtSchedule(
                    period;
                    start_date,
                    date_last = date_last,
                ),
                reduction_time_func = reduction,
                output_writer = output_writer,
                pre_output_hook! = pre_output_hook!,
            )
        end...,
    )
end

#! format: off
"""
    HOURLY_DIAGS

Short names computed hourly, rather than six-hourly, on long output periods.

These are the precipitation and radiation variables, whose sub-daily variability would
otherwise be aliased by a six-hourly sampling of a monthly or multi-day mean. Consulted by
`make_compute_schedule`.
"""
const HOURLY_DIAGS = Set([
    "pr", "prra", "prsn", "prw", "rsd", "rsdt", "rsds", "rsu", "rsut", "rsus", "rld",
    "rlds", "rlu", "rlut", "rlus", "rsdcs", "rsdscs", "rsucs", "rsutcs", "rsuscs", "rldcs",
    "rldscs", "rldscs", "rlucs", "rlutcs"
    ]
)
#! format: on

"""
    make_compute_schedule(variable, period, start_date, date_last)

Return the compute schedule to pair with an output `period` for `variable`.

Short output periods — anything other than a month, a week, or several days, so including
hourly and daily — are computed every timestep. Longer output periods subsample: variables
in `HOURLY_DIAGS` are computed hourly, all others six-hourly.

# Arguments

  - `variable`: The `ClimaDiagnostics.DiagnosticVariable` being scheduled.
  - `period`: `Dates.Period` of the output schedule.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `date_last`: Date from which the schedule counts, i.e. the date of `t_start`.

# Returns

An `EveryStepSchedule`, or an `EveryCalendarDtSchedule` of one or six hours.
"""
function make_compute_schedule(variable, period, start_date, date_last)
    if !(
        period isa Dates.Month || period isa Dates.Week ||
        (period isa Dates.Day && Dates.value(period) > 1)
    )
        return EveryStepSchedule()
    end
    short_name = ClimaDiagnostics.DiagnosticVariables.short_name(variable)
    compute_every = short_name in HOURLY_DIAGS ? Dates.Hour(1) : Dates.Hour(6)
    return EveryCalendarDtSchedule(
        compute_every;
        start_date,
        date_last = date_last,
    )
end

include("standard_diagnostic_frequencies.jl")

"""
    frequency_averages(duration)

Return the time-averaging helper appropriate to the total simulation length.

The returned closure has the signature `(short_names...; kwargs...)`, forwarding to one of
the `*_averages` helpers with `FT = eltype(duration)` already applied:

  - `duration < 1 hour`: returns an empty tuple, i.e. no averaged diagnostics.
  - `1 hour ≤ duration < 1 day`: `hourly_averages`.
  - `1 day ≤ duration < 30 days`: `daily_averages`.
  - `30 days ≤ duration < 90 days`: `tendaily_averages`.
  - `duration ≥ 90 days`: `monthly_averages`.

# Arguments

  - `duration`: Expected duration of the simulation [s].
"""
function frequency_averages(duration)
    FT = eltype(duration)
    duration = Float64(duration)
    if duration >= 90 * 86400
        return (args...; kwargs...) -> monthly_averages(FT, args...; kwargs...)
    elseif duration >= 30 * 86400
        return (args...; kwargs...) -> tendaily_averages(FT, args...; kwargs...)
    elseif duration >= 86400
        return (args...; kwargs...) -> daily_averages(FT, args...; kwargs...)
    elseif duration >= 3600
        return (args...; kwargs...) -> hourly_averages(FT, args...; kwargs...)
    else
        return (args...; kwargs...) -> ()
    end
end

# Include all the subdefaults

########
# Core #
########
"""
    core_default_diagnostics(output_writer, duration, start_date, t_start, topography)

Return the model-independent default diagnostics.

These are time averages of the core dynamical and surface-flux variables, plus the minimum
and maximum of the surface temperature `ts`, all at the frequency chosen by
`frequency_averages` for `duration`. When `topography` is `true`, a single instantaneous
surface-altitude diagnostic (`orog`, written as `orog_inst`) is prepended; its schedules
never fire again, so it is written once at initialization.

Called from `default_diagnostics`.
"""
function core_default_diagnostics(output_writer, duration, start_date, t_start, topography)
    core_diagnostics = [
        "ts",
        "ta",
        "tas",
        "uas",
        "vas",
        "thetaa",
        "msea",
        "pfull",
        "zg",
        "rhoa",
        "ua",
        "va",
        "wa",
        "hfes",
        "hfss",
    ]

    average_func = frequency_averages(duration)
    FT = eltype(duration)

    duration = Float64(duration)

    if duration >= 90 * 86400
        min_func = (args...; kwargs...) -> monthly_min(FT, args...; kwargs...)
        max_func = (args...; kwargs...) -> monthly_max(FT, args...; kwargs...)
    elseif duration >= 30 * 86400
        min_func = (args...; kwargs...) -> tendaily_min(FT, args...; kwargs...)
        max_func = (args...; kwargs...) -> tendaily_max(FT, args...; kwargs...)
    elseif duration >= 86400
        min_func = (args...; kwargs...) -> daily_min(FT, args...; kwargs...)
        max_func = (args...; kwargs...) -> daily_max(FT, args...; kwargs...)
    else
        min_func = (args...; kwargs...) -> hourly_min(FT, args...; kwargs...)
        max_func = (args...; kwargs...) -> hourly_max(FT, args...; kwargs...)
    end
    # Base diagnostics for all cases
    base_diagnostics = [
        average_func(core_diagnostics...; output_writer, start_date, t_start)...,
        min_func("ts"; output_writer, start_date, t_start),
        max_func("ts"; output_writer, start_date, t_start),
    ]

    # Prepend orography diagnostic if topography is enabled
    if topography
        orog_diagnostic = ScheduledDiagnostic(;
            variable = get_diagnostic_variable("orog"),
            output_schedule_func = (integrator) -> false,
            compute_schedule_func = (integrator) -> false,
            output_writer,
            output_short_name = "orog_inst",
        )
        return [orog_diagnostic, base_diagnostics...]
    else
        return base_diagnostics
    end
end

######################
# Microphysics model #
######################

"""
    _moist_default_diagnostics(duration, start_date, t_start; output_writer)

Return the time-averaged moisture, cloud, and precipitation diagnostics common to every
microphysics model. Called from the `default_diagnostics` microphysics methods, which
append their own scheme-specific variables.
"""
function _moist_default_diagnostics(duration, start_date, t_start; output_writer)
    moist_diagnostics = [
        "hur",
        "hus",
        "cl",
        "clw",
        "cli",
        "hussfc",
        "evspsbl",
        "hfls",
        "pr",
        "prra",
        "prsn",
        "prw",
        "lwp",
        "iwp",
        "clwvi",
        "clivi",
    ]
    average_func = frequency_averages(duration)
    return [average_func(moist_diagnostics...; output_writer, start_date, t_start)...]
end

function default_diagnostics(
    ::EquilibriumMicrophysics0M,
    duration,
    start_date,
    t_start;
    output_writer,
)
    return _moist_default_diagnostics(duration, start_date, t_start; output_writer)
end

function default_diagnostics(
    ::NonEquilibriumMicrophysics1M,
    duration,
    start_date,
    t_start;
    output_writer,
)
    precip_diagnostics = ["husra", "hussn", "rwp", "swp"]
    average_func = frequency_averages(duration)
    return [
        _moist_default_diagnostics(duration, start_date, t_start; output_writer)...,
        average_func(precip_diagnostics...; output_writer, start_date, t_start)...,
    ]
end

function default_diagnostics(
    ::NonEquilibriumMicrophysics2M,
    duration,
    start_date,
    t_start;
    output_writer,
)
    precip_diagnostics = ["husra", "hussn", "rwp", "swp", "cdnc", "ncra"]
    average_func = frequency_averages(duration)
    return [
        _moist_default_diagnostics(duration, start_date, t_start; output_writer)...,
        average_func(precip_diagnostics...; output_writer, start_date, t_start)...,
    ]
end

function default_diagnostics(
    ::NonEquilibriumMicrophysics2MP3,
    duration,
    start_date,
    t_start;
    output_writer,
)
    return _moist_default_diagnostics(duration, start_date, t_start; output_writer)
end

function default_diagnostics(
    atmos_water::AtmosWater,
    duration,
    start_date,
    t_start;
    output_writer,
)
    if !isnothing(atmos_water.microphysics_model)
        return default_diagnostics(
            atmos_water.microphysics_model,
            duration,
            start_date,
            t_start;
            output_writer,
        )
    else
        return []
    end
end

##################
# Radiation mode #
##################
"""
    _radiation_default_diagnostics(duration, start_date, t_start; output_writer)

Return the time-averaged all-sky TOA and surface radiative fluxes common to every RRTMGP
radiation mode. Called from the `default_diagnostics` radiation methods, which append
cloud cover and, for `AllSkyRadiationWithClearSkyDiagnostics`, the clear-sky fluxes.
"""
function _radiation_default_diagnostics(duration, start_date, t_start; output_writer)
    rad_diagnostics = [
        "rsdt",
        "rsds",
        "rsut",
        "rsus",
        "rlds",
        "rlut",
        "rlus",
    ]

    average_func = frequency_averages(duration)

    return [average_func(rad_diagnostics...; output_writer, start_date, t_start)...]
end

function default_diagnostics(
    ::RRTMGPI.AbstractRRTMGPMode,
    duration,
    start_date,
    t_start;
    output_writer,
)
    return _radiation_default_diagnostics(duration, start_date, t_start; output_writer)
end

function default_diagnostics(
    ::RRTMGPI.AllSkyRadiation,
    duration,
    start_date,
    t_start;
    output_writer,
)
    average_func = frequency_averages(duration)

    return [
        _radiation_default_diagnostics(duration, start_date, t_start; output_writer)...,
        average_func("clt"; output_writer, start_date, t_start)...,
    ]
end

function default_diagnostics(
    ::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
    duration,
    start_date,
    t_start;
    output_writer,
)
    average_func = frequency_averages(duration)
    rad_clearsky_diagnostics = [
        "rsdscs",
        "rsutcs",
        "rsuscs",
        "rldscs",
        "rlutcs",
    ]

    return [
        _radiation_default_diagnostics(duration, start_date, t_start; output_writer)...,
        average_func(
            "clt",
            rad_clearsky_diagnostics...;
            output_writer,
            start_date,
            t_start,
        )...,
    ]
end

##################
# Turbconv model #
##################
function default_diagnostics(
    ::PrognosticEDMFX,
    duration,
    start_date,
    t_start;
    output_writer,
)
    edmfx_draft_diagnostics = [
        "arup",
        "rhoaup",
        "waup",
        "taup",
        "thetaaup",
        "haup",
        "husup",
        "hurup",
        "clwup",
        "cliup",
    ]
    edmfx_env_diagnostics = [
        "aren",
        "rhoaen",
        "waen",
        "taen",
        "thetaaen",
        "haen",
        "husen",
        "huren",
        "clwen",
        "clien",
        "tke",
        "lmix",
        "evu",
        "edt",
    ]

    average_func = frequency_averages(duration)

    return [
        average_func(edmfx_draft_diagnostics...; output_writer, start_date, t_start)...,
        average_func(edmfx_env_diagnostics...; output_writer, start_date, t_start)...,
    ]
end


function default_diagnostics(::EDOnlyEDMFX, duration, start_date, t_start; output_writer)
    edonly_edmfx_diagnostics = ["tke", "lmix", "evu", "edt"]

    average_func = frequency_averages(duration)

    return [
        average_func(edonly_edmfx_diagnostics...; output_writer, start_date, t_start)...,
    ]
end

function default_diagnostics(
    atmos_radiation::AtmosRadiation,
    duration,
    start_date,
    t_start;
    output_writer,
)
    # Add radiation mode diagnostics
    if !isnothing(atmos_radiation.radiation_mode)
        return default_diagnostics(
            atmos_radiation.radiation_mode,
            duration,
            start_date,
            t_start;
            output_writer,
        )
    else
        return []
    end
end

function default_diagnostics(
    atmos_turbconv::AtmosTurbconv,
    duration,
    start_date,
    t_start;
    output_writer,
)
    # Add turbulence convection model diagnostics
    if !isnothing(atmos_turbconv.turbconv_model)
        return default_diagnostics(
            atmos_turbconv.turbconv_model,
            duration,
            start_date,
            t_start;
            output_writer,
        )
    else
        return []
    end
end
