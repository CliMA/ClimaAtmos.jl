# This file is included by Diagnostics.jl and defines all the defaults for various models. A
# model here is either a global AtmosModel, or small (sub)models (e.g., DryModel()).
#
# If you are developing new models, add your defaults here. If you want to add more high
# level interfaces, add them here. Feel free to include extra files.

"""
    default_diagnostics(model, duration, start_date, t_start; output_writer)

Return a list of `ScheduledDiagnostic`s associated with the given `model` that use
`output_write` to write to disk. `duration` is the expected duration of the simulation and
it is used to choose the most reasonable output frequency.

`start_date` is the date that we assign at the start of the simulation.
We convert time to date as

```julia
current_date = start_date + integrator.t
```

`t_start` is the start time of the simulation and `t_start` is not necessarily zero.

The logic is as follows:

If `duration < 1 day` take hourly means,
if `duration < 30 days` take daily means,
if `duration < 90 days` take means over ten days,
If `duration >= 90 year` take monthly means.
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
    produce_common_diagnostic_function(period, reduction)

Helper function to define functions like `daily_max`.
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
                    reference_date = start_date,
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
    A list of short names of diagnostic that are computed every hour for longer
    simulations. The diagnostics conist of precipitation and radiation
    variables. This is used in `make_compute_schedule`.
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

Return an appropriate compute schedule for the given `variable` and output
`period`.

For shorter output periods (e.g., hourly), diagnostics are computed every
timestep. For longer output periods (e.g., monthly and daily), diagnostics are
computed less frequently.
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
        reference_date = start_date,
        date_last = date_last,
    )
end

include("standard_diagnostic_frequencies.jl")

"""
    frequency_averages(duration::Real)

Return the correct averaging function depending on the total simulation time.

If `duration < 1 hour` do nothing,
If `duration < 1 day` take hourly means,
if `duration < 30 days` take daily means,
if `duration < 90 days` take means over ten days,
If `duration >= 3 months` take monthly means.
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
function core_default_diagnostics(output_writer, duration, start_date, t_start, topography)
    core_diagnostics = [
        "ts",
        "ta",
        "tas",
        "uas",
        "vas",
        "thetaa",
        "ha",
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
    precip_diagnostics = ["husra", "hussn"]
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
    precip_diagnostics = ["husra", "hussn", "cdnc", "ncra"]
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
function default_diagnostics(
    ::RRTMGPI.AbstractRRTMGPMode,
    duration,
    start_date,
    t_start;
    output_writer,
)
    rad_diagnostics = [
        "rsd",
        "rsdt",
        "rsds",
        "rsu",
        "rsut",
        "rsus",
        "rld",
        "rlds",
        "rlu",
        "rlut",
        "rlus",
    ]

    average_func = frequency_averages(duration)

    return [average_func(rad_diagnostics...; output_writer, start_date, t_start)...]
end


function default_diagnostics(
    ::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
    duration,
    start_date,
    t_start;
    output_writer,
)
    rad_diagnostics = [
        "rsd",
        "rsdt",
        "rsds",
        "rsu",
        "rsut",
        "rsus",
        "rld",
        "rlds",
        "rlu",
        "rlut",
        "rlus",
    ]
    rad_clearsky_diagnostics = [
        "rsdcs",
        "rsdscs",
        "rsucs",
        "rsutcs",
        "rsuscs",
        "rldcs",
        "rldscs",
        "rlucs",
        "rlutcs",
    ]

    average_func = frequency_averages(duration)

    return [
        average_func(rad_diagnostics...; output_writer, start_date, t_start)...,
        average_func(rad_clearsky_diagnostics...; output_writer, start_date, t_start)...,
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
    ]

    average_func = frequency_averages(duration)

    return [
        average_func(edmfx_draft_diagnostics...; output_writer, start_date, t_start)...,
        average_func(edmfx_env_diagnostics...; output_writer, start_date, t_start)...,
    ]
end


function default_diagnostics(
    ::DiagnosticEDMFX,
    duration,
    start_date,
    t_start;
    output_writer,
)
    diagnostic_edmfx_draft_diagnostics = [
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
    diagnostic_edmfx_env_diagnostics = ["waen", "tke", "lmix"]

    average_func = frequency_averages(duration)

    return [
        average_func(
            diagnostic_edmfx_draft_diagnostics...;
            output_writer,
            start_date,
            t_start,
        )...,
        average_func(
            diagnostic_edmfx_env_diagnostics...;
            output_writer,
            start_date,
            t_start,
        )...,
    ]
end

function default_diagnostics(::EDOnlyEDMFX, duration, start_date, t_start; output_writer)
    edonly_edmfx_diagnostics = ["tke"]

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
