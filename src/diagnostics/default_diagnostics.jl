# This file is included by Diagnostics.jl and defines all the defaults for various models. A
# model here is either a global AtmosModel, or small (sub)models (e.g., DryModel()).
#
# If you are developing new models, add your defaults here. If you want to add more high
# level interfaces, add them here. Feel free to include extra files.

"""
    default_diagnostics(model, duration, start_date; output_writer)

Return a list of `ScheduledDiagnostic`s associated with the given `model` that use
`output_write` to write to disk. `duration` is the expected duration of the simulation and
it is used to choose the most reasonable output frequency.

`start_date` is the date that we assign at the start of the simulation.
We convert time to date as

```julia
current_date = start_date + integrator.t
```

The logic is as follows:

If `duration < 1 day` take hourly means,
if `duration < 30 days` take daily means,
if `duration < 90 days` take means over ten days,
If `duration >= 90 year` take monthly means.
"""
function default_diagnostics(
    model::AtmosModel,
    duration,
    start_date::DateTime;
    output_writer,
)
    # Unfortunately, [] is not treated nicely in a map (we would like it to be "excluded"),
    # so we need to manually filter out the submodels that don't have defaults associated
    # to
    non_empty_fields = filter(
        x ->
            default_diagnostics(
                getfield(model, x),
                duration,
                start_date;
                output_writer,
            ) != [],
        fieldnames(AtmosModel),
    )

    # We use a map because we want to ensure that diagnostics is a well defined type, not
    # Any. This reduces latency.
    return vcat(
        core_default_diagnostics(output_writer, duration, start_date),
        map(non_empty_fields) do field
            default_diagnostics(
                getfield(model, field),
                duration,
                start_date;
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
default_diagnostics(submodel, duration, start_date; output_writer) = []

"""
    produce_common_diagnostic_function(period, reduction)

Helper function to define functions like `daily_max`.
"""
function common_diagnostics(
    period,
    reduction,
    output_writer,
    start_date,
    short_names...;
    pre_output_hook! = nothing,
)
    return vcat(
        map(short_names) do short_name
            output_schedule_func =
                period isa Period ?
                EveryCalendarDtSchedule(period; reference_date = start_date) :
                EveryDtSchedule(period)
            return ScheduledDiagnostic(
                variable = get_diagnostic_variable(short_name),
                compute_schedule_func = EveryStepSchedule(),
                output_schedule_func = output_schedule_func,
                reduction_time_func = reduction,
                output_writer = output_writer,
                pre_output_hook! = pre_output_hook!,
            )
        end...,
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
If `duration >= 90 year` take monthly means.
"""
function frequency_averages(duration)
    FT = eltype(duration)
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
function core_default_diagnostics(output_writer, duration, start_date)
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
    ]

    average_func = frequency_averages(duration)
    FT = eltype(duration)

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

    return [
        average_func(core_diagnostics...; output_writer, start_date)...,
        min_func("ts"; output_writer, start_date),
        max_func("ts"; output_writer, start_date),
    ]
end

##################
# Moisture model #
##################
function default_diagnostics(
    ::T,
    duration,
    start_date;
    output_writer,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    moist_diagnostics = [
        "hur",
        "hus",
        "cl",
        "clw",
        "cli",
        "hussfc",
        "evspsbl",
        "pr",
        "prra",
        "prsn",
        "prw",
        "lwp",
        "clwvi",
        "clivi",
    ]
    average_func = frequency_averages(duration)
    return [average_func(moist_diagnostics...; output_writer, start_date)...]
end

#######################
# Precipitation model #
#######################
function default_diagnostics(
    ::Microphysics1Moment,
    duration,
    start_date;
    output_writer,
)
    precip_diagnostics = ["husra", "hussn"]

    average_func = frequency_averages(duration)

    return [average_func(precip_diagnostics...; output_writer, start_date)...]
end

##################
# Radiation mode #
##################
function default_diagnostics(
    ::RRTMGPI.AbstractRRTMGPMode,
    duration,
    start_date;
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

    return [average_func(rad_diagnostics...; output_writer, start_date)...]
end


function default_diagnostics(
    ::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
    duration,
    start_date;
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
        average_func(rad_diagnostics...; output_writer, start_date)...,
        average_func(rad_clearsky_diagnostics...; output_writer, start_date)...,
    ]
end

##################
# Turbconv model #
##################
function default_diagnostics(
    ::PrognosticEDMFX,
    duration,
    start_date;
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
        average_func(edmfx_draft_diagnostics...; output_writer, start_date)...,
        average_func(edmfx_env_diagnostics...; output_writer, start_date)...,
    ]
end


function default_diagnostics(
    ::DiagnosticEDMFX,
    duration,
    start_date;
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
        )...,
        average_func(
            diagnostic_edmfx_env_diagnostics...;
            output_writer,
            start_date,
        )...,
    ]
end
