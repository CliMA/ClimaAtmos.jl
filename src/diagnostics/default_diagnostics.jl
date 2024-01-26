# This file is included by Diagnostics.jl and defines all the defaults for various models. A
# model here is either a global AtmosModel, or small (sub)models (e.g., DryModel()).
#
# If you are developing new models, add your defaults here. If you want to add more high
# level interfaces, add them here. Feel free to include extra files.

"""
    default_diagnostics(model, output_writer)


Return a list of `ScheduledDiagnostic`s associated with the given `model` that use
`output_write` to write to disk.

"""
function default_diagnostics(model::AtmosModel; output_writer)
    # Unfortunately, [] is not treated nicely in a map (we would like it to be "excluded"),
    # so we need to manually filter out the submodels that don't have defaults associated
    # to
    non_empty_fields = filter(
        x -> default_diagnostics(getfield(model, x); output_writer) != [],
        fieldnames(AtmosModel),
    )

    # We use a map because we want to ensure that diagnostics is a well defined type, not
    # Any. This reduces latency.
    return vcat(
        core_default_diagnostics(output_writer),
        map(non_empty_fields) do field
            default_diagnostics(getfield(model, field); output_writer)
        end...,
    )
end

# Base case: if we call default_diagnostics on something that we don't have information
# about, we get nothing back (to be specific, we get an empty list, so that we can assume
# that all the default_diagnostics return the same type). This is used by
# default_diagnostics(model::AtmosModel; output_writer), so that we can ignore defaults for
# submodels that have no given defaults.
default_diagnostics(submodel; output_writer) = []


"""
    produce_common_diagnostic_function(period, reduction)


Helper function to define functions like `daily_max`.
"""
function common_diagnostics(
    period,
    reduction,
    output_writer,
    short_names...;
    pre_output_hook! = nothing,
)
    return [
        ScheduledDiagnosticTime(
            variable = get_diagnostic_variable(short_name),
            compute_every = :timestep,
            output_every = period, # seconds
            reduction_time_func = reduction,
            output_writer = output_writer,
            pre_output_hook! = pre_output_hook!,
        ) for short_name in short_names
    ]
end

function average_pre_output_hook!(accum, counter)
    @. accum = accum / counter
    nothing
end

"""
    daily_maxs(short_names...; output_writer)


Return a list of `ScheduledDiagnostics` that compute the daily max for the given variables.
"""
daily_maxs(short_names...; output_writer) =
    common_diagnostics(24 * 60 * 60, max, output_writer, short_names...)
"""
    daily_max(short_names; output_writer)


Return a `ScheduledDiagnostics` that computes the daily max for the given variable.
"""
daily_max(short_names; output_writer) =
    daily_maxs(short_names; output_writer)[1]

"""
    daily_mins(short_names...; output_writer)


Return a list of `ScheduledDiagnostics` that compute the daily min for the given variables.
"""
daily_mins(short_names...; output_writer) =
    common_diagnostics(24 * 60 * 60, min, output_writer, short_names...)
"""
    daily_min(short_names; output_writer)


Return a `ScheduledDiagnostics` that computes the daily min for the given variable.
"""
daily_min(short_names; output_writer) =
    daily_mins(short_names; output_writer)[1]

"""
    daily_averages(short_names...; output_writer)


Return a list of `ScheduledDiagnostics` that compute the daily average for the given variables.
"""
# An average is just a sum with a normalization before output
daily_averages(short_names...; output_writer) = common_diagnostics(
    24 * 60 * 60,
    (+),
    output_writer,
    short_names...;
    pre_output_hook! = average_pre_output_hook!,
)
"""
    daily_average(short_names; output_writer)


Return a `ScheduledDiagnostics` that compute the daily average for the given variable.
"""
# An average is just a sum with a normalization before output
daily_average(short_names; output_writer) =
    daily_averages(short_names; output_writer)[1]
"""
    hourly_maxs(short_names...; output_writer)


Return a list of `ScheduledDiagnostics` that compute the hourly max for the given variables.
"""
hourly_maxs(short_names...; output_writer) =
    common_diagnostics(60 * 60, max, output_writer, short_names...)

"""
    hourly_max(short_names; output_writer)


Return a `ScheduledDiagnostics` that computes the hourly max for the given variable.
"""
hourly_max(short_names...; output_writer) =
    hourly_maxs(short_names; output_writer)[1]

"""
    hourly_mins(short_names...; output_writer)


Return a list of `ScheduledDiagnostics` that compute the hourly min for the given variables.
"""
hourly_mins(short_names...; output_writer) =
    common_diagnostics(60 * 60, min, output_writer, short_names...)
"""
    hourly_mins(short_names...; output_writer)


Return a `ScheduledDiagnostics` that computes the hourly min for the given variable.
"""
hourly_min(short_names; output_writer) =
    hourly_mins(short_names; output_writer)[1]

# An average is just a sum with a normalization before output
"""
    hourly_averages(short_names...; output_writer)


Return a list of `ScheduledDiagnostics` that compute the hourly average for the given variables.
"""
hourly_averages(short_names...; output_writer) = common_diagnostics(
    60 * 60,
    (+),
    output_writer,
    short_names...;
    pre_output_hook! = average_pre_output_hook!,
)

"""
    hourly_average(short_names...; output_writer)


Return a `ScheduledDiagnostics` that computes the hourly average for the given variable.
"""
hourly_average(short_names; output_writer) =
    hourly_averages(short_names; output_writer)[1]

# Include all the subdefaults

########
# Core #
########
function core_default_diagnostics(output_writer)
    core_diagnostics =
        ["ts", "ta", "thetaa", "ha", "pfull", "rhoa", "ua", "va", "wa", "hfes"]
    return [
        # We need to compute the topography at the beginning of the simulation (and only at
        # the beginning), so we set output_every = 0 (it still called at the first timestep)
        ScheduledDiagnosticIterations(;
            variable = get_diagnostic_variable("orog"),
            output_every = 0,
            output_writer,
        ),
        daily_averages(core_diagnostics...; output_writer)...,
        daily_max("ts"; output_writer),
        daily_min("ts"; output_writer),
    ]
end

##################
# Moisture model #
##################
function default_diagnostics(
    ::T;
    output_writer,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    moist_diagnostics = ["hur", "hus", "cl", "clw", "cli", "hussfc", "evspsbl"]

    return [daily_averages(moist_diagnostics...; output_writer)...]
end

#######################
# Precipitation model #
#######################
function default_diagnostics(::Microphysics0Moment; output_writer)
    precip_diagnostics = ["pr"]

    return [daily_averages(precip_diagnostics...; output_writer)...]
end

##################
# Radiation mode #
##################
function default_diagnostics(::RRTMGPI.AbstractRRTMGPMode; output_writer)
    rad_diagnostics = ["rsd", "rsu", "rld", "rlu"]

    return [daily_averages(rad_diagnostics...; output_writer)...]
end


function default_diagnostics(
    ::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics;
    output_writer,
)
    rad_diagnostics =
        ["rsd", "rsu", "rld", "rlu", "rsdcs", "rsucs", "rldcs", "rlucs"]

    return [daily_averages(rad_diagnostics...; output_writer)...]
end

##################
# Turbconv model #
##################
function default_diagnostics(::PrognosticEDMFX; output_writer)
    edmfx_tenmin_diagnostics = [
        "ts",
        "ta",
        "thetaa",
        "ha",
        "pfull",
        "rhoa",
        "ua",
        "va",
        "wa",
        "hur",
        "hus",
        "cl",
        "clw",
        "cli",
        "hussfc",
        "evspsbl",
        "arup",
        "waup",
        "taup",
        "thetaaup",
        "haup",
        "husup",
        "hurup",
        "clwup",
        "cliup",
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

    thirtymin_insts(short_names...; output_writer) =
        common_diagnostics(30 * 60, nothing, output_writer, short_names...;)

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

    return [
        thirtymin_insts(edmfx_tenmin_diagnostics...; output_writer)...,
        daily_averages(edmfx_draft_diagnostics...; output_writer)...,
        daily_averages(edmfx_env_diagnostics...; output_writer)...,
    ]
end


function default_diagnostics(::DiagnosticEDMFX; output_writer)
    diagnostic_edmfx_tenmin_diagnostics = [
        "ts",
        "ta",
        "thetaa",
        "ha",
        "pfull",
        "rhoa",
        "ua",
        "va",
        "wa",
        "hur",
        "hus",
        "cl",
        "clw",
        "cli",
        "hussfc",
        "evspsbl",
        "arup",
        "waup",
        "taup",
        "thetaaup",
        "haup",
        "husup",
        "hurup",
        "clwup",
        "cliup",
        "waen",
        "tke",
        "lmix",
    ]

    thirtymin_insts(short_names...; output_writer) =
        common_diagnostics(30 * 60, nothing, output_writer, short_names...;)


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

    return [
        thirtymin_insts(
            diagnostic_edmfx_tenmin_diagnostics...;
            output_writer,
        )...,
        daily_averages(diagnostic_edmfx_draft_diagnostics...; output_writer)...,
        daily_averages(diagnostic_edmfx_env_diagnostics...; output_writer)...,
    ]
end
