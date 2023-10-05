# This file is included by Diagnostics.jl and defines all the defaults for various models. A
# model here is either a global AtmosModel, or small (sub)models (e.g., DryModel()).
#
# If you are developing new models, add your defaults here. If you want to add more high
# level interfaces, add them here. Feel free to include extra files.

"""
    default_diagnostics(model)


Return a list of `ScheduledDiagnostic`s associated with the given `model`.

"""
function default_diagnostics(model::AtmosModel)
    # Unfortunately, [] is not treated nicely in a map (we would like it to be "excluded"),
    # so we need to manually filter out the submodels that don't have defaults associated
    # to
    non_empty_fields = filter(
        x -> default_diagnostics(getfield(model, x)) != [],
        fieldnames(AtmosModel),
    )

    # We use a map because we want to ensure that diagnostics is a well defined type, not
    # Any. This reduces latency.
    return vcat(
        core_default_diagnostics(),
        map(non_empty_fields) do field
            default_diagnostics(getfield(model, field))
        end...,
    )
end

# Base case: if we call default_diagnostics on something that we don't have information
# about, we get nothing back (to be specific, we get an empty list, so that we can assume
# that all the default_diagnostics return the same type). This is used by
# default_diagnostics(model::AtmosModel), so that we can ignore defaults for submodels
# that have no given defaults.
default_diagnostics(_) = []


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
    daily_maxs(short_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the daily max for the given variables.
"""
daily_maxs(short_names...; output_writer = HDF5Writer()) =
    common_diagnostics(24 * 60 * 60, max, output_writer, short_names...)
"""
    daily_max(short_names; output_writer = HDF5Writer())


Return a `ScheduledDiagnostics` that computes the daily max for the given variable.
"""
daily_max(short_names; output_writer = HDF5Writer()) =
    daily_maxs(short_names; output_writer)[1]

"""
    daily_mins(short_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the daily min for the given variables.
"""
daily_mins(short_names...; output_writer = HDF5Writer()) =
    common_diagnostics(24 * 60 * 60, min, output_writer, short_names...)
"""
    daily_min(short_names; output_writer = HDF5Writer())


Return a `ScheduledDiagnostics` that computes the daily min for the given variable.
"""
daily_min(short_names; output_writer = HDF5Writer()) =
    daily_mins(short_names; output_writer)[1]

"""
    daily_averages(short_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the daily average for the given variables.
"""
# An average is just a sum with a normalization before output
daily_averages(short_names...; output_writer = HDF5Writer()) =
    common_diagnostics(
        24 * 60 * 60,
        (+),
        output_writer,
        short_names...;
        pre_output_hook! = average_pre_output_hook!,
    )
"""
    daily_average(short_names; output_writer = HDF5Writer())


Return a `ScheduledDiagnostics` that compute the daily average for the given variable.
"""
# An average is just a sum with a normalization before output
daily_average(short_names; output_writer = HDF5Writer()) =
    daily_averages(short_names; output_writer)[1]
"""
    hourly_maxs(short_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the hourly max for the given variables.
"""
hourly_maxs(short_names...; output_writer = HDF5Writer()) =
    common_diagnostics(60 * 60, max, output_writer, short_names...)

"""
    hourly_max(short_names; output_writer = HDF5Writer())


Return a `ScheduledDiagnostics` that computse the hourly max for the given variable.
"""
hourly_max(short_names...; output_writer = HDF5Writer()) =
    hourly_maxs(short_names)[1]

"""
    hourly_mins(short_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the hourly min for the given variables.
"""
hourly_mins(short_names...; output_writer = HDF5Writer()) =
    common_diagnostics(60 * 60, min, output_writer, short_names...)
"""
    hourly_mins(short_names...; output_writer = HDF5Writer())


Return a `ScheduledDiagnostics` that computes the hourly min for the given variable.
"""
hourly_min(short_names; output_writer = HDF5Writer()) =
    hourly_mins(short_names; output_writer)[1]

# An average is just a sum with a normalization before output
"""
    hourly_averages(short_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the hourly average for the given variables.
"""
hourly_averages(short_names...; output_writer = HDF5Writer()) =
    common_diagnostics(
        60 * 60,
        (+),
        output_writer,
        short_names...;
        pre_output_hook! = average_pre_output_hook!,
    )

"""
    hourly_average(short_names...; output_writer = HDF5Writer())


Return a `ScheduledDiagnostics` that computes the hourly average for the given variable.
"""
hourly_average(short_names; output_writer = HDF5Writer()) =
    hourly_averages(short_names; output_writer)[1]

# Include all the subdefaults

########
# Core #
########
function core_default_diagnostics()
    core_diagnostics = ["ts", "ta", "thetaa", "pfull", "rhoa", "ua", "va", "wa"]

    return [
        # We need to compute the topography at the beginning of the simulation (and only at
        # the beginning), so we set output_every = 0 (it still called at the first timestep)
        ScheduledDiagnosticIterations(;
            variable = get_diagnostic_variable("orog"),
            output_every = 0,
        ),
        daily_averages(core_diagnostics...; output_writer)...,
        daily_max("ts"; output_writer),
        daily_min("ts"; output_writer),
    ]
end

###############
# Energy form #
###############
function default_diagnostics(::TotalEnergy)
    total_energy_diagnostics = ["hfes"]

    return [daily_averages(total_energy_diagnostics...)...]
end


##################
# Moisture model #
##################
function default_diagnostics(
    ::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    moist_diagnostics = ["hur", "hus", "hussfc", "evspsbl"]

    return [daily_averages(moist_diagnostics...)...]
end

#######################
# Precipitation model #
#######################
function default_diagnostics(::Microphysics0Moment)
    precip_diagnostics = ["pr"]

    return [daily_averages(precip_diagnostics...)...]
end

##################
# Radiation mode #
##################
function default_diagnostics(::RRTMGPI.AbstractRRTMGPMode)
    allsky_diagnostics = ["rsd", "rsu", "rld", "rlu"]

    return [daily_averages(allsky_diagnostics...)...]
end


function default_diagnostics(::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics)
    clear_diagnostics = ["rsdcs", "rsucs", "rldcs", "rlucs"]

    return [daily_averages(clear_diagnostics...)...]
end
