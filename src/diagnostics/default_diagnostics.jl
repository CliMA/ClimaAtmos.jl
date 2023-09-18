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
    # TODO: Probably not the most elegant way to do this...
    defaults = Any[]

    for field in fieldnames(AtmosModel)
        def_model = default_diagnostics(getfield(model, field))
        append!(defaults, def_model)
    end

    return defaults
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
include("defaults/moisture_model.jl")
