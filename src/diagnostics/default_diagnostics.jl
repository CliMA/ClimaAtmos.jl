# This file is included by Diagnostics.jl and defines all the defaults for various models. A
# model here is either a global AtmosModel, or small (sub)models (e.g., DryModel()).
#
# If you are developing new models, add your defaults here. If you want to add more high
# level interfaces, add them here. Feel free to include extra files.

"""
    get_default_diagnostics(model)


Return a list of `ScheduledDiagnostic`s associated with the given `model`.

"""
function get_default_diagnostics(model::AtmosModel)
    # TODO: Probably not the most elegant way to do this...
    defaults = Any[]

    for field in fieldnames(AtmosModel)
        def_model = get_default_diagnostics(getfield(model, field))
        append!(defaults, def_model)
    end

    return defaults
end

# Base case: if we call get_default_diagnostics on something that we don't have information
# about, we get nothing back (to be specific, we get an empty list, so that we can assume
# that all the get_default_diagnostics return the same type). This is used by
# get_default_diagnostics(model::AtmosModel), so that we can ignore defaults for submodels
# that have no given defaults.
get_default_diagnostics(_) = []


"""
    produce_common_diagnostic_function(period, reduction)


Helper function to define functions like `get_daily_max`.
"""
function produce_common_diagnostic_function(
    period,
    reduction;
    pre_output_hook! = (accum, count) -> nothing,
)
    return (long_names...; output_writer = HDF5Writer()) -> begin
        [
            ScheduledDiagnosticTime(
                variable = ALL_DIAGNOSTICS[long_name],
                compute_every = :timestep,
                output_every = period, # seconds
                reduction_time_func = reduction,
                output_writer = output_writer,
                pre_output_hook! = pre_output_hook!,
            ) for long_name in long_names
        ]
    end
end

function average_pre_output_hook!(accum, counter)
    @. accum = accum / counter
    nothing
end

"""
    get_daily_max(long_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the daily max for the given variables.
"""
get_daily_max = produce_common_diagnostic_function(24 * 60 * 60, max)
"""
    get_daily_min(long_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the daily min for the given variables.
"""
get_daily_min = produce_common_diagnostic_function(24 * 60 * 60, min)
"""
    get_daily_average(long_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the daily average for the given variables.
"""
# An average is just a sum with a normalization before output
get_daily_average = produce_common_diagnostic_function(
    24 * 60 * 60,
    (+);
    pre_output_hook! = average_pre_output_hook!,
)

"""
    get_hourly_max(long_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the hourly max for the given variables.
"""
get_hourly_max = produce_common_diagnostic_function(60 * 60, max)

"""
    get_hourly_min(long_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the hourly min for the given variables.
"""
get_hourly_min = produce_common_diagnostic_function(60 * 60, min)

"""
    get_daily_average(long_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the hourly average for the given variables.
"""

# An average is just a sum with a normalization before output
get_hourly_average = produce_common_diagnostic_function(
    60 * 60,
    (+);
    pre_output_hook! = average_pre_output_hook!,
)

# Include all the subdefaults
include("defaults/moisture_model.jl")
