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
    get_daily_max(long_names...; output_writer = HDF5Writer())


Return a list of `ScheduledDiagnostics` that compute the daily max for the given variables.
"""
function get_daily_max(long_names...; output_writer = HDF5Writer())
    # TODO: Add mechanism to print out reasonable error on variables that are not in ALL_DIAGNOSTICS
    return [
        ScheduledDiagnosticTime(
            variable = ALL_DIAGNOSTICS[long_name],
            compute_every = :timestep,
            output_every = 86400, # seconds
            reduction_time_func = max,
            output_writer = output_writer,
        ) for long_name in long_names
    ]
end

# Include all the subdefaults
include("defaults/moisture_model.jl")
