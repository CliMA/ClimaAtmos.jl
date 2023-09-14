# FIXME: Gabriele added this as an example. Put something meaningful here!
function get_default_diagnostics(::DryModel)
    return [
        daily_averages("air_density")...,
        ScheduledDiagnosticTime(
            variable = get_diagnostic_variable("air_density"),
            compute_every = :timestep,
            output_every = 86400, # seconds
            reduction_time_func = min,
            output_writer = HDF5Writer(),
        ),
        ScheduledDiagnosticIterations(
            variable = get_diagnostic_variable("air_density"),
            compute_every = 1,
            output_every = 1, # iteration
            output_writer = HDF5Writer(),
        ),
    ]
end
