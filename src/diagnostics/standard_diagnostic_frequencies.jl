"""
    monthly_maxs(short_names...; output_writer)

Return a list of `ScheduledDiagnostics` that compute the monthly max for the given variables.

A month is defined as 30 days.
"""
monthly_maxs(short_names...; output_writer) =
    common_diagnostics(30 * 24 * 60 * 60, max, output_writer, short_names...)
"""
    monthly_max(short_names; output_writer)

Return a `ScheduledDiagnostics` that computes the monthly max for the given variable.

A month is defined as 30 days.
"""
monthly_max(short_names; output_writer) =
    monthly_maxs(short_names; output_writer)[1]

"""
    monthly_mins(short_names...; output_writer)

Return a list of `ScheduledDiagnostics` that compute the monthly min for the given variables.
"""
monthly_mins(short_names...; output_writer) =
    common_diagnostics(30 * 24 * 60 * 60, min, output_writer, short_names...)
"""
    monthly_min(short_names; output_writer)

Return a `ScheduledDiagnostics` that computes the monthly min for the given variable.

A month is defined as 30 days.
"""
monthly_min(short_names; output_writer) =
    monthly_mins(short_names; output_writer)[1]

"""
    monthly_averages(short_names...; output_writer)

Return a list of `ScheduledDiagnostics` that compute the monthly average for the given variables.

A month is defined as 30 days.
"""
# An average is just a sum with a normalization before output
monthly_averages(short_names...; output_writer) = common_diagnostics(
    30 * 24 * 60 * 60,
    (+),
    output_writer,
    short_names...;
    pre_output_hook! = average_pre_output_hook!,
)
"""
    monthly_average(short_names; output_writer)

Return a `ScheduledDiagnostics` that compute the monthly average for the given variable.

A month is defined as 30 days.
"""
# An average is just a sum with a normalization before output
monthly_average(short_names; output_writer) =
    monthly_averages(short_names; output_writer)[1]

"""
    tendaily_maxs(short_names...; output_writer)

Return a list of `ScheduledDiagnostics` that compute the max over ten days for the given variables.
"""
tendaily_maxs(short_names...; output_writer) =
    common_diagnostics(10 * 24 * 60 * 60, max, output_writer, short_names...)
"""
    tendaily_max(short_names; output_writer)

Return a `ScheduledDiagnostics` that computes the max over ten days for the given variable.
"""
tendaily_max(short_names; output_writer) =
    tendaily_maxs(short_names; output_writer)[1]

"""
    tendaily_mins(short_names...; output_writer)

Return a list of `ScheduledDiagnostics` that compute the min over ten days for the given variables.
"""
tendaily_mins(short_names...; output_writer) =
    common_diagnostics(10 * 24 * 60 * 60, min, output_writer, short_names...)
"""
    tendaily_min(short_names; output_writer)

Return a `ScheduledDiagnostics` that computes the min over ten days for the given variable.
"""
tendaily_min(short_names; output_writer) =
    tendaily_mins(short_names; output_writer)[1]

"""
    tendaily_averages(short_names...; output_writer)

Return a list of `ScheduledDiagnostics` that compute the average over ten days for the given variables.
"""
# An average is just a sum with a normalization before output
tendaily_averages(short_names...; output_writer) = common_diagnostics(
    10 * 24 * 60 * 60,
    (+),
    output_writer,
    short_names...;
    pre_output_hook! = average_pre_output_hook!,
)
"""
    tendaily_average(short_names; output_writer)

Return a `ScheduledDiagnostics` that compute the average over ten days for the given variable.
"""
# An average is just a sum with a normalization before output
tendaily_average(short_names; output_writer) =
    tendaily_averages(short_names; output_writer)[1]

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
hourly_max(short_names; output_writer) =
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

"""
    thirtymin_insts(short_names...; output_writer)

Return a `ScheduledDiagnostics` that outputs the instantaneous value for the given variables
every 30 minutes.
"""
function thirtymin_insts(short_names...; output_writer)
    period = 30 * 60
    return [
        ScheduledDiagnosticTime(
            variable = get_diagnostic_variable(short_name),
            compute_every = period,
            output_every = period, # seconds
            output_writer = output_writer,
        ) for short_name in short_names
    ]
end

"""
    hourly_average(short_names...; output_writer)

Return a `ScheduledDiagnostics` that outputs the instantaneous value for the given variable
every 30 minutes.
"""
thirtymin_inst(short_names; output_writer) =
    thirtymin_insts(short_names; output_writer)[1]
