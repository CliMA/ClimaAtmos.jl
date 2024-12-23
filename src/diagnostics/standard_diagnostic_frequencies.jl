"""
    monthly_maxs(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the monthly max for the given variables.
"""
monthly_maxs(FT, short_names...; output_writer, start_date) =
    common_diagnostics(Month(1), max, output_writer, start_date, short_names...)
"""
    monthly_max(FT, short_names; output_writer, start_date)

Return a `ScheduledDiagnostics` that computes the monthly max for the given variable.
"""
monthly_max(FT, short_names; output_writer, start_date) =
    monthly_maxs(FT, short_names; output_writer, start_date)[1]

"""
    monthly_mins(short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the monthly min for the given variables.
"""
monthly_mins(FT, short_names...; output_writer, start_date) =
    common_diagnostics(Month(1), min, output_writer, start_date, short_names...)
"""
    monthly_min(FT, short_names; output_writer, start_date)

Return a `ScheduledDiagnostics` that computes the monthly min for the given variable.
"""
monthly_min(FT, short_names; output_writer, start_date) =
    monthly_mins(FT, short_names; output_writer, start_date)[1]

"""
    monthly_averages(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the monthly average for the given variables.
"""
# An average is just a sum with a normalization before output
monthly_averages(FT, short_names...; output_writer, start_date) =
    common_diagnostics(
        Month(1),
        (+),
        output_writer,
        start_date,
        short_names...,
        ;
        pre_output_hook! = average_pre_output_hook!,
    )

"""
    monthly_average(FT, short_names; output_writer, start_date)

Return a `ScheduledDiagnostics` that compute the monthly average for the given variable.
"""
# An average is just a sum with a normalization before output
monthly_average(FT, short_names; output_writer, start_date) =
    monthly_averages(FT, short_names; output_writer, start_date)[1]

"""
    tendaily_maxs(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the max over ten days for the given variables.
"""
tendaily_maxs(FT, short_names...; output_writer, start_date) =
    common_diagnostics(
        Day(10),
        max,
        output_writer,
        start_date,
        short_names...,
    )
"""
    tendaily_max(FT, short_names; output_writer, start_date)

Return a `ScheduledDiagnostics` that computes the max over ten days for the given variable.
"""
tendaily_max(FT, short_names; output_writer, start_date) =
    tendaily_maxs(FT, short_names; output_writer, start_date)[1]

"""
    tendaily_mins(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the min over ten days for the given variables.
"""
tendaily_mins(FT, short_names...; output_writer, start_date) =
    common_diagnostics(
        Day(10),
        min,
        output_writer,
        start_date,
        short_names...,
    )
"""
    tendaily_min(FT, short_names; output_writer, start_date)

Return a `ScheduledDiagnostics` that computes the min over ten days for the given variable.
"""
tendaily_min(FT, short_names; output_writer, start_date) =
    tendaily_mins(FT, short_names; output_writer, start_date)[1]

"""
    tendaily_averages(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the average over ten days for the given variables.
"""
# An average is just a sum with a normalization before output
tendaily_averages(FT, short_names...; output_writer, start_date) =
    common_diagnostics(
        Day(10),
        (+),
        output_writer,
        start_date,
        short_names...;
        pre_output_hook! = average_pre_output_hook!,
    )
"""
    tendaily_average(FT, short_names; output_writer, start_date)

Return a `ScheduledDiagnostics` that compute the average over ten days for the given variable.
"""
# An average is just a sum with a normalization before output
tendaily_average(FT, short_names; output_writer, start_date) =
    tendaily_averages(FT, short_names; output_writer, start_date)[1]

"""
    daily_maxs(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the daily max for the given variables.
"""
daily_maxs(FT, short_names...; output_writer, start_date) = common_diagnostics(
    Day(1),
    max,
    output_writer,
    start_date,
    short_names...,
)
"""
    daily_max(FT, short_names; output_writer, start_date)


Return a `ScheduledDiagnostics` that computes the daily max for the given variable.
"""
daily_max(FT, short_names; output_writer, start_date) =
    daily_maxs(FT, short_names; output_writer, start_date)[1]

"""
    daily_mins(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the daily min for the given variables.
"""
daily_mins(FT, short_names...; output_writer, start_date) = common_diagnostics(
    Day(1),
    min,
    output_writer,
    start_date,
    short_names...,
)
"""
    daily_min(FT, short_names; output_writer, start_date)

Return a `ScheduledDiagnostics` that computes the daily min for the given variable.
"""
daily_min(FT, short_names; output_writer, start_date) =
    daily_mins(FT, short_names; output_writer, start_date)[1]

"""
    daily_averages(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the daily average for the given variables.
"""
# An average is just a sum with a normalization before output
daily_averages(FT, short_names...; output_writer, start_date) =
    common_diagnostics(
        Day(1),
        (+),
        output_writer,
        start_date,
        short_names...;
        pre_output_hook! = average_pre_output_hook!,
    )
"""
    daily_average(FT, short_names; output_writer, start_date)

Return a `ScheduledDiagnostics` that compute the daily average for the given variable.
"""
# An average is just a sum with a normalization before output
daily_average(FT, short_names; output_writer, start_date) =
    daily_averages(FT, short_names; output_writer, start_date)[1]

"""
    hourly_maxs(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the hourly max for the given variables.
"""
hourly_maxs(FT, short_names...; output_writer, start_date) = common_diagnostics(
    Hour(1),
    max,
    output_writer,
    start_date,
    short_names...,
)

"""
    hourly_max(FT, short_names; output_writer, start_date)

Return a `ScheduledDiagnostics` that computes the hourly max for the given variable.
"""
hourly_max(FT, short_names; output_writer, start_date) =
    hourly_maxs(FT, short_names; output_writer, start_date)[1]

"""
    hourly_mins(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the hourly min for the given variables.
"""
hourly_mins(FT, short_names...; output_writer, start_date) = common_diagnostics(
    Hour(1),
    min,
    output_writer,
    start_date,
    short_names...,
)
"""
    hourly_mins(FT, short_names...; output_writer, start_date)


Return a `ScheduledDiagnostics` that computes the hourly min for the given variable.
"""
hourly_min(FT, short_names; output_writer, start_date) =
    hourly_mins(FT, short_names; output_writer, start_date)[1]

# An average is just a sum with a normalization before output
"""
    hourly_averages(FT, short_names...; output_writer, start_date)

Return a list of `ScheduledDiagnostics` that compute the hourly average for the given variables.
"""
hourly_averages(FT, short_names...; output_writer, start_date) =
    common_diagnostics(
        Hour(1),
        (+),
        output_writer,
        start_date,
        short_names...;
        pre_output_hook! = average_pre_output_hook!,
    )

"""
    hourly_average(FT, short_names...; output_writer, start_date)

Return a `ScheduledDiagnostics` that computes the hourly average for the given variable.
"""
hourly_average(FT, short_names; output_writer, start_date) =
    hourly_averages(FT, short_names; output_writer, start_date)[1]
