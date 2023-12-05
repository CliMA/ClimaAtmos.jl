# diagnostic_utils.jl
#
# This file contains:
# - descriptive_short_name: to condense ScheduledDiagnostic information into few characters.
# - descriptive_long_name: to produce full names that are clearly human-understandable


"""
    descriptive_short_name(variable::DiagnosticVariable,
                           output_every,
                           reduction_time_func,
                           pre_output_hook!;
                           units_are_seconds = true)


Return a compact, unique-ish, identifier generated from the given information.

`output_every` is interpreted as in seconds if `units_are_seconds` is `true`. Otherwise, it
is interpreted as in units of number of iterations.

This function is useful for filenames and error messages.

"""
function descriptive_short_name(
    variable::DiagnosticVariable,
    output_every,
    reduction_time_func,
    pre_output_hook!;
    units_are_seconds = true,
)
    var = "$(variable.short_name)"
    isa_reduction = !isnothing(reduction_time_func)


    if isa_reduction
        red = "$(reduction_time_func)"

        # Let's check if we are computing the average. Note that this might slip under the
        # radar if the user passes their own pre_output_hook!.
        if reduction_time_func == (+) &&
           pre_output_hook! == average_pre_output_hook!
            red = "average"
        end

        if units_are_seconds

            # Convert period from seconds to days, hours, minutes, seconds
            period = ""

            days, rem_seconds = divrem(output_every, 24 * 60 * 60)
            hours, rem_seconds = divrem(rem_seconds, 60 * 60)
            minutes, seconds = divrem(rem_seconds, 60)

            days > 0 && (period *= "$(days)d_")
            hours > 0 && (period *= "$(hours)h_")
            minutes > 0 && (period *= "$(minutes)m_")
            seconds > 0 && (period *= "$(seconds)s_")

            suffix = period * red
        else
            suffix = "$(output_every)it_$(red)"
        end
    else
        suffix = "inst"
    end
    return "$(var)_$(suffix)"
end

"""
    descriptive_long_name(variable::DiagnosticVariable,
                          output_every,
                          reduction_time_func,
                          pre_output_hook!;
                          units_are_seconds = true)


Return a verbose description of the given output variable.

`output_every` is interpreted as in seconds if `units_are_seconds` is `true`. Otherwise, it
is interpreted as in units of number of iterations.

This function is useful for attributes in output files.

"""
function descriptive_long_name(
    variable::DiagnosticVariable,
    output_every,
    reduction_time_func,
    pre_output_hook!;
    units_are_seconds = true,
)
    var = "$(variable.long_name)"
    isa_reduction = !isnothing(reduction_time_func)

    if isa_reduction
        red = "$(reduction_time_func)"

        # Let's check if we are computing the average. Note that this might slip under the
        # radar if the user passes their own pre_output_hook!.
        if reduction_time_func == (+) &&
           pre_output_hook! == average_pre_output_hook!
            red = "average"
        end

        if units_are_seconds

            # Convert period from seconds to days, hours, minutes, seconds
            period = ""

            days, rem_seconds = divrem(output_every, 24 * 60 * 60)
            hours, rem_seconds = divrem(rem_seconds, 60 * 60)
            minutes, seconds = divrem(rem_seconds, 60)

            days > 0 && (period *= "$(days) Day(s)")
            hours > 0 && (period *= "$(hours) Hour(s)")
            minutes > 0 && (period *= "$(minutes) Minute(s)")
            seconds > 0 && (period *= "$(seconds) Second(s)")

            period_str = period
        else
            period_str = "$(output_every) Iterations"
        end
        suffix = "$(red) within $(period_str)"
    else
        suffix = "Instantaneous"
    end
    return "$(var), $(suffix)"
end
