# diagnostic_utils.jl
#
# This file contains:
# - get_descriptive_name: to condense ScheduledDiagnostic information into few characters.


"""
    get_descriptive_name(variable::DiagnosticVariable,
                         output_every,
                         reduction_time_func;
                         units_are_seconds = true)


Return a compact, unique-ish, identifier generated from the given information.

`output_every` is interpreted as in seconds if `units_are_seconds` is `true`. Otherwise, it
is interpreted as in units of number of iterations.

This function is useful for filenames and error messages.

 """

function get_descriptive_name(
    variable::DiagnosticVariable,
    output_every,
    reduction_time_func;
    units_are_seconds = true,
)
    var = "$(variable.short_name)"
    isa_reduction = !isnothing(reduction_time_func)

    if units_are_seconds
        if isa_reduction
            red = "$(reduction_time_func)"

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
            # Not a reduction
            suffix = "inst"
        end
    else
        if isa_reduction
            suffix = "$(output_every)it_(reduction_time_func)"
        else
            suffix = "inst"
        end
    end
    return "$(var)_$(suffix)"
end
