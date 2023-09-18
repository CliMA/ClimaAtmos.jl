# diagnostic_utils.jl
#
# This file contains:
# - get_descriptive_name: to condense ScheduledDiagnostic information into few characters.


"""
    get_descriptive_name(variable::DiagnosticVariable,
                         output_every,
                         reduction_time_func,
                         pre_output_hook!;
                         units_are_seconds = true)


Return a compact, unique-ish, identifier generated from the given information.

`output_every` is interpreted as in seconds if `units_are_seconds` is `true`. Otherwise, it
is interpreted as in units of number of iterations.

This function is useful for filenames and error messages.

"""
function get_descriptive_name(
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
