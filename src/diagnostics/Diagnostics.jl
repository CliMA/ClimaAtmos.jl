# Diagnostics.jl
#
# This file contains:
#
# - The definition of what a DiagnosticVariable is. Morally, a DiagnosticVariable is a
#   variable we know how to compute from the state. We attach more information to it for
#   documentation and to reference to it with its long name. DiagnosticVariables can exist
#   irrespective of the existence of an actual simulation that is being run.

"""
    DiagnosticVariable


A recipe to compute a diagnostic variable from the state, along with some useful metadata.

The primary use for `DiagnosticVariable`s is to be embedded in a `ScheduledDiagnostic` to
compute diagnostics while the simulation is running.

The metadata is used exclusively by the `output_writer` in the `ScheduledDiagnostic`. It is
responsibility of the `output_writer` to follow the conventions about the meaning of the
metadata and their use.

In `ClimaAtmos`, we roughly follow the naming conventions listed in this file:
https://docs.google.com/spreadsheets/d/1qUauozwXkq7r1g-L4ALMIkCNINIhhCPx

Keyword arguments
=================

- `short_name`: Name used to identify the variable in the output files and in the file
                names. Short but descriptive. `ClimaAtmos` follows the CMIP conventions and
                the diagnostics are identified by the short name.

- `long_name`: Name used to identify the variable in the input files.

- `units`: Physical units of the variable.

- `comments`: More verbose explanation of what the variable is, or comments related to how
              it is defined or computed.

- `compute_from_integrator`: Function that compute the diagnostic variable from the state.
                             It has to take two arguments: the `integrator`, and a
                             pre-allocated area of memory where to write the result of the
                             computation. It the no pre-allocated area is available, a new
                             one will be allocated. To avoid extra allocations, this
                             function should perform the calculation in-place (i.e., using
                             `.=`).
"""
Base.@kwdef struct DiagnosticVariable{T <: AbstractString, T2}
    short_name::T
    long_name::T
    units::T
    comments::T
    compute_from_integrator::T2
end
