# Diagnostics.jl
#
# This file contains:
#
# - The definition of what a DiagnosticVariable is. Morally, a DiagnosticVariable is a
#   variable we know how to compute from the state. We attach more information to it for
#   documentation and to reference to it with its long name. DiagnosticVariables can exist
#   irrespective of the existence of an actual simulation that is being run. ClimaAtmos
#   comes with several diagnostics already defined (in the `ALL_DIAGNOSTICS` dictionary).
#
# - A dictionary `ALL_DIAGNOSTICS` with all the diagnostics we know how to compute, keyed
#   over their long name. If you want to add more diagnostics, look at the included files.
#   You can add your own file if you want to define several new diagnostics that are
#   conceptually related.

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

# ClimaAtmos diagnostics

const ALL_DIAGNOSTICS = Dict{String, DiagnosticVariable}()

"""

    add_diagnostic_variable!(; short_name,
                               long_name,
                               units,
                               description,
                               compute_from_integrator)


Add a new variable to the `ALL_DIAGNOSTICS` dictionary (this function mutates the state of
`ClimaAtmos.ALL_DIAGNOSTICS`).

If possible, please follow the naming scheme outline in
https://docs.google.com/spreadsheets/d/1qUauozwXkq7r1g-L4ALMIkCNINIhhCPx

Keyword arguments
=================


- `short_name`: Name used to identify the variable in the output files and in the file
                names. Short but descriptive. `ClimaAtmos` diagnostics are identified by the
                short name. We follow the Coupled Model Intercomparison Project conventions.

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
function add_diagnostic_variable!(;
    short_name,
    long_name,
    units,
    comments,
    compute_from_integrator
)
    haskey(ALL_DIAGNOSTICS, short_name) &&
        error("diagnostic $short_name already defined")

    ALL_DIAGNOSTICS[short_name] = DiagnosticVariable(;
        short_name,
        long_name,
        units,
        comments,
        compute_from_integrator
    )
end

# Do you want to define more diagnostics? Add them here
include("core_diagnostics.jl")
