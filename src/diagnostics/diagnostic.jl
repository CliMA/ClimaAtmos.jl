# ClimaAtmos diagnostics

# - A dictionary `ALL_DIAGNOSTICS` with all the diagnostics we know how to compute, keyed
#   over their short name. If you want to add more diagnostics, look at the included files.
#   You can add your own file if you want to define several new diagnostics that are
#   conceptually related. The dictionary `ALL_DIAGNOSTICS` should be considered an
#   implementation detail.

import ClimaUtilities
import Dates

const ALL_DIAGNOSTICS = Dict{String, DiagnosticVariable}()

"""
    add_diagnostic_variable!(;
        short_name,
        long_name,
        standard_name = "",
        units,
        comments = "",
        compute = nothing,
        compute! = nothing,
    )

Add a new variable to the `ALL_DIAGNOSTICS` dictionary (this function mutates the state of
`ClimaAtmos.ALL_DIAGNOSTICS`).

If possible, please follow the naming scheme outline in
https://airtable.com/appYNLuWqAgzLbhSq/shrKcLEdssxb8Yvcp/tblL7dJkC3vl5zQLb

Keyword arguments
=================

- `short_name`: Name used to identify the variable in the output files and in the file names.
                Short but descriptive.
                `ClimaAtmos` diagnostics are identified by the short name.
                We follow the Coupled Model Intercomparison Project conventions.

- `long_name`: Name used to identify the variable in the output files.

- `standard_name`: Standard name, as in
                   http://cfconventions.org/Data/cf-standard-names/71/build/cf-standard-name-table.html

- `units`: Physical units of the variable.

- `comments`: More verbose explanation of what the variable is,
              or comments related to how it is defined or computed.

- `compute`: Function that compute the diagnostic variable from the state.
             It has to take three arguments, `compute(state, cache, time)`:
             - The `state` contains the prognostic variables (`Y` in the source code),
             - The `cache` contains parameters and precomputed quantities (`p` in the source code),
             - The current `time`, usually in seconds, (`t` in the source code).

!!! Note "Backward compatibility with ClimaDiagnostics v0.2.13"
    For backward compatibility, a function can be passed to the
    keyword argument `compute!` instead of `compute`. The function has to take
    four arguments, `compute!(out, state, cache, time)`, where `out` is either
    `nothing` or an array of memory where to write the result of the computation.
    In the first case, the function should allocate memory and return the result.
    In the second case, the function should write the result in the provided array.
    In both cases, the function should return the result.
              

"""
function add_diagnostic_variable!(;
    short_name, long_name, standard_name = "", units, comments = "",
    compute = nothing, compute! = nothing,
)
    # Warn if the diagnostic already exists
    haskey(ALL_DIAGNOSTICS, short_name) && begin
        # Get non-function fields (e.g. `short_name`, `long_name`, `standard_name`, `units`, `comments`)
        var_fields = filter(∉((:compute!, :compute)), fieldnames(DiagnosticVariable))
        diag_as_str = mapreduce(*, var_fields) do field
            "  - $(field): $(getfield(ALL_DIAGNOSTICS[short_name], field))\n"
        end
        @warn("overwriting diagnostic `$short_name` entry containing fields\n$diag_as_str")
    end

    ALL_DIAGNOSTICS[short_name] = DiagnosticVariable(;
        short_name, long_name, standard_name, units, comments, compute, compute!,
    )
end

"""
    get_diagnostic_variable(short_name)

Return a `DiagnosticVariable` from its `short_name`, if it exists.
"""
function get_diagnostic_variable(short_name)
    haskey(ALL_DIAGNOSTICS, short_name) || error("diagnostic $short_name does not exist")
    return ALL_DIAGNOSTICS[short_name]
end

# Do you want to define more diagnostics? Add them here
include("core_diagnostics.jl")
include("radiation_diagnostics.jl")
include("edmfx_diagnostics.jl")
include("tracer_diagnostics.jl")
include("gravitywave_diagnostics.jl")
include("conservation_diagnostics.jl")
include("negative_scalars_diagnostics.jl")

# Default diagnostics and higher level interfaces
include("default_diagnostics.jl")
