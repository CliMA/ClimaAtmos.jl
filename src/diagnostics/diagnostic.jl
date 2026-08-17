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

Register a diagnostic variable in the `ALL_DIAGNOSTICS` registry.

This mutates the global `ClimaAtmos.Diagnostics.ALL_DIAGNOSTICS` dictionary, keyed on
`short_name`. Registering a `short_name` that is already present overwrites the existing
entry and emits a warning listing the fields of the entry being replaced.

Where possible, follow the naming scheme outlined in
https://airtable.com/appYNLuWqAgzLbhSq/shrKcLEdssxb8Yvcp/tblL7dJkC3vl5zQLb

# Keyword Arguments

  - `short_name`: Short, descriptive name identifying the variable in the registry, the
    output files, and the output file names. ClimaAtmos follows the Coupled Model
    Intercomparison Project conventions.
  - `long_name`: Human-readable name written to the output files.
  - `standard_name = ""`: CF standard name, as in
    http://cfconventions.org/Data/cf-standard-names/71/build/cf-standard-name-table.html
  - `units`: Physical units of the variable, as a string (e.g. `"kg m^-3"`).
  - `comments = ""`: Longer explanation of what the variable is, or of how it is defined
    and computed.
  - `compute = nothing`: Function `compute(state, cache, time)` returning the diagnostic,
    preferably as a lazy broadcast. `state` holds the prognostic variables (`Y` in the
    source), `cache` holds parameters and precomputed quantities (`p`), and `time` is the
    current simulation time (`t`), usually in seconds.
  - `compute! = nothing`: Alternative in-place form; see the note below. Exactly one of
    `compute` and `compute!` is expected.

# Returns

The newly registered `ClimaDiagnostics.DiagnosticVariable`.

# Examples

```julia
import ClimaAtmos as CA

CA.Diagnostics.add_diagnostic_variable!(
    short_name = "rhoa",
    long_name = "Air Density",
    standard_name = "air_density",
    units = "kg m^-3",
    compute = (state, cache, time) -> state.c.ρ,
)
```

!!! note "Backward compatibility with ClimaDiagnostics v0.2.13"

    For backward compatibility, a function can be passed to the keyword argument
    `compute!` instead of `compute`. It takes four arguments,
    `compute!(out, state, cache, time)`, where `out` is either `nothing` or preallocated
    memory to write the result into. In the first case the function allocates and returns
    the result; in the second it writes into `out`. In both cases it returns the result.
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

Look up a registered diagnostic variable by its short name.

Throws an error if `short_name` is not in the `ALL_DIAGNOSTICS` registry. Diagnostics are
registered with `add_diagnostic_variable!`.

# Returns

The `ClimaDiagnostics.DiagnosticVariable` registered under `short_name`.

# Examples

```julia
import ClimaAtmos as CA

variable = CA.Diagnostics.get_diagnostic_variable("ta")
```
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
include("microphysics_diagnostics.jl")
include("cosp_diagnostics.jl")
include("tendency_diagnostics.jl")

# Default diagnostics and higher level interfaces
include("default_diagnostics.jl")
include("diagnostics_config.jl")
