# ClimaAtmos diagnostics

# - A dictionary `ALL_DIAGNOSTICS` with all the diagnostics we know how to compute, keyed
#   over their short name. If you want to add more diagnostics, look at the included files.
#   You can add your own file if you want to define several new diagnostics that are
#   conceptually related. The dictionary `ALL_DIAGNOSTICS` should be considered an
#   implementation detail.


const ALL_DIAGNOSTICS = Dict{String, DiagnosticVariable}()

"""

    add_diagnostic_variable!(; short_name,
                               long_name,
                               standard_name,
                               units,
                               description,
                               compute!,
                               compute,)


Add a new variable to the `ALL_DIAGNOSTICS` dictionary (this function mutates the state of
`ClimaAtmos.ALL_DIAGNOSTICS`).

If possible, please follow the naming scheme outline in
https://airtable.com/appYNLuWqAgzLbhSq/shrKcLEdssxb8Yvcp/tblL7dJkC3vl5zQLb

Keyword arguments
=================


- `short_name`: Name used to identify the variable in the output files and in the file
                names. Short but descriptive. `ClimaAtmos` diagnostics are identified by the
                short name. We follow the Coupled Model Intercomparison Project conventions.

- `long_name`: Name used to identify the variable in the output files.

- `standard_name`: Standard name, as in
                   http://cfconventions.org/Data/cf-standard-names/71/build/cf-standard-name-table.html

- `units`: Physical units of the variable.

- `comments`: More verbose explanation of what the variable is, or comments related to how
              it is defined or computed.

- `compute`: Function that computes the diagnostic variable from the state, cache, and time.
             It should return a `Field` or a `Base.Broadcast.Broadcasted` expression. For
             best performance, it should not allocate. If the function contains at `@.`,
             that is a good indication that it should also contain a `@lazy`.
"""
function add_diagnostic_variable!(;
    short_name,
    long_name,
    standard_name = "",
    units,
    comments = "",
    compute! = nothing,
    compute = nothing,
)
    haskey(ALL_DIAGNOSTICS, short_name) && @warn(
        "overwriting diagnostic `$short_name` entry containing fields\n" *
        "$(map(
            field -> "$(getfield(ALL_DIAGNOSTICS[short_name], field))",
            filter(field -> !(field in (:compute!, :compute)), fieldnames(DiagnosticVariable)),
        ))"
    )

    ALL_DIAGNOSTICS[short_name] = DiagnosticVariable(;
        short_name,
        long_name,
        standard_name,
        units,
        comments,
        compute!,
        compute,
    )
end

"""

    get_diagnostic_variable!(short_name)


Return a `DiagnosticVariable` from its `short_name`, if it exists.
"""
function get_diagnostic_variable(short_name)
    haskey(ALL_DIAGNOSTICS, short_name) ||
        error("diagnostic $short_name does not exist")

    return ALL_DIAGNOSTICS[short_name]
end

# Do you want to define more diagnostics? Add them here
include("core_diagnostics.jl")
include("radiation_diagnostics.jl")
include("edmfx_diagnostics.jl")
include("tracer_diagnostics.jl")
include("conservation_diagnostics.jl")

# Default diagnostics and higher level interfaces
include("default_diagnostics.jl")
