# Computing and saving diagnostics

## I want to add a new diagnostic variable

Diagnostic variables are represented in `ClimaAtmos` with a `DiagnosticVariable`
`struct`. Fundamentally, a `DiagnosticVariable` contains metadata about the
variable, and a function that computes it from the state.

### Metadata

The metadata we currently support is `short_name`, `long_name`, `units`,
`comments`. This metadata is relevant mainly in the context of how the variable
is output. Therefore, it is responsibility of the `output_writer` (see
`ScheduledDiagnostic`) to handle the metadata properly. The `output_writer`s
provided by `ClimaAtmos` use this metadata.

In `ClimaAtmos`, we follow the convention that:

- `short_name` is the name used to identify the variable in the output files and
                in the file names. It is short, but descriptive. We identify
                diagnostics by their short name, so the diagnostics defined by
                `ClimaAtmos` have to have unique `long_name`s. We follow the
                Coupled Model Intercomparison Project (CMIP) convetions.

- `long_name`: Name used to identify the variable in the input files.

- `units`: Physical units of the variable.

- `comments`: More verbose explanation of what the variable is, or comments related to how
              it is defined or computed.

In `ClimaAtmos`, we try to follow [this Google
spreadsheet](https://docs.google.com/spreadsheets/d/1qUauozwXkq7r1g-L4ALMIkCNINIhhCPx)
for variable naming (except for the `short_names`, which we prefer being more
descriptive).

### Compute function

The other piece of information needed to specify a `DiagnosticVariable` is a
function `compute_from_integrator`. Schematically, a `compute_from_integrator` has to look like
```julia
function compute_from_integrator(integrator, out)
    # FIXME: Remove this line when ClimaCore implements the broadcasting to enable this
    out .= # Calculcations with the state (= integrator.u) and the parameters (= integrator.p)
end
```
Diagnostics are implemented as callbacks function which pass the `integrator`
object (from `OrdinaryDiffEq`) to `compute_from_integrator`.

`compute_from_integrator` also takes a second argument, `out`, which is used to
avoid extra memory allocations (which hurt performance). If `out` is `nothing`,
and new area of memory is allocated. If `out` is a `ClimaCore.Field`, the
operation is done in-place without additional memory allocations.

### Adding to the `ALL_DIAGNOSTICS` dictionary

`ClimaAtmos` comes with a collection of pre-defined `DiagnosticVariable` in the
`ALL_DIAGNOSTICS` dictionary. `ALL_DIAGNOSTICS` maps a `long_name` with the
corresponding `DiagnosticVariable`.

If you are extending `ClimaAtmos` and want to add a new diagnostic variable to
`ALL_DIAGNOSTICS`, go ahead and look at the files we `include` in
`diagnostics/Diagnostics.jl`. You can add more diagnostics in those files or add
a new one. We provide a convenience function, `add_diagnostic_variable!` to add
new `DiagnosticVariable`s to the `ALL_DIAGNOSTICS` dictionary.
`add_diagnostic_variable!` take the same arguments as the constructor for
`DiagnosticVariable`, but also performs additional checks. So, use
`add_diagnostic_variable!` instead of editing the `ALL_DIAGNOSTICS` directly.
