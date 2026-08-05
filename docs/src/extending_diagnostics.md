# Adding a Diagnostic Variable

How to define a new diagnostic variable in ClimaAtmos. For computing and
saving the diagnostics that already exist, see
[Computing and saving diagnostics](diagnostics.md).

Diagnostic variables are represented in `ClimaAtmos` with a `DiagnosticVariable`
`struct`. A `DiagnosticVariable` contains metadata about the variable and a
function that computes it from the state.

## Metadata

The metadata we currently support is `short_name`, `long_name`,
`standard_name`, `units`, and `comments`. This metadata is relevant mainly to how the variable is output.
Therefore, it is the responsibility of the `output_writer` (see
`ScheduledDiagnostic`) to handle the metadata properly. The `output_writer`s
provided by `ClimaAtmos` use this metadata.

In `ClimaAtmos`, we follow the convention that:

  - `short_name` is the name used to identify the variable in the output files and
    in the file names. It is short, but descriptive. We identify
    diagnostics by their short name, so the diagnostics defined by
    `ClimaAtmos` must have unique `short_name`s.

  - `long_name`: Name used to describe the variable in the output file as an attribute.

  - `standard_name`: Standard name, as in
    [CF
    conventions](http://cfconventions.org/Data/cf-standard-names/71/build/cf-standard-name-table.html).

  - `units`: Physical units of the variable.

  - `comments`: More verbose explanation of what the variable is, or comments related to how
    it is defined or computed.

In `ClimaAtmos`, we follow the [CMIP6 MIP table](https://airtable.com/appYNLuWqAgzLbhSq/shrKcLEdssxb8Yvcp/tblL7dJkC3vl5zQLb)
for short names and long names where available. Standard names in the table are not used.

## Compute function

The other piece of information needed to specify a `DiagnosticVariable` is a
function `compute`. Schematically, a `compute` has to look like

```julia
function compute(state, cache, time)
    return _ # Calculations with the state and the cache
end
```

The function takes the `state`, `cache`, and `time` from the integrator and returns
the value of the diagnostic variable.

### In-place computation

You can alternatively provide a `compute!` function.
`compute!` takes a fourth argument, `out`, which is used to avoid extra memory allocations.

```julia
function compute!(out, state, cache, time)
    if isnothing(out)
        return _ # Calculations with the state and the cache
    else
        out .= _ # Calculations with the state and the cache
    end
end
```

The first time `compute!` is called, `out` is `nothing`, and the function has to
allocate memory and return its output. On all subsequent calls, `out` will be
the pre-allocated area of memory, so the function has to write the new value in
place.

If your diagnostic depends on the details of the model, we recommend using
additional functions so that the correct one can be found through dispatch.
The following example demonstrates this using the `compute` interface.
For instance, if you want to compute relative humidity, which does not make
sense for dry simulations, you should define the functions

```julia
function compute_relative_humidity(
    state, cache, time, microphysics_model::T,
) where {T}
    error("Cannot compute relative_humidity with microphysics_model = $T")
end

function compute_relative_humidity(
    state, cache, time, microphysics_model::MoistMicrophysics,
)
    tps = CAP.thermodynamics_params(cache.params)
    (; ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = cache.precomputed
    return @. lazy(
        TD.relative_humidity(tps, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice),
    )
end

compute_relative_humidity(state, cache, time) =
    compute_relative_humidity(state, cache, time, cache.atmos.microphysics_model)
```

This will return the correct relative humidity and throw informative errors when
it cannot be computed. We could specialize
`compute_relative_humidity` further if the relative humidity
were computed differently for `EquilibriumMicrophysics0M` and `NonEquilibriumMicrophysics`.

In `ClimaAtmos`, we define some helper functions to produce error messages, so
the error fallback can be written as

```julia
compute_relative_humidity(state, cache, time, model) =
    error_diagnostic_variable("relative_humidity", model)
```

## The `ClimaAtmos` `DiagnosticVariable`s

`ClimaAtmos` comes with a collection of pre-defined `DiagnosticVariable`s,
indexed by their `short_name`s. If you are extending `ClimaAtmos` and want to
add a new diagnostic variable, look at the files we `include` in
`diagnostics/Diagnostics.jl`. You can add more diagnostics in those files or add
a new one. We provide the convenience function `add_diagnostic_variable!` to add
new `DiagnosticVariable`s. `add_diagnostic_variable!` takes the same arguments
as the constructor for `DiagnosticVariable`, but also performs additional
checks. Similarly, if you want to retrieve a diagnostic from `ALL_DIAGNOSTICS`,
use the `get_diagnostic_variable` function.
