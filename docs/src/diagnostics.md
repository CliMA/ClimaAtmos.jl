# Computing and saving diagnostics

## I want to compute and output a diagnostic variable

### From a YAML file

If you configure your simulation with YAML files, there are two options that are
important to know about. When `output_default_diagnostics` is set to `true`, the
default diagnostics for the given atmospheric model will be output. Note that
they might be incompatible with your simulation (e.g., you want to output
hourly maxima when the timestep is 4 hours).

Second, you can specify the diagnostics you want to output directly in the
`diagnostics` section of your YAML file. For instance:
```
diagnostics:
  - short_name: rhoa
    output_name: a_name
    period: 3hours
    writer: nc
    compute_every: 1hours
  - reduction_time: average
    short_name: rhoa
    period: 12hours
    writer: h5
    compute_every: 2steps
```
This adds two diagnostics (both for `rhoa`). The `period` keyword
identifies the period over which to compute the reduction and how often to save
to disk. `output_name` is optional, and if provided, it identifies the name of the
output file. The `compute_every` keyword identifies how often the field should
be computed.

For multiple diagnostics with the same specs, it is also possible to directly
pass a vector of `short_names`, as in
```
diagnostics:
  - short_name: [rhoa, ua, ta]
    reduction_time: average
    period: 12hours
```

The default `writer` is NetCDF. If `writer` is `nc` or `netcdf`, the output is
remapped non-conservatively on a Cartesian grid and saved to a NetCDF file.
Currently, only 3D fields on cubed spheres are supported.

#### Writing in pressure coordinates

!!! compat "Compatibility"
    This is only available in versions after ClimaAtmos v0.35.2.

You can write diagnostics to NetCDF files in pressure coordinates by setting
`pressure_coordinates` to true. This replaces the vertical dimension `z` in
the NetCDF files with the dimension `pressure_level`. For more information about
writing diagnostics in pressure coordinates, see the
[documentation](https://clima.github.io/ClimaDiagnostics.jl/dev/writers/#Output-diagnostics-in-pressure-coordinates)
in ClimaDiagnostics.

```
diagnostics:
  - short_name: [pfull, wa, va, rv, hus, ke]
    period: 1days
    pressure_coordinates: true
```

### From a script

The simplest way to get started with diagnostics is to use the defaults for your
atmospheric model. `ClimaAtmos` defines a function `default_diagnostic`. You
can execute this function on an `AtmosModel` or on any of its fields to obtain a
list of diagnostics ready to be passed to the simulation. So, for example

```julia

model = ClimaAtmos.AtmosModel(..., microphysics_model = ClimaAtmos.DryModel(), ...)

diagnostics = ClimaAtmos.default_diagnostics(model)
# => List of diagnostics that include the ones specified for the DryModel
```

Technically, the diagnostics are represented as `ScheduledDiagnostic` objects,
which contain information about what variable has to be computed, how often,
where to save it, and so on (read below for more information on this). You can
construct your own lists of `ScheduledDiagnostic`s starting from the variables
defined by `ClimaAtmos`. The `DiagnosticVariable`s in `ClimaAtmos` are
identified with by the short and unique name, so that you can access them
directly with the function `diagnostic_variable`. One way to do so is by
using the provided convenience functions for common operations, e.g., continuing
the previous example

```julia

push!(diagnostics, daily_max("air_density", "air_temperature"))
```

Now `diagnostics` will also contain the instructions to compute the daily
maximum of `air_density` and `air_temperature`.

The diagnostics that are built-in `ClimaAtmos` are collected in [Available
diagnostic variables](@ref).

If you are using `ClimaAtmos` with a script-based interface, you have access to
the complete flexibility in your diagnostics. Read the section about the
low-level interface to see how to implement custom diagnostics, reductions, or
writers.

### The low-level interface

Check out the documentation for the `ClimaDiagnostics` to find more information
about the low level interface.

## The NetCDF output

The NetCDF writer in `ClimaAtmos` saves different diagnostics to different files
in the same output folder. Files are named after a combination of the diagnostic
variable `short_name`, and the details of the temporal reduction. Inside each
NetCDF file, there is only one diagnostic variable, along with the various
dimensions (e.g., `lat`, `lon`, and `z`/`z_reference`).

When topography is present, a new 1D dimension is defined `z_reference`. This
dimension does not have direct physical meaning but can be assumed to be the "z"
axis. Along with dimension, a new variable `z` is saved to the NetCDF file. In
this case, `z` is a multidimensional array (in general 3D). `z[i, j, k]` which
defines the elevation on the sea level of the point of indices `[i, j, k]`.

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
                `ClimaAtmos` have to have unique `short_name`s.

- `long_name`: Name used to describe the variable in the output file as attribute.

- `standard_name`: Standard name, as in
  [CF
  conventions](http://cfconventions.org/Data/cf-standard-names/71/build/cf-standard-name-table.html)

- `units`: Physical units of the variable.

- `comments`: More verbose explanation of what the variable is, or comments related to how
              it is defined or computed.

In `ClimaAtmos`, we follow the [CMIP6 MIP table](https://airtable.com/appYNLuWqAgzLbhSq/shrKcLEdssxb8Yvcp/tblL7dJkC3vl5zQLb)
for short names and long names where available. Standard names in the table are not used.

### Compute function

The other piece of information needed to specify a `DiagnosticVariable` is a
function `compute`. Schematically, a `compute` has to look like
```julia
function compute(state, cache, time)
    return ... # Calculations with the state and the cache
end
```
The function takes the `state`, `cache`, and `time` from the integrator and returns
the value of the diagnostic variable.

#### In-place computation

You can alternatively provide a `compute!` function. 
`compute!` takes a fourth argument, `out`, which is used to avoid extra memory allocations.

```julia
function compute!(out, state, cache, time)
    if isnothing(out)
        return ... # Calculations with the state and the cache
    else
        out .= ... # Calculations with the state and the cache
    end
end
```

The first time `compute!` is called, `out` is `nothing`, and the function has to
allocate memory and return its output. All the subsequent times, `out` will be
the pre-allocated area of memory, so the function has to write the new value in
place.

If your diagnostic depends on the details of the model, we recommend using
additional functions so that the correct one can be found through dispatching.
The following example demonstrates this using the `compute` interface.
For instance, if you want to compute relative humidity, which does not make
sense for dry simulations, you should define the functions

```julia
function compute_relative_humidity(state, cache, time, microphysics_model::T) where {T}
    error("Cannot compute relative_humidity with microphysics_model = $T")
end

function compute_relative_humidity(
    state, cache, time, microphysics_model::MoistMicrophysics,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.relative_humidity(thermo_params, cache.ᶜts))
end

compute_relative_humidity!(out, state, cache, time) =
    compute_relative_humidity!(out, state, cache, time, cache.atmos.microphysics_model)
```

This will return the correct relative humidity and throw informative errors when
it cannot be computed. We could specialize
`compute_relative_humidity` further if the relative humidity
were computed differently for `EquilibriumMicrophysics0M` and `NonEquilibriumMicrophysics`.

In `ClimaAtmos`, we define some helper functions to produce error messages, so
the above code can be written as
```julia
function compute_relative_humidity(
    state, cache, time, microphysics_model::MoistMicrophysics,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.relative_humidity(thermo_params, cache.ᶜts))
end

compute_relative_humidity!(out, state, cache, time) =
    compute_relative_humidity!(out, state, cache, time, cache.atmos.microphysics_model)
compute_relative_humidity!(_, _, _, _, model) =
    error_diagnostic_variable("relative_humidity", model)
```

### The `ClimaAtmos` `DiagnosticVariable`s

`ClimaAtmos` comes with a collection of pre-defined `DiagnosticVariable`, index
with their `short_name`s. If you are extending `ClimaAtmos` and want to add a
new diagnostic variable, go ahead and look at the files we `include` in
`diagnostics/Diagnostics.jl`. You can add more diagnostics in those files or add
a new one. We provide a convenience function, `add_diagnostic_variable!` to add
new `DiagnosticVariable`s. `add_diagnostic_variable!` take the same arguments as
the constructor for `DiagnosticVariable`, but also performs additional checks.
Similarly, if you want to retrieve a diagnostic from `ALL_DIAGNOSTICS`, use the
`get_diagnostic_variable`function.
