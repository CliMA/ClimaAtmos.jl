# Computing and saving diagnostics

## I want to compute and output a diagnostic variable

### From a script

The simplest way to get started with diagnostics is to use the defaults for your
atmospheric model. `ClimaAtmos` defines a function `get_default_diagnostic`. You
can execute this function on an `AtmosModel` or on any of its fields to obtain a
list of diagnostics ready to be passed to the simulation. So, for example

```julia

model = ClimaAtmos.AtmosModel(..., moisture_model = ClimaAtmos.DryModel(), ...)

diagnostics = ClimaAtmos.get_default_diagnostics(model)
# => List of diagnostics that include the ones specified for the DryModel
```

Technically, the diagnostics are represented as `ScheduledDiagnostic` objects,
which contain information about what variable has to be computed, how often,
where to save it, and so on (read below for more information on this). You can
construct your own lists of `ScheduledDiagnostic`s starting from the variables
defined by `ClimaAtmos`. The diagnostics that `ClimaAtmos` knows how to compute
are collected in a global dictionary called `ALL_DIAGNOSTICS`. The variables in
`ALL_DIAGNOSTICS` are identified with by the short and unique name, so that you
can access them directly. One way to do so is by using the provided convenience
functions for common operations, e.g., continuing the previous example

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

Diagnostics are computed and output through callbacks to the main integrator.
`ClimaAtmos` produces the list of callbacks from a ordered list of
`ScheduledDiagnostic`s.

A `ScheduledDiagnostic` is an instruction on how to compute and output a given
`DiagnosticVariable` (see below), along with specific choices regarding
reductions, compute/output frequencies, and so on. It can be point-wise in space
and time, or can be the result of a reduction in a period that is defined by
`output_every` (e.g., the daily average temperature).

More specifically, a `ScheduledDiagnostic` contains the following pieces of data

- `variable`: The diagnostic variable that has to be computed and output.
- `output_every`: Frequency of how often to save the results to disk.
- `output_writer`: Function that controls out to save the computed diagnostic
  variable to disk.
- `reduction_time_func`: If not `nothing`, the `ScheduledDiagnostic` receives an
  area of scratch space `acc` where to accumulate partial results. Then, at
  every `compute_every`, `reduction_time_func` is computed between the
  previously stored value in `acc` and the new value. This implements a running
  reduction. For example, if `reduction_time_func = max`, the space `acc` will
  hold the running maxima of the diagnostic. `acc` is reset after output.
- `reduction_space_func`: NOT IMPLEMENTED YET
- `compute_every`: Run the computations every `compute_every`. This is not
  particularly useful for point-wise diagnostics, where we enforce that
  `compute_every` = `output_every`. `compute_every` has to evenly divide
  `output_every`.
- `pre_output_hook!`: Function that has to be run before saving to disk for
   reductions (mostly used to implement averages). The function
   `pre_output_hook!` is called with two arguments: the value accumulated during
   the reduction, and the number of times the diagnostic was computed from the
   last time it was output.

To implement operations like the arithmetic average, the `reduction_time_func`
has to be chosen as `+`, and a `pre_output_hook!` that renormalize `acc` by the
number of samples has to be provided. `pre_output_hook!` should mutate the
accumulator in place. The return value of `pre_output_hook!` is discarded. An
example of `pre_output_hook!` to compute the arithmetic average is
`pre_output_hook!(acc, N) = @. acc = acc / N`. `ClimaAtmos` provides an alias to
the function needed to compute averages `ClimaAtmos.average_pre_output_hook!`.

For custom reductions, it is necessary to also specify the identity of operation
by defining a new method to `identity_of_reduction`. `identity_of_reduction` is
a function that takes a `op` argument, where `op` is the operation for which we
want to define the identity. For instance, for the `max`,
`identity_of_reduction` would be `identity_of_reduction(::typeof{max}) = -Inf`.
The identities known to `ClimaAtmos` are defined in the
`diagnostics/reduction_identities.jl` file. The identity is needed to ensure
that we have a neutral state for the accumulators that are used in the
reductions.

A key entry in a `ScheduledDiagnostic` is the `output_writer`, the function
responsible for saving the output to disk. `output_writer` is called with three
arguments: the value that has to be output, the `ScheduledDiagnostic`, and the
integrator. Internally, the integrator contains extra information (such as the
current timestep), so the `output_writer` can implement arbitrarily complex
behaviors. It is responsibility of the `output_writer` to properly use the
provided information for meaningful output. `ClimaAtmos` provides functions that
return `output_writer`s for standard operations. The main one is currently
`HDF5Writer`, which should be enough for most use cases. To use it, just
initialize a `ClimaAtmos.HDF5Writer` object with your choice of configuration
and pass it as a `output_writer` argument to the `ScheduledDiagnostic`. More
information about the options supported by `ClimaAtmos.HDF5Writer` is available
in its constructor.

There are two flavors of `ScheduledDiagnostic`s:
`ScheduledDiagnosticIterations`, and `ScheduledDiagnosticTime`. The main
difference between the two is the domain of definition of their frequencies,
which is measured in timesteps for the first one, and in seconds for the other
one. `ScheduledDiagnosticTime`s offer a more natural way to set up physically
meaningful reductions (e.g., we want a daily average). However, it is not clear
what to do when the period does not line up with the timestep. What is the
expected behavior when we want a daily average but our timestep is of 10 hours?
There are multiple possible answer to this question. In `ClimaAtmos`, we enforce
that all the periods are multiples of the timestep. With this restriction, a
`ScheduledDiagnosticTime` can be translated to a
`ScheduledDiagnosticIterations`, where the problem is perfectly represented (in
this sense, one can think of `ScheduledDiagnosticIterations` as being as
internal representation and as `ScheduledDiagnosticTime` being the user-facing
one).

`ScheduledDiagnosticTime` behave like as `ScheduledDiagnosticIterations`, with
the exception that they can take a special value for `compute_every`, which can
be set to `:timestep` (the symbol) to ensure that the diagnostic is computed at
the end of every integration step. This is particularly convenient when defining
default diagnostics, as they should be largely independent on the choice of the
specific simulation being run (and its timestep).

Given a timestep `dt`, a `ScheduledDiagnosticIterations` can be obtained from a
`ScheduledDiagnosticTime` `sd` simply by calling
``ScheduledDiagnosticIterations(sd, dt)`.

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
                `ClimaAtmos` have to have unique `short_name`s. We follow the
                Coupled Model Intercomparison Project (CMIP) conventions.

- `long_name`: Name used to describe the variable in the output file.

- `units`: Physical units of the variable.

- `comments`: More verbose explanation of what the variable is, or comments related to how
              it is defined or computed.

In `ClimaAtmos`, we try to follow [this Google
spreadsheet](https://docs.google.com/spreadsheets/d/1qUauozwXkq7r1g-L4ALMIkCNINIhhCPx)
for variable naming, with a `long_name` the does not have spaces and capital
letters. [Standard
names](http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
are not used.

### Compute function

The other piece of information needed to specify a `DiagnosticVariable` is a
function `compute_from_integrator!`. Schematically, a `compute_from_integrator!` has to look like
```julia
function compute_from_integrator!(out, integrator)
    # FIXME: Remove this line when ClimaCore implements the broadcasting to enable this
    out .= # Calculcations with the state (= integrator.u) and the parameters (= integrator.p)
end
```
Diagnostics are implemented as callbacks function which pass the `integrator`
object (from `OrdinaryDiffEq`) to `compute_from_integrator!`.

`compute_from_integrator!` also takes a second argument, `out`, which is used to
avoid extra memory allocations (which hurt performance). If `out` is `nothing`,
and new area of memory is allocated. If `out` is a `ClimaCore.Field`, the
operation is done in-place without additional memory allocations.

If your diagnostic depends on the details of the model, we recommend using
additional functions so that the correct one can be found through dispatching.
For instance, if you want to compute relative humidity, which does not make
sense for dry simulations, you should define the functions

```julia
function compute_relative_humidity_from_integrator!(
    out,
    integrator,
    moisture_model::T,
) where {T}
    error("Cannot compute relative_humidity with moisture_model = $T")
end

function compute_relative_humidity_from_integrator!(
    out,
    integrator,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    thermo_params = CAP.thermodynamics_params(integrator.p.params)
    return TD.relative_humidity.(thermo_params, integrator.p.ᶜts)
end

compute_relative_humidity_from_integrator!(out, integrator) =
    compute_relative_humidity_from_integrator!(
        out,
        integrator,
        integrator.p.atmos,
    )
```

This will return the correct relative humidity and throw informative errors when
it cannot be computed. We could specialize
`compute_relative_humidity_from_integrator` further if the relative humidity
were computed differently for `EquilMoistModel` and `NonEquilMoistModel`.

### Adding to the `ALL_DIAGNOSTICS` dictionary

`ClimaAtmos` comes with a collection of pre-defined `DiagnosticVariable` in the
`ALL_DIAGNOSTICS` dictionary. `ALL_DIAGNOSTICS` maps a `short_name` with the
corresponding `DiagnosticVariable`.

If you are extending `ClimaAtmos` and want to add a new diagnostic variable to
`ALL_DIAGNOSTICS`, go ahead and look at the files we `include` in
`diagnostics/Diagnostics.jl`. You can add more diagnostics in those files or add
a new one. We provide a convenience function, `add_diagnostic_variable!` to add
new `DiagnosticVariable`s to the `ALL_DIAGNOSTICS` dictionary.
`add_diagnostic_variable!` take the same arguments as the constructor for
`DiagnosticVariable`, but also performs additional checks. So, use
`add_diagnostic_variable!` instead of editing the `ALL_DIAGNOSTICS` directly.