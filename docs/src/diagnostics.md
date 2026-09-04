# Computing and saving diagnostics

## I want to compute and output a diagnostic variable

### From a YAML file

If you configure your simulation with YAML files, there are two important
options. When `output_default_diagnostics` is set to `true`, the
default diagnostics for the given atmospheric model will be output. These may
be incompatible with your simulation, if for instance you ask for hourly maxima
when the timestep is 4 hours.

Second, you can specify the diagnostics you want to output directly in the
`diagnostics` section of your YAML file. For instance:

```yaml
diagnostics:
  - short_name: rhoa
    output_name: a_name
    period: 3hours
    writer: nc
  - reduction_time: average
    short_name: rhoa
    period: 12hours
    writer: h5
    compute_every: 2steps
```

This adds two diagnostics (both for `rhoa`). The `period` keyword
identifies the period over which to compute the reduction and how often to save
to disk. `reduction_time` is one of `inst` (instantaneous, the default),
`average`, `max`, or `min`. `output_name` is optional, and if provided, it
identifies the name of the output file. The `compute_every` keyword identifies
how often the field should be computed; it applies only when a
`reduction_time` is specified, and **defaults to every timestep**, so it is
what to change when a reduced diagnostic is expensive to compute. Without a
reduction, the field is computed on the output schedule.

For multiple diagnostics with the same specs, you can also pass a vector of
`short_names` directly, as in

```yaml
diagnostics:
  - short_name: [rhoa, ua, ta]
    reduction_time: average
    period: 12hours
```

The accepted `writer` values are `nc`/`netcdf` (the default), `h5`/`hdf5`, and
`dict` (in-memory, for testing). If `writer` is `nc` or `netcdf`, the output is
remapped non-conservatively on a Cartesian grid and saved to a NetCDF file.
Remapping is most commonly used for cubed-sphere runs; column and box
configurations are also supported. By default
(`netcdf_output_at_levels: true`), fields are written at the model levels, with
no vertical interpolation.

!!! note "Did you know?"

    For the `period`, you can also specify `"monthly"`, `"weekly"`, and
    `"daily"`. These options align the reductions to start at the beginning of
    each month, week, and day, respectively.

    For example:

      - If `period: monthly` and `reduction_time: average` are used, and the
        simulation begins on `2010-01-15`, then the first time saved represents
        the time average of the second half of January.
      - The next time saved represents the time average of the data for February,
        and so on.

    This is useful to account for spinup.

#### Writing in pressure coordinates

You can write diagnostics to NetCDF files in pressure coordinates by setting
`pressure_coordinates` to true. This replaces the vertical dimension `z` in
the NetCDF files with the dimension `pressure_level`. For more information about
writing diagnostics in pressure coordinates, see the
[documentation](https://clima.github.io/ClimaDiagnostics.jl/dev/writers/#Output-diagnostics-in-pressure-coordinates)
in ClimaDiagnostics.

```yaml
diagnostics:
  - short_name: [pfull, wa, va, rv, hus, ke]
    period: 1days
    pressure_coordinates: true
```

### From a script

The simplest way to get started with diagnostics is to use the defaults for your
atmospheric model. The `ClimaAtmos.Diagnostics` submodule defines a function
`default_diagnostics`. You can execute this function on an `AtmosModel` or on any
of its fields to obtain a list of diagnostics ready to be passed to the
simulation. So, for example

```julia
import ClimaAtmos as CA
import ClimaAtmos.Diagnostics as CAD

model = CA.AtmosModel(grid; microphysics_model = CA.DryModel())

diagnostics = CAD.default_diagnostics(
    model, duration, start_date, t_start;
    output_writer, topography,
)
# => List of diagnostics that include the ones specified for the DryModel
```

(The block is schematic: `duration`, `start_date`, `t_start`, `output_writer`,
and `topography` stand for values you supply, and the last two are required
keyword arguments. `DiagnosticsConfig(; default = true)` is the higher-level
entry point that supplies all of them for you.)

Technically, the diagnostics are represented as `ScheduledDiagnostic` objects,
which contain information about what variable has to be computed, how often,
where to save it, and so on (read below for more information on this). You can
construct your own lists of `ScheduledDiagnostic`s starting from the variables
defined by `ClimaAtmos`. The `DiagnosticVariable`s in `ClimaAtmos` are
identified by their short, unique names, so that you can access them
directly with the function `CAD.get_diagnostic_variable`. One way to do so is by
using the provided convenience functions for common operations, e.g., continuing
the previous example

```julia
append!(
    diagnostics,
    CAD.daily_maxs(FT, "rhoa", "ta"; output_writer, start_date, t_start),
)
```

Now `diagnostics` will also contain the instructions to compute the daily
maxima of the air density (`rhoa`) and air temperature (`ta`); `FT` is the
simulation's float type (e.g. `Float32`).

The diagnostics built into `ClimaAtmos` are collected in [Available
diagnostic variables](@ref).

If you are using `ClimaAtmos` with a script-based interface, you have complete
flexibility in your diagnostics. Read the section about the
low-level interface to see how to implement custom diagnostics, reductions, or
writers.

### The low-level interface

See the `ClimaDiagnostics` documentation for more information about the
low-level interface.

## The NetCDF output

The NetCDF writer in `ClimaAtmos` saves different diagnostics to different files
in the same output folder. Files are named after a combination of the diagnostic
variable `short_name` and the details of the temporal reduction. Inside each
NetCDF file, there is only one diagnostic variable, along with the various
dimensions (e.g., `lat`, `lon`, and `z`/`z_reference`).

When topography is present, a new 1D dimension `z_reference` is defined. This
dimension does not have direct physical meaning but can be assumed to be the "z"
axis. Along with this dimension, a new variable `z` is saved to the NetCDF file.
In this case, `z` is a multidimensional array (in general 3D), and `z[i, j, k]`
gives the elevation above sea level of the point with indices `[i, j, k]`.

## Adding a new diagnostic variable

Defining new diagnostic variables is developer territory; see
[Adding a Diagnostic Variable](extending_diagnostics.md) in the Developer
Guide.
