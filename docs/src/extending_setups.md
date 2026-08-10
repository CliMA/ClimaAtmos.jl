# Adding a Setup

How to define a new simulation case in a runscript, using the setup
interface documented on the [Setups](setups.md) reference page.

## A data-driven column case

For an externally-driven single-column case, no new setup is needed:
[`ClimaAtmos.Setups.ForcingFromFile`](@ref) builds the initial condition,
external forcing, surface treatment, and insolation from a single forcing file
in the native ClimaColumn format. See the
[Column Datasets](column_datasets_reference.md) reference page for the file
layout, and [Adding a Column Dataset](extending_column_datasets.md) for
adding a new format as a small dataset module.

The cleanest runscript drives the case through a config dictionary. It merges
over the defaults and wires the setup's forcing, insolation, and surface models
into the [`AtmosModel`](@ref ClimaAtmos.AtmosModel) for you:

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig(
    Dict(
        "config" => "column",
        "initial_condition" => "ForcingFromFile",
        "external_forcing_file" => "path/to/forcing.nc",
        "start_date" => "20070701",
        "turbconv" => "prognostic_edmfx",
        "dt" => "50secs",
        "t_end" => "30hours",
    ),
)
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

The forcing is a tuple of
[`AbstractForcingTerm`](@ref ClimaAtmos.AbstractForcingTerm)s (`HorizontalAdvection()`, `VerticalFluctuation()`,
`Nudging(variables...; timescale, mask)`, `Subsidence()`) passed to the setup's
`forcing` slot. Note that the `AtmosSimulation(; model, setup)` constructor uses
`setup` only for the initial state, so the setup's forcing / insolation / surface
models must be threaded into the `AtmosModel` explicitly (tracked by [#4696](https://github.com/CliMA/ClimaAtmos.jl/issues/4696)).

```julia
import ClimaAtmos as CA
import Dates

FT = Float64
params = CA.ClimaAtmosParameters(FT)

setup = CA.Setups.ForcingFromFile(
    "path/to/forcing.nc",
    "20070701";
    # horizontal advection only (drop the other default terms)
    forcing = (CA.HorizontalAdvection(),),
)

surface = CA.Setups.surface_condition(setup, params)
model = CA.AtmosModel(;
    external_forcing = CA.Setups.external_forcing(setup, FT),
    insolation = CA.Setups.insolation_model(setup),
    temperature = CA.Setups.surface_temperature_model(setup),
    flux_scheme = surface.flux_scheme,
    # ...
)
grid = CA.ColumnGrid(FT; z_elem = 63, z_max = FT(60e3), z_stretch = true)
simulation = CA.AtmosSimulation{FT}(;
    model, setup, grid, params,
    start_date = Dates.DateTime(2007, 7, 1), dt = 50, t_end = 30 * 3600,
)
CA.solve_atmos!(simulation)
```

Per-variable relaxation timescales and height-dependent masks compose as
multiple [`Nudging`](@ref ClimaAtmos.Nudging) terms, e.g. relax temperature only
above an inversion:

```julia
z_inv = 800.0
forcing = (
    CA.HorizontalAdvection(),
    CA.Nudging(:ta; timescale = 3600.0, mask = z -> z < z_inv ? 0.0 : 1.0),
    CA.Nudging(:ua, :va; timescale = 7200.0),
    CA.Subsidence(),
)
```

For nonstandard forcing (per-variable relaxation timescales, custom height or
time masks, an in-memory data source), define a small forcing type in the
runscript instead. See
[Nonstandard forcing behavior from a runscript](@ref) on the
[Adding a Column Dataset](extending_column_datasets.md) page.

## A custom analytic case

Define a type and extend the setup interface directly:

```julia
import ClimaAtmos as CA

struct MyCase end

function CA.Setups.center_initial_condition(
    ::MyCase,
    local_geometry,
    params,
)
    FT = eltype(params)
    (; z) = local_geometry.coordinates
    T = FT(300) - FT(0.01) * z
    p = FT(101500)
    return CA.Setups.physical_state(; T, p)
end

setup = MyCase()
simulation = CA.AtmosSimulation{Float64}(; setup, model, grid)
```

Optionally extend the other setup methods documented above in the same
runscript.
