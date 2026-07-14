# Setups

A setup defines the initial conditions for a simulation case. At its core, a
setup is a struct that implements `center_initial_condition`, which returns a
physical state NamedTuple at each grid point. The physical state describes the
thermodynamic and kinematic state through temperature, pressure or density,
moisture, velocity and is converted into prognostic variables automatically
based on the model configuration.

## `center_initial_condition`

Every setup must implement this method. It is called pointwise over the grid
and returns a [`ClimaAtmos.Setups.physical_state`](@ref) NamedTuple. Only `T` and one of `p`
or `ρ` are required; all other fields default to zero.

For example, a minimal setup:

```julia
struct MySetup end

function Setups.center_initial_condition(::MySetup, local_geometry, params)
    z = local_geometry.coordinates.z
    FT = typeof(z)
    return physical_state(; T = FT(300), p = FT(101500))
end
```

```@docs
ClimaAtmos.Setups.physical_state
```

## `face_initial_condition`

Returns face (vertical interface) state variables. Must include `w` (vertical
velocity); may also include `w_draft` for EDMF updraft initialization.
Defaults to zero vertical velocity.

```@docs
ClimaAtmos.Setups.face_initial_condition
```

## `surface_condition`

Returns surface boundary data for the setup as a NamedTuple
`(; flux_scheme, temperature, overrides)`. Any field may be `nothing` to fall
through to the config-based default. See the
[Surface Conditions](@ref "Surface Conditions") page for what each field means
and the available options.

Not all setups need this — only those that prescribe case-specific surface
properties (e.g., roughness length, surface fluxes, surface temperature).

```@docs
ClimaAtmos.Setups.surface_condition
```

## `overwrite_initial_state!`

For file-based setups (e.g., ERA5, GCM-driven) that operate on the full
prognostic state `Y` rather than pointwise. Called after the standard
pointwise initialization and overwrites fields in-place with regridded file
data. Defaults to a no-op.

```@docs
ClimaAtmos.Setups.overwrite_initial_state!
```

## SCM Forcing Methods

Single-column setups can provide forcing profiles that replace the
corresponding YAML config keys. When a method returns `nothing` (the
default), the config key is used instead.

```@docs
ClimaAtmos.Setups.subsidence_forcing
ClimaAtmos.Setups.large_scale_advection_forcing
ClimaAtmos.Setups.coriolis_forcing
```

## Model Methods

Setups can return model objects directly. When a method returns `nothing`
(the default), the model construction layer falls through to config-based
dispatch.

```@docs
ClimaAtmos.Setups.external_forcing
ClimaAtmos.Setups.insolation_model
ClimaAtmos.Setups.surface_temperature_model
ClimaAtmos.Setups.prescribed_flow_model
ClimaAtmos.Setups.radiation_model
```

## Defining a Case in a Runscript

### A data-driven column case

For an externally-driven single-column case, no new setup is needed:
[`ClimaAtmos.Setups.ForcingFromFile`](@ref) builds the initial condition,
external forcing, surface treatment, and insolation from a single forcing file
in the native ClimaColumn format. See the "Column forcing datasets"
section of the Single Column Models page for the file layout and how to add a
new format as a small dataset module.

The cleanest runscript drives the case through a config dictionary. It merges
over the defaults and wires the setup's forcing, insolation, and surface models
into the `AtmosModel` for you:

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
models must be threaded into the `AtmosModel` explicitly (this will be addressed, tracked by [#4696](https://github.com/CliMA/ClimaAtmos.jl/issues/4696)).

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
multiple `Nudging` terms, e.g. relax temperature only above an inversion:

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
[Nonstandard forcing behavior from a runscript](@ref) on the Single Column
Models page.

### A custom analytic case

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

## Available Setups

### SCM Cases

```@docs
ClimaAtmos.Setups.Bomex
ClimaAtmos.Setups.Rico
ClimaAtmos.Setups.Soares
ClimaAtmos.Setups.GABLS
ClimaAtmos.Setups.GATE_III
ClimaAtmos.Setups.DYCOMS
ClimaAtmos.Setups.TRMM_LBA
ClimaAtmos.Setups.ISDAC
ClimaAtmos.Setups.SimplePlume
ClimaAtmos.Setups.PrecipitatingColumn
ClimaAtmos.Setups.ShipwayHill2012
ClimaAtmos.Setups.RCEMIPIIProfile
```

### Global Cases

```@docs
ClimaAtmos.Setups.DecayingProfile
ClimaAtmos.Setups.IsothermalProfile
ClimaAtmos.Setups.ConstantBuoyancyFrequencyProfile
ClimaAtmos.Setups.DryBaroclinicWave
ClimaAtmos.Setups.MoistBaroclinicWave
ClimaAtmos.Setups.MoistBaroclinicWaveWithEDMF
ClimaAtmos.Setups.DryDensityCurrentProfile
ClimaAtmos.Setups.RisingThermalBubbleProfile
ClimaAtmos.Setups.MoistAdiabaticProfileEDMFX
```

### Data-Driven

```@docs
ClimaAtmos.Setups.GCMDriven
ClimaAtmos.Setups.ForcingFromFile
ClimaAtmos.Setups.MoistFromFile
ClimaAtmos.Setups.WeatherModel
ClimaAtmos.Setups.AMIPFromERA5
```
