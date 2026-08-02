# Setups

A setup defines the initial conditions for a simulation case. At its core, a
setup is a struct that implements `center_initial_condition`, which returns a
physical state NamedTuple at each grid point. The physical state describes the
thermodynamic and kinematic state through temperature, pressure or density,
moisture, and velocity, and is converted into prognostic variables
automatically based on the model configuration.

## `initial_state`

The entry point that builds the full prognostic state from a setup, calling
`center_initial_condition` and `face_initial_condition` pointwise and assembling
the prognostic variables selected by the model configuration.

```@docs
ClimaAtmos.Setups.initial_state
```

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
velocity); may also include `w_draft` for PROPHET updraft initialization.
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

Not all setups need this; only those that prescribe case-specific surface
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
dispatch. The exception is `surface_temperature_model`, whose default is an
`AnalyticTemperature` using `zonally_symmetric_temperature`.

```@docs
ClimaAtmos.Setups.external_forcing
ClimaAtmos.Setups.insolation_model
ClimaAtmos.Setups.surface_temperature_model
ClimaAtmos.Setups.prescribed_flow_model
ClimaAtmos.Setups.radiation_model
```

## Defining a case in a runscript

Worked walkthroughs for defining data-driven and analytic cases in a
runscript are in [Adding a Setup](extending_setups.md) in the Developer
Guide.

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
ClimaAtmos.Setups.Larcform1
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
