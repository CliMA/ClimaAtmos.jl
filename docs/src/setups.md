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

Returns a NamedTuple of surface boundary data (e.g., surface temperature `T`
and roughness length `z0`). Used by the surface model construction layer to
set surface boundary conditions. Not all setups need this — only those that
prescribe case-specific surface properties.

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

## Helper Functions

```@docs
ClimaAtmos.Setups.total_specific_energy
ClimaAtmos.Setups.get_density
ClimaAtmos.Setups.moist_static_energy
ClimaAtmos.Setups.hydrostatic_pressure_profile
ClimaAtmos.Setups.perturb_coeff
```

## Available Setups

### SCM Cases

```@docs
ClimaAtmos.Setups.Bomex
ClimaAtmos.Setups.Rico
```

### Data-Driven

```@docs
ClimaAtmos.Setups.GCMDriven
ClimaAtmos.Setups.InterpolatedColumnProfile
ClimaAtmos.Setups.MoistFromFile
ClimaAtmos.Setups.WeatherModel
ClimaAtmos.Setups.AMIPFromERA5
```
