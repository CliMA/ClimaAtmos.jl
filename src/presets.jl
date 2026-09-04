module Presets

import ..AtmosModel, ..AtmosSimulation
import ..SphereGrid
import ..Setups
import ..ClimaAtmosParameters
import ..Parameters as CAP
import ..DryModel
import ..EquilibriumMicrophysics0M
import ..NonEquilibriumMicrophysics1M
import ..GridScaleCloud
import ..SurfaceConditions
import ..IdealizedInsolation
import ..Explicit
import ..PrognosticEDMFX
import ..EDMFXModel
import ..InvZEntrainment, ..BuoyancyVelocityDetrainment
import ..SmoothMinimumBlending

# ============================================================================
# Model presets: A bundle of defaults that the case setup and explicit kwargs
# override. Each returns a NamedTuple of AtmosModel leaf kwargs, passed to the
# `defaults` kwarg: `AtmosModel(grid; setup, defaults = Presets.dry())`.
# ============================================================================

"""
    dry(; kwargs...)

Dry-atmosphere model defaults (`microphysics_model = DryModel()`).
Keyword arguments are merged in, and override the preset. Any `AtmosModel`
field keyword argument is accepted (see [`AtmosModel`](@ref)); names that
are not fields raise an error when the model is built.

# Returns

A NamedTuple of `AtmosModel` keyword arguments, for its `defaults` slot.

# Examples

```julia
import ClimaAtmos as CA
model = CA.AtmosModel(grid; defaults = CA.Presets.dry())
```
"""
dry(; kwargs...) = (; microphysics_model = DryModel(), kwargs...)

"""
    equil_moist_0m(; kwargs...)

Equilibrium-moisture model defaults: 0-moment microphysics, grid-scale
cloud, prescribed zonally-symmetric SST, and idealized insolation.
Keyword arguments are merged in, and override the preset. Any `AtmosModel`
field keyword argument is accepted (see [`AtmosModel`](@ref)); names that
are not fields raise an error when the model is built.

# Returns

A NamedTuple of `AtmosModel` keyword arguments, for its `defaults` slot.

# Examples

```julia
import ClimaAtmos as CA
model = CA.AtmosModel(grid; defaults = CA.Presets.equil_moist_0m())
```
"""
equil_moist_0m(; kwargs...) = (;
    microphysics_model = EquilibriumMicrophysics0M(),
    cloud_model = GridScaleCloud(),
    temperature = SurfaceConditions.AnalyticTemperature(
        Setups.zonally_symmetric_temperature,
    ),
    insolation = IdealizedInsolation(),
    kwargs...,
)

"""
    nonequil_moist_1m(; kwargs...)

[`equil_moist_0m`](@ref) with 1-moment non-equilibrium microphysics and
explicit microphysics tendency timestepping in place of 0-moment equilibrium.
Keyword arguments are merged in, and override the preset. Any `AtmosModel`
field keyword argument is accepted (see [`AtmosModel`](@ref)); names that
are not fields raise an error when the model is built.

# Returns

A NamedTuple of `AtmosModel` keyword arguments, for its `defaults` slot.

# Examples

```julia
import ClimaAtmos as CA
model = CA.AtmosModel(grid; defaults = CA.Presets.nonequil_moist_1m())
```
"""
nonequil_moist_1m(; kwargs...) = (;
    microphysics_model = NonEquilibriumMicrophysics1M(),
    microphysics_tendency_timestepping = Explicit(),
    cloud_model = GridScaleCloud(),
    temperature = SurfaceConditions.AnalyticTemperature(
        Setups.zonally_symmetric_temperature,
    ),
    insolation = IdealizedInsolation(),
    kwargs...,
)

"""
    prognostic_edmf([FT = Float32]; area_fraction, n_updrafts, prognostic_tke, kwargs...)

Equilibrium-moist model defaults with the `PrognosticEDMFX`
turbulence-convection scheme. This uses `Generalized` entrainment/detrainment,
SGS mass & diffusive fluxes, and non-hydrostatic pressure drag. Also enables
prognostic updraft vertical diffusion and the relaxation filter on negative
updraft velocities (matches the canonical `prognostic_edmfx_*` configs).
Mixing-length scales are blended with `SmoothMinimumBlending`, and the
microphysics is 0-moment equilibrium with grid-scale cloud.

# Arguments

  - `FT = Float32`: Float type of the scheme's parameters.

# Keyword Arguments

  - `area_fraction = FT(1e-5)`: "Small" updraft area threshold passed to
    `PrognosticEDMFX` [-].
  - `n_updrafts = 1`: Number of updraft subdomains [-].
  - `prognostic_tke = true`: Whether TKE is prognostic.
  - `kwargs...`: Merged in, overriding the preset. Any `AtmosModel` field
    keyword argument is accepted (see [`AtmosModel`](@ref)); names that are not
    fields raise an error when the model is built.

# Returns

A NamedTuple of `AtmosModel` keyword arguments, for its `defaults` slot.

# Examples

```julia
import ClimaAtmos as CA
model = CA.AtmosModel(grid; defaults = CA.Presets.prognostic_edmf(Float64; n_updrafts = 2))
```
"""
function prognostic_edmf(
    ::Type{FT} = Float32;
    area_fraction = FT(1e-5),
    n_updrafts = 1,
    prognostic_tke = true,
    kwargs...,
) where {FT}
    return (;
        microphysics_model = EquilibriumMicrophysics0M(),
        cloud_model = GridScaleCloud(),
        turbconv_model = PrognosticEDMFX(;
            n_updrafts, prognostic_tke, area_fraction,
        ),
        edmfx_model = EDMFXModel(;
            entr_model = InvZEntrainment(),
            detr_model = BuoyancyVelocityDetrainment(),
            sgs_mass_flux = true,
            sgs_diffusive_flux = true,
            nh_pressure = true,
            vertical_diffusion = true,
            filter = true,
            scale_blending_method = SmoothMinimumBlending(),
        ),
        kwargs...,
    )
end

"""
    prognostic_edmf_1m([FT = Float32]; kwargs...)

[`prognostic_edmf`](@ref) with 1-moment non-equilibrium microphysics and explicit
microphysics tendency timestepping (matches the canonical `prognostic_edmfx_*`
configs that use `microphysics_model: "1M"`). All keyword arguments are
forwarded to [`prognostic_edmf`](@ref).

# Returns

A NamedTuple of `AtmosModel` keyword arguments, for its `defaults` slot.

# Examples

```julia
import ClimaAtmos as CA
model = CA.AtmosModel(grid; defaults = CA.Presets.prognostic_edmf_1m(Float32))
```
"""
function prognostic_edmf_1m(::Type{FT} = Float32; kwargs...) where {FT}
    return prognostic_edmf(
        FT;
        microphysics_model = NonEquilibriumMicrophysics1M(),
        microphysics_tendency_timestepping = Explicit(),
        kwargs...,
    )
end

# ============================================================================
# Simulation presets — return an AtmosSimulation
# ============================================================================

"""
    aquaplanet([FT = Float32]; grid, params, setup, model_kwargs, kwargs...)

Aquaplanet simulation preset: global [`SphereGrid`](@ref) with
[`equil_moist_0m`](@ref) physics (0M microphysics, prescribed zonally-symmetric
SST, idealized insolation). Uses the default `DecayingProfile` setup from
[`AtmosModel`](@ref), and the `dt = 600 s` and `t_end = 10 days` defaults of
[`AtmosSimulation`](@ref). `model_kwargs` are forwarded to [`AtmosModel`](@ref)
and override the preset physics; all other keyword arguments go to
[`AtmosSimulation`](@ref).

# Returns

An [`AtmosSimulation`](@ref).

# Examples

```julia
import ClimaAtmos as CA
simulation = CA.Presets.aquaplanet(Float32; t_end = "1days")
```
"""
function aquaplanet(
    ::Type{FT} = Float32;
    params = ClimaAtmosParameters(FT),
    grid = SphereGrid(FT; radius = CAP.planet_radius(params)),
    setup = nothing,
    model_kwargs = (;),
    kwargs...,
) where {FT}
    model = AtmosModel(
        grid;
        params,
        setup,
        defaults = equil_moist_0m(),
        model_kwargs...,
    )
    return AtmosSimulation(model; kwargs...)
end

end # module
