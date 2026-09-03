module Setups

import Adapt

import StaticArrays as SA
import ClimaInterpolations.Interpolation1D as CI1D
import ClimaCore.Geometry as Geometry
import ClimaCore: Fields
import Thermodynamics as TD
import Thermodynamics.TemperatureProfiles: DecayingTemperatureProfile, DryAdiabaticProfile
import AtmosphericProfilesLibrary as APL
import ClimaParams

import ..Parameters as CAP
import ..geopotential
import ..C12, ..C3
import ..background_p_and_T, ..background_u

# File-based IC infrastructure (overwrite_from_file.jl, ForcingFromFile.jl)
import Dates
import ClimaUtilities.SpaceVaryingInputs
import ClimaUtilities.ClimaArtifacts: @clima_artifact
import Interpolations as Intp
import NCDatasets as NC
import Statistics: mean
import ..ᶜinterp, ..ᶠinterp
import ..compute_kinetic
import ..weather_model_data_path
import ..parse_date
import ..pressure_to_height

# Model types for dispatch (used by prognostic_variables.jl)
import ..DryModel
import ..EquilibriumMicrophysics0M
import ..NonEquilibriumMicrophysics1M
import ..NonEquilibriumMicrophysics2M
import ..NonEquilibriumMicrophysics2MP3
import ..NonEquilibriumMicrophysics
import ..MoistMicrophysics
import ..PrognosticEDMFX
import ..EDOnlyEDMFX
import ..n_mass_flux_subdomains
import ..AbstractChemistryModel
import ..AtmosAerosols
import ..PrognosticSeaSalt
import ..AbstractPrognosticAerosol
import ..species_models, ..bin_names
import ..Parameters.ClimaAtmosParameters
import Thermodynamics.Parameters.ThermodynamicsParameters

# Model types returned by setup interface methods
import ..ISDACForcing
import ..ExternalDrivenTVForcing, ..default_forcing_terms
import ..ColumnDatasets
import ..ExternalTVInsolation, ..TimeVaryingInsolation
import ..RCEMIPIIInsolation
import ..ShipwayHill2012VelocityProfile
import ..RadiationDYCOMS, ..RadiationTRMM_LBA, ..RadiationISDAC
import ..SurfaceConditions
import ..SurfaceConditions:
    MoninObukhov, ExchangeCoefficients, HeatFluxes,
    SurfaceBoundaryOverrides,
    AnalyticTemperature, ExternalTemperature

# ============================================================================
# Layer 1 interface — implemented by each setup
# ============================================================================

"""
    face_initial_condition(setup, local_geometry, params)

Return the face (vertical interface) state of `setup` at one grid point.

Called pointwise by [`initial_state`](@ref), which converts the result into
face prognostic variables. The default is a state at rest.

# Returns

`(; w, w_draft)`, the grid-mean vertical velocity and the EDMF draft vertical
velocity [m/s]. The default is `(; w = 0, w_draft = 0)`; `w_draft` is used only
by a prognostic-EDMF configuration.
"""
function face_initial_condition(setup, local_geometry, params)
    FT = eltype(params)
    return (; w = FT(0), w_draft = FT(0))
end

"""
    overwrite_initial_state!(setup, Y, thermo_params)

Overwrite the initial state `Y` in place after it has been constructed, and
return `nothing`.

The extension point for file-based setups (e.g. `ForcingFromFile`, `WeatherModel`),
which regrid whole fields rather than working pointwise. Called by the
simulation setup after [`initial_state`](@ref). The default is a no-op.
"""
overwrite_initial_state!(setup, Y, thermo_params) = nothing

# ============================================================================
# SCM forcing interface — optional, for single-column setups
# ============================================================================

"""
    subsidence_forcing(setup, ::Type{FT})

Return the large-scale subsidence profile `z -> w_subsidence` [m/s] prescribed
by `setup`, or `nothing` for no subsidence (the default).

The model construction layer wraps a non-`nothing` profile in a
`LargeScaleSubsidence` and stores it as `atmos.subsidence`. There is no config
key for subsidence: it is owned by the setup.
"""
subsidence_forcing(setup, ::Type{FT}) where {FT} = nothing

"""
    large_scale_advection_forcing(setup, ::Type{FT})

Return the prescribed large-scale advective tendencies of `setup`, or `nothing`
for none (the default).

# Returns

`(; prof_dTdt, prof_dqtdt)`, the raw profile functions of the
AtmosphericProfilesLibrary form `(exner, z) -> dTdt` [K/s] and `z -> dqtdt`
[kg/kg/s]. The model construction layer adapts their argument lists and wraps
them in a `LargeScaleAdvection` stored as `atmos.ls_adv`; there is no config
key for them.
"""
large_scale_advection_forcing(setup, ::Type{FT}) where {FT} = nothing

"""
    coriolis_forcing(setup, ::Type{FT})

Return the single-column Coriolis forcing of `setup`, or `nothing` for none
(the default).

# Returns

`(; prof_ug, prof_vg, coriolis_param)`, the geostrophic-wind profiles
`z -> u_g`, `z -> v_g` [m/s] and the Coriolis parameter [1/s]. Stored as
`atmos.scm_coriolis` by the model construction layer; there is no config key
for it.
"""
coriolis_forcing(setup, ::Type{FT}) where {FT} = nothing

"""
    surface_condition(setup, params)

Return the surface pieces prescribed by `setup`.

Consumed by `AtmosSurface(::AtmosConfig, params, FT; setup_type)`, where a
non-`nothing` field takes precedence over the corresponding config key. Only
setups with case-specific surface properties (roughness, prescribed fluxes, a
case SST) need to extend this.

# Returns

`(; flux_scheme, temperature, overrides)`: a
`SurfaceConditions.SurfaceParameterization`, a
`SurfaceConditions.SurfaceTemperature`, and a
`SurfaceConditions.SurfaceBoundaryOverrides`. Each defaults to `nothing`,
falling through to the configuration. Note that `temperature` is used only when
`prognostic_surface` is `"PrescribedSST"`.
"""
surface_condition(setup, params) =
    (; flux_scheme = nothing, temperature = nothing, overrides = nothing)

# ============================================================================
# Model interface — optional, returns model objects directly
# ============================================================================

"""
    external_forcing(setup, ::Type{FT})

Return the external (large-scale) forcing model of `setup`, e.g. an
`ISDACForcing` or `ExternalDrivenTVForcing`.

Defaults to `nothing`, in which case the model construction layer falls back to
the `external_forcing` config key.
"""
external_forcing(setup, ::Type{FT}) where {FT} = nothing

"""
    insolation_model(setup)

Return the insolation model of `setup`, e.g. an `ExternalTVInsolation` or
`RCEMIPIIInsolation`.

Defaults to `nothing`, in which case the `insolation` config key is used.
"""
insolation_model(setup) = nothing

"""
    zonally_symmetric_temperature(coordinates, surface_temp_params, t)

Return the default analytic surface temperature [K]: a steady, zonally
symmetric aquaplanet SST.

On the sphere (`LatLongZPoint` coordinates) the profile is a Gaussian in
latitude, `271 + 29 exp(-φ² / (2 · 26²))` with `φ` in degrees, reduced by a
6.5 K/km lapse rate over the surface elevation. Every other geometry gets a
constant 300 K, the tropical value. Both are independent of `t`.

Used as the default of [`surface_temperature_model`](@ref), wrapped in an
`AnalyticTemperature`.
"""
function zonally_symmetric_temperature(coordinates, surface_temp_params, _)
    (; z) = coordinates
    FT = eltype(z)
    return FT(300)
end
function zonally_symmetric_temperature(
    coordinates::Geometry.LatLongZPoint, surface_temp_params, _,
)
    (; lat, z) = coordinates
    FT = eltype(lat)
    return FT(271) + FT(29) * exp(-coordinates.lat^2 / (2 * 26^2)) - FT(6.5e-3) * z
end

"""
    surface_temperature_model(setup)

Return the default `SurfaceConditions.SurfaceTemperature` of `setup`.

Used when `prognostic_surface == "PrescribedSST"` and
[`surface_condition`](@ref) supplies no `temperature`. Unlike the other model
methods, the default is not `nothing` but an `AnalyticTemperature` wrapping
`zonally_symmetric_temperature`.
"""
surface_temperature_model(setup) =
    AnalyticTemperature(zonally_symmetric_temperature)

"""
    prescribed_flow_model(setup, ::Type{FT})

Return the prescribed velocity profile of `setup`, which replaces the
prognostic momentum solution (e.g. `ShipwayHill2012VelocityProfile`).

Defaults to `nothing`, in which case the `prescribed_flow` config key is used.
"""
prescribed_flow_model(setup, ::Type{FT}) where {FT} = nothing

"""
    radiation_model(setup, ::Type{FT})

Return the case-specific radiation model of `setup`, e.g. `RadiationDYCOMS`,
`RadiationTRMM_LBA`, or `RadiationISDAC`.

Defaults to `nothing`. It is also ignored when the `rad` config key is set
explicitly, so a configuration can always override the setup's radiation.
"""
radiation_model(setup, ::Type{FT}) where {FT} = nothing

# ============================================================================
# Layer 2 and helpers — included files
# ============================================================================

include("common/physical_state.jl")
include("common/prognostic_variables.jl")

# ============================================================================
# Glue: initial_state
# ============================================================================

"""
    initial_condition_field(f, space)

Evaluate the pointwise initial-condition closure `f` over the local geometry of
`space`, returning the resulting `Field`.

The closure is broadcast on the device when it is `isbits` after adapting to
the device array type; otherwise it is broadcast on the host and copied over.
Closures that capture host-resident data — such as the interpolant profiles of
the AtmosphericProfilesLibrary setups — take the host path.
"""
function initial_condition_field(f, space)
    local_geometry = Fields.local_geometry_field(space)
    device = ClimaComms.device(space)
    if device isa ClimaComms.AbstractCPUDevice ||
       isbits(Adapt.adapt(ClimaComms.array_type(device), f))
        return f.(local_geometry)
    end
    field_host = f.(Adapt.adapt(Array, local_geometry))
    field = Fields.Field(eltype(field_host), space)
    copyto!(parent(field), parent(field_host))
    return field
end

"""
    initial_state(setup, params, atmos_model, center_space, face_space)

Construct the prognostic state vector `Y` (a `Fields.FieldVector`) for `setup`.

Two layers, applied pointwise at every grid point:

 1. `center_initial_condition` and [`face_initial_condition`](@ref) give the
    physical state — thermodynamic and kinematic variables, with no knowledge
    of the model configuration.
 2. `center_prognostic_variables` and `face_prognostic_variables` convert it
    into the prognostic variables that `atmos_model` requires.

Surface prognostic variables are added only for a `SlabOceanTemperature`
surface. File-based setups then overwrite fields through
[`overwrite_initial_state!`](@ref), which the caller invokes separately.

# Arguments

  - `setup`: A setup instance, e.g. `Bomex`, `Rico`, or `ForcingFromFile`.
  - `params`: The ClimaAtmos parameter set.
  - `atmos_model`: The `AtmosModel`, whose component models select the prognostic
    variables.
  - `center_space`: The center extruded finite-difference space.
  - `face_space`: The face extruded finite-difference space.
"""
function initial_state(
    setup,
    params,
    atmos_model,
    center_space,
    face_space,
)
    center_ic(lg) = center_prognostic_variables(
        center_initial_condition(setup, lg, params), lg, params, atmos_model,
    )
    face_ic(lg) = face_prognostic_variables(
        face_initial_condition(setup, lg, params), lg, atmos_model,
    )
    surface_space = Fields.level(face_space, Fields.half)

    return Fields.FieldVector(;
        c = initial_condition_field(center_ic, center_space),
        f = initial_condition_field(face_ic, face_space),
        surface_kwargs(surface_space, atmos_model.surface.temperature)...,
    )
end

# ============================================================================
# Setup implementations
# ============================================================================

include("Bomex.jl")
include("Rico.jl")
include("DecayingProfile.jl")
include("DryBaroclinicWave.jl")
include("Soares.jl")
include("GABLS.jl")
include("GATE_III.jl")
include("DYCOMS.jl")
include("TRMM_LBA.jl")
include("ISDAC.jl")
include("Larcform1.jl")
include("IsothermalProfile.jl")
include("ConstantBuoyancyFrequencyProfile.jl")
include("DryDensityCurrentProfile.jl")
include("RisingThermalBubbleProfile.jl")
include("MoistAdiabaticProfileEDMFX.jl")
include("SimplePlume.jl")
include("MoistBaroclinicWave.jl")
include("RCEMIPIIProfile.jl")
include("PrecipitatingColumn.jl")
include("ShipwayHill2012.jl")

# File-based setups (depend on common/overwrite_from_file.jl)
include("common/overwrite_from_file.jl")
include("ForcingFromFile.jl")
include("MoistFromFile.jl")
include("WeatherModel.jl")
include("AMIPFromERA5.jl")

end # module
