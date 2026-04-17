module Setups

import ClimaCore.Geometry as Geometry
import ClimaCore: Fields
import Thermodynamics as TD
import Thermodynamics.TemperatureProfiles: DecayingTemperatureProfile, DryAdiabaticProfile
import AtmosphericProfilesLibrary as APL

import ..Parameters as CAP
import ..geopotential
import ..C12, ..C3
import ..background_p_and_T, ..background_u

# File-based IC infrastructure (overwrite_from_file.jl, GCMDriven.jl, InterpolatedColumnProfile.jl)
import Dates
import ClimaUtilities.SpaceVaryingInputs
import ClimaUtilities.ClimaArtifacts: @clima_artifact
import Interpolations as Intp
import NCDatasets as NC
import Statistics: mean
import ..ᶜinterp, ..ᶠinterp
import ..compute_kinetic
import ..gcm_height, ..gcm_driven_profile_tmean, ..gcm_driven_timeseries
import ..weather_model_data_path
import ..parse_date

# Model types for dispatch (used by prognostic_variables.jl)
import ..DryModel
import ..EquilibriumMicrophysics0M
import ..NonEquilibriumMicrophysics1M
import ..NonEquilibriumMicrophysics2M
import ..NonEquilibriumMicrophysics2MP3
import ..NonEquilibriumMicrophysics
import ..MoistMicrophysics
import ..PrognosticEDMFX
import ..DiagnosticEDMFX
import ..EDOnlyEDMFX
import ..n_mass_flux_subdomains
import ..PrescribedSST
import ..SlabOceanSST
import ..Parameters.ClimaAtmosParameters
import Thermodynamics.Parameters.ThermodynamicsParameters

# Model types returned by setup interface methods
import ..ZonallySymmetricSST
import ..GCMForcing, ..ISDACForcing
import ..GCMDrivenInsolation, ..ExternalTVInsolation
import ..RCEMIPIIInsolation, ..RCEMIPIISST
import ..ExternalColumnInputSST
import ..ShipwayHill2012VelocityProfile
import ..RadiationDYCOMS, ..RadiationTRMM_LBA, ..RadiationISDAC
import ..SurfaceConditions: MoninObukhov, SurfaceState

# ============================================================================
# Layer 1 interface — implemented by each setup
# ============================================================================

"""
    face_initial_condition(setup, local_geometry, params)

Return a NamedTuple of face state variables:
- `w`: Vertical velocity (m/s)
- `w_draft`: EDMF draft vertical velocity (m/s)

## Default
Returns `(; w = 0, w_draft = 0)`.
"""
function face_initial_condition(setup, local_geometry, params)
    FT = eltype(params)
    return (; w = FT(0), w_draft = FT(0))
end

"""
    overwrite_initial_state!(setup, Y, thermo_params)

Optionally overwrite the initial state `Y` after construction. Used by
file-based setups that operate at the field level rather than pointwise.

Default: no-op.
"""
overwrite_initial_state!(setup, Y, thermo_params) = nothing

# ============================================================================
# SCM forcing interface — optional, for single-column setups
# ============================================================================

"""
    subsidence_forcing(setup, FT)

Return a subsidence profile function `z -> w_subsidence`, or `nothing`.
When non-nothing, the returned profile is wrapped in a `Subsidence` struct
by the model construction layer, replacing the `subsidence` config key.
"""
subsidence_forcing(setup, ::Type{FT}) where {FT} = nothing

"""
    large_scale_advection_forcing(setup, FT)

Return `(; prof_dTdt, prof_dqtdt)` as raw APL profile functions, or `nothing`.
The model construction layer wraps these into a `LargeScaleAdvection` struct,
replacing the `ls_adv` config key.
"""
large_scale_advection_forcing(setup, ::Type{FT}) where {FT} = nothing

"""
    coriolis_forcing(setup, FT)

Return `(; prof_ug, prof_vg, coriolis_param)`, or `nothing`.
Replaces the `scm_coriolis` config key.
"""
coriolis_forcing(setup, ::Type{FT}) where {FT} = nothing

"""
    surface_condition(setup, params)

Return the **state-layer** surface configuration for this setup, or `nothing`.

This controls the surface flux parameterization and boundary values (roughness
lengths, exchange coefficients, prescribed T/q, etc.) — i.e. what gets stored
in `p.sfc_setup`. It does *not* control the type of temperature evolution
(prescribed vs. prognostic vs. external), which is set in the model layer via
`AtmosSurface.sfc_temperature` and `AtmosSurface.surface_model`.

When a setup provides a non-`nothing` return, it takes priority over the
`surface_setup` config key / kwarg.

The return value may be:
- A `SurfaceState` (static surface conditions)
- A callable `(surface_coordinates, interior_z, t) -> SurfaceState` (time-varying)
- `nothing` (falls through to config-based surface condition)

Default: `nothing`.
"""
surface_condition(setup, params) = nothing

# ============================================================================
# Model interface — optional, returns model objects directly
# ============================================================================

"""
    external_forcing(setup, ::Type{FT})

Return the external forcing model for this setup, or `nothing`.

Default: `nothing`.
"""
external_forcing(setup, ::Type{FT}) where {FT} = nothing

"""
    insolation_model(setup)

Return the insolation model for this setup, or `nothing`.

Default: `nothing`.
"""
insolation_model(setup) = nothing

"""
    surface_temperature_model(setup)

Return the surface temperature model for this setup.

Default: `ZonallySymmetricSST()`.
"""
surface_temperature_model(setup) = ZonallySymmetricSST()

"""
    prescribed_flow_model(setup, ::Type{FT})

Return the prescribed flow model for this setup, or `nothing`.

Default: `nothing`.
"""
prescribed_flow_model(setup, ::Type{FT}) where {FT} = nothing

"""
    radiation_model(setup, ::Type{FT})

Return the radiation model for this setup, or `nothing`.

Default: `nothing`.
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
    initial_state(setup, params, atmos_model, center_space, face_space)

Construct the full prognostic state vector `Y` (a `Fields.FieldVector`) for the
given setup. Uses the two-layer design:

1. Call `center_initial_condition` / `face_initial_condition` to get
   the physical state at each grid point.
2. Call `center_prognostic_variables` / `face_prognostic_variables`
   to convert the physical state into model-specific prognostic variables.

## Arguments

- `setup`: A setup instance (e.g. `Bomex`, `Rico`, `GCMDriven`)
- `params`: ClimaAtmos parameter set
- `atmos_model`: The atmosphere model (provides model types for dispatch)
- `center_space`: The center finite-difference space
- `face_space`: The face finite-difference space
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
        c = center_ic.(Fields.local_geometry_field(center_space)),
        f = face_ic.(Fields.local_geometry_field(face_space)),
        surface_kwargs(surface_space, atmos_model.surface_model)...,
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
include("GCMDriven.jl")
include("InterpolatedColumnProfile.jl")
include("MoistFromFile.jl")
include("WeatherModel.jl")
include("AMIPFromERA5.jl")

end # module
