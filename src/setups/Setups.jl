module Setups

import ClimaCore.Geometry as Geometry
import ClimaCore: Fields
import Thermodynamics as TD
import AtmosphericProfilesLibrary as APL

import ..Parameters as CAP
import ..geopotential
import ..C12, ..C3

# File-based IC infrastructure (overwrite_from_file.jl, GCMDriven.jl, InterpolatedColumnProfile.jl)
import Dates
import ClimaUtilities.SpaceVaryingInputs
import ClimaUtilities.ClimaArtifacts: @clima_artifact
import NCDatasets as NC
import Statistics: mean
import ..ᶜinterp, ..ᶠinterp
import ..compute_kinetic
import ..gcm_height, ..gcm_driven_profile_tmean, ..gcm_driven_timeseries
import ..weather_model_data_path
import ..parse_date

# Model types for dispatch (used by prognostic_variables.jl)
import ..DryModel
import ..EquilMoistModel
import ..NonEquilMoistModel
import ..NoPrecipitation
import ..Microphysics0Moment
import ..Microphysics1Moment
import ..Microphysics2Moment
import ..Microphysics2MomentP3
import ..PrognosticEDMFX
import ..DiagnosticEDMFX
import ..EDOnlyEDMFX
import ..n_mass_flux_subdomains
import ..PrescribedSST
import ..SlabOceanSST

# ============================================================================
# Layer 1 interface — implemented by each setup
# ============================================================================

"""
    face_initial_condition(setup, local_geometry, params)

Return a NamedTuple of face state variables. At minimum, must include:
- `w`: Vertical velocity (m/s)

Optionally:
- `w_draft`: EDMF draft vertical velocity (m/s)

## Default
Returns `(; w = zero(eltype(params)))` (zero vertical velocity).
"""
function face_initial_condition(setup, local_geometry, params)
    FT = eltype(params)
    return (; w = FT(0))
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
The model construction layer wraps these into a `SCMCoriolis` struct,
replacing the `scm_coriolis` config key.
"""
coriolis_forcing(setup, ::Type{FT}) where {FT} = nothing

# ============================================================================
# Layer 2 and helpers — included files
# ============================================================================

include("common.jl")
include("hydrostatic.jl")
include("prognostic_variables.jl")

# ============================================================================
# Glue: initial_state
# ============================================================================

"""
    initial_state(setup, params, atmos_model, center_space, face_space)

Construct the full prognostic state vector `Y` (a `Fields.FieldVector`) for the
given setup. Uses the two-layer design:

1. Call Layer 1 (`center_initial_condition` / `face_initial_condition`) to get
   the physical state at each grid point.
2. Call Layer 2 (`center_prognostic_variables` / `face_prognostic_variables`)
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

    return Fields.FieldVector(;
        c = center_ic.(Fields.local_geometry_field(center_space)),
        f = face_ic.(Fields.local_geometry_field(face_space)),
        atmos_surface_field(
            Fields.level(face_space, Fields.half),
            atmos_model.surface_model,
        )...,
    )
end

# ============================================================================
# Setup implementations
# ============================================================================

include("Bomex.jl")
include("Rico.jl")

# File-based IC infrastructure and setups
include("overwrite_from_file.jl")
include("GCMDriven.jl")
include("InterpolatedColumnProfile.jl")
include("MoistFromFile.jl")
include("WeatherModel.jl")
include("AMIPFromERA5.jl")

end # module
