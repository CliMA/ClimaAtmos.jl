"""
    Setups

Module defining the interface for simulation setups in ClimaAtmos. A "setup"
encapsulates the initial conditions for a specific simulation case (e.g., Bomex,
GABLS, baroclinic wave).

# Architecture

Setups use a **two-layer design**:

**Layer 1 — Setup implementations** (one file per case):
- `center_initial_condition(setup, local_geometry, params)` → physical state
- `face_initial_condition(setup, local_geometry, params)` → `(; w)` or `(; w, w_draft)`
- `surface_condition(setup, params)` → surface boundary data
- Setups know nothing about the atmos model

**Layer 2 — Prognostic variable assembly** (`prognostic_variables.jl`):
- `center_prognostic_variables(ps, local_geometry, params, atmos_model)` → prognostic NamedTuple
- `face_prognostic_variables(fs, local_geometry, atmos_model)` → face prognostic NamedTuple
- Dispatches on moisture, microphysics, and turbconv model types
"""
module Setups

import ClimaCore.Geometry as Geometry
import ClimaCore: Fields
import Thermodynamics as TD
import AtmosphericProfilesLibrary as APL
import LinearAlgebra: norm_sqr

import ..Parameters as CAP
import ..geopotential
import ..C12, ..C3

# File-based IC infrastructure (overwrite_from_file.jl, GCMDriven.jl)
import ClimaUtilities.SpaceVaryingInputs
import NCDatasets as NC
import Statistics: mean
import ..ᶜinterp, ..ᶠinterp
import ..compute_kinetic
import ..gcm_height, ..gcm_driven_profile_tmean, ..gcm_driven_timeseries

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

abstract type AbstractSetup end

# ============================================================================
# Layer 1 interface — implemented by each setup
# ============================================================================

"""
    center_initial_condition(setup::AbstractSetup, local_geometry, params)

Return a `physical_state` NamedTuple describing the thermodynamic and kinematic
state at a cell center. The assembly layer (`prognostic_variables.jl`) converts
this into model-specific prognostic variables.

Setups should call `physical_state(; T, p, ...)` to construct the return value.

## Arguments

- `setup`: An instance of an `AbstractSetup` subtype
- `local_geometry`: The local geometry at the grid point
- `params`: ClimaAtmos parameter set
"""
function center_initial_condition end

"""
    face_initial_condition(setup::AbstractSetup, local_geometry, params)

Return a NamedTuple of face state variables. At minimum, must include:
- `w`: Vertical velocity (m/s)

Optionally:
- `w_draft`: EDMF draft vertical velocity (m/s)

## Default
Returns `(; w = zero(eltype(params)))` (zero vertical velocity).
"""
function face_initial_condition(::AbstractSetup, local_geometry, params)
    FT = eltype(params)
    return (; w = FT(0))
end

"""
    surface_condition(::AbstractSetup, params)

Return surface/boundary condition data for the given setup type.
"""
function surface_condition end

"""
    overwrite_initial_state!(::AbstractSetup, Y, thermo_params)

Optionally overwrite the initial state `Y` after construction. Used by
file-based setups that operate at the field level rather than pointwise.

Default: no-op.
"""
overwrite_initial_state!(::AbstractSetup, Y, thermo_params) = nothing

# ============================================================================
# SCM forcing interface — optional, for single-column setups
# ============================================================================

"""
    subsidence_forcing(::AbstractSetup, FT)

Return a subsidence profile function `z -> w_subsidence`, or `nothing`.
When non-nothing, the returned profile is wrapped in a `Subsidence` struct
by the model construction layer, replacing the `subsidence` config key.
"""
subsidence_forcing(::AbstractSetup, ::Type{FT}) where {FT} = nothing

"""
    large_scale_advection_forcing(::AbstractSetup, FT)

Return `(; prof_dTdt, prof_dqtdt)` as raw APL profile functions, or `nothing`.
The model construction layer wraps these into a `LargeScaleAdvection` struct,
replacing the `ls_adv` config key.
"""
large_scale_advection_forcing(::AbstractSetup, ::Type{FT}) where {FT} = nothing

"""
    coriolis_forcing(::AbstractSetup, FT)

Return `(; prof_ug, prof_vg, coriolis_param)`, or `nothing`.
The model construction layer wraps these into a `SCMCoriolis` struct,
replacing the `scm_coriolis` config key.
"""
coriolis_forcing(::AbstractSetup, ::Type{FT}) where {FT} = nothing

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
    initial_state(setup::AbstractSetup, params, atmos_model, center_space, face_space)

Construct the full prognostic state vector `Y` (a `Fields.FieldVector`) for the
given setup. Uses the two-layer design:

1. Call Layer 1 (`center_initial_condition` / `face_initial_condition`) to get
   the physical state at each grid point.
2. Call Layer 2 (`center_prognostic_variables` / `face_prognostic_variables`)
   to convert the physical state into model-specific prognostic variables.

## Arguments

- `setup`: An instance of an `AbstractSetup` subtype
- `params`: ClimaAtmos parameter set
- `atmos_model`: The atmosphere model (provides model types for dispatch)
- `center_space`: The center finite-difference space
- `face_space`: The face finite-difference space
"""
function initial_state(
    setup::AbstractSetup,
    params,
    atmos_model,
    center_space,
    face_space,
)
    center_ic =
        lg -> center_prognostic_variables(
            center_initial_condition(setup, lg, params),
            lg,
            params,
            atmos_model,
        )
    face_ic =
        lg -> face_prognostic_variables(
            face_initial_condition(setup, lg, params),
            lg,
            atmos_model,
        )

    Fields.FieldVector(;
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

end # module
