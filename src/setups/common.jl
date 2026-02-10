"""
    Common helper functions for simulation setups.

These utilities simplify computing prognostic variables from standard
meteorological inputs (temperature, pressure, moisture, velocity).
"""

# ============================================================================
# Physical state constructor
# ============================================================================

"""
    physical_state(; T, p, kwargs...)

Construct a NamedTuple representing the physical state at a grid point.
This is the return type of `center_initial_condition` — it describes the
thermodynamic and kinematic state without any knowledge of the atmos model.

The assembly layer (`prognostic_variables.jl`) converts this into model-specific
prognostic variables.

## Required arguments
- `T`: Temperature (K)
- `p`: Pressure (Pa)

## Optional arguments (default to zero)
- `q_tot`, `q_liq`, `q_ice`: Specific humidities
- `velocity`: Horizontal velocity as a `Geometry.UVVector` or `UVector` (default: nothing → zero)
- `tke`: Turbulent kinetic energy (specific)
- `draft_area`: EDMF draft area fraction
- `w_draft`: EDMF draft vertical velocity
- `q_rai`, `q_sno`: Precipitation specific humidities
- `n_liq`, `n_rai`: Number densities (2-moment microphysics)
- `n_ice`, `q_rim`, `b_rim`: P3 microphysics fields
"""
function physical_state(;
    T,
    p,
    q_tot = zero(T),
    q_liq = zero(T),
    q_ice = zero(T),
    velocity = nothing,
    tke = zero(T),
    draft_area = zero(T),
    w_draft = zero(T),
    q_rai = zero(T),
    q_sno = zero(T),
    n_liq = zero(T),
    n_rai = zero(T),
    n_ice = zero(T),
    q_rim = zero(T),
    b_rim = zero(T),
)
    return (;
        T, p, q_tot, q_liq, q_ice, velocity, tke, draft_area, w_draft,
        q_rai, q_sno, n_liq, n_rai, n_ice, q_rim, b_rim,
    )
end

# ============================================================================
# Perturbation coefficient
# ============================================================================

"""
    perturb_coeff(point)

Return a geometry-dependent perturbation coefficient for the given coordinate
point. Used by setups that apply spatial perturbations to initial profiles.
"""
perturb_coeff(p::Geometry.AbstractPoint{FT}) where {FT} = FT(0)
perturb_coeff(p::Geometry.LatLongZPoint{FT}) where {FT} = sind(p.long)
perturb_coeff(p::Geometry.XZPoint{FT}) where {FT} = sin(p.x)
perturb_coeff(p::Geometry.XYZPoint{FT}) where {FT} = sin(p.x)

# ============================================================================
# Thermodynamic helpers
# ============================================================================

"""
    total_specific_energy(params, T, local_geometry; velocity=nothing, q_tot=0, q_liq=0, q_ice=0)

Compute the specific total energy `e_tot = e_int + e_kin + e_pot` from
temperature, geometry, and optional velocity/moisture fields.

This is the per-unit-mass total energy (multiply by `ρ` to get `ρe_tot`).
"""
function total_specific_energy(
    params,
    T,
    local_geometry;
    velocity = nothing,
    q_tot = zero(T),
    q_liq = zero(T),
    q_ice = zero(T),
)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    grav = CAP.grav(params)
    z = local_geometry.coordinates.z

    e_int = TD.internal_energy(thermo_params, T, q_tot, q_liq, q_ice)
    e_kin = isnothing(velocity) ? zero(FT) : norm_sqr(velocity) / 2
    e_pot = geopotential(grav, z)

    return e_int + e_kin + e_pot
end

"""
    air_density(params, T, p; q_tot=0, q_liq=0, q_ice=0)

Compute air density from temperature, pressure, and optional moisture.
"""
function air_density(
    params,
    T,
    p;
    q_tot = zero(T),
    q_liq = zero(T),
    q_ice = zero(T),
)
    thermo_params = CAP.thermodynamics_params(params)
    return TD.air_density(thermo_params, T, p, q_tot, q_liq, q_ice)
end

"""
    moist_static_energy(params, T, local_geometry; q_tot=0, q_liq=0, q_ice=0)

Compute the moist static energy `h = c_p T + g z + L_v q_tot` (enthalpy + geopotential).

Used for initializing EDMFX subdomain thermodynamic variables.
"""
function moist_static_energy(
    params,
    T,
    local_geometry;
    q_tot = zero(T),
    q_liq = zero(T),
    q_ice = zero(T),
)
    thermo_params = CAP.thermodynamics_params(params)
    grav = CAP.grav(params)
    z = local_geometry.coordinates.z
    return TD.enthalpy(thermo_params, T, q_tot, q_liq, q_ice) +
           geopotential(grav, z)
end
