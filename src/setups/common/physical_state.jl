# ============================================================================
# Physical state constructor
# ============================================================================

"""
    physical_state(; T, p=NaN, ρ=NaN, kwargs...)

Construct a NamedTuple representing the physical state at a grid point.
This is the return type of `center_initial_condition` — it describes the
thermodynamic and kinematic state without any knowledge of the atmos model.

The assembly layer (`prognostic_variables.jl`) converts this into model-specific
prognostic variables.

## Required arguments
- `T`: Temperature (K)
- At least one of `p` (pressure, Pa) or `ρ` (density, kg/m³). If only one is
  provided, the assembly layer computes the other via `Thermodynamics`.

## Optional arguments (default to zero)
- `u`, `v`: Zonal and meridional velocity (m/s)
- `q_tot`, `q_liq`, `q_ice`: Specific humidities
- `tke`: Turbulent kinetic energy (specific)
- `draft_area`: EDMF draft area fraction
- `q_rai`, `q_sno`: Precipitation specific humidities
- `n_liq`, `n_rai`: Number densities (2-moment microphysics)
- `n_ice`, `q_rim`, `b_rim`: P3 microphysics fields
"""
function physical_state(;
    T,
    # Use NaN sentinels instead of `nothing` so that every field has the same
    # concrete float type.  Mixed Nothing/Float breaks ClimaCore broadcast
    # inference and GPU scalar-indexing fallbacks.
    p = oftype(T, NaN),
    ρ = oftype(T, NaN),
    u = zero(T),
    v = zero(T),
    q_tot = zero(T),
    q_liq = zero(T),
    q_ice = zero(T),
    tke = zero(T),
    draft_area = zero(T),
    q_rai = zero(T),
    q_sno = zero(T),
    n_liq = zero(T),
    n_rai = zero(T),
    n_ice = zero(T),
    q_rim = zero(T),
    b_rim = zero(T),
)
    # Validate only for real states (T is finite). Placeholder states with
    # T = NaN (e.g. WeatherModel, AMIPFromERA5) skip validation because their
    # prognostic fields are overwritten after construction.
    !isnan(T) && isnan(p) && isnan(ρ) &&
        error("physical_state requires at least one of `p` or `ρ`")
    return (;
        T, p, ρ, u, v, q_tot, q_liq, q_ice, tke, draft_area,
        q_rai, q_sno, n_liq, n_rai, n_ice, q_rim, b_rim,
    )
end

# ============================================================================
# Column profiles (shared by GCMDriven, InterpolatedColumnProfile)
# ============================================================================

"""
    ColumnProfiles{F}

A set of 1D vertical interpolators for the five standard atmospheric
variables: temperature, winds, specific humidity, and density. Shared by
setups that initialize from externally-read vertical profiles.
"""
struct ColumnProfiles{F}
    T::F
    u::F
    v::F
    q_tot::F
    ρ::F
end

"""
    ColumnProfiles(z, T, u, v, q_tot, ρ)

Build `ColumnProfiles` from height vector `z` and corresponding value vectors.
Uses linear interpolation with flat extrapolation.
"""
function ColumnProfiles(z, T, u, v, q_tot, ρ)
    interp(vals) = Intp.extrapolate(
        Intp.interpolate((z,), vals, Intp.Gridded(Intp.Linear())),
        Intp.Flat(),
    )
    return ColumnProfiles(interp(T), interp(u), interp(v), interp(q_tot), interp(ρ))
end

"""
    center_initial_condition(profiles::ColumnProfiles, local_geometry)

Evaluate column profiles at the grid point height and return a `physical_state`.
"""
function column_profiles_ic(profiles::ColumnProfiles, local_geometry)
    (; z) = local_geometry.coordinates
    FT = typeof(z)
    return physical_state(;
        T = FT(profiles.T(z)),
        ρ = FT(profiles.ρ(z)),
        q_tot = FT(profiles.q_tot(z)),
        u = FT(profiles.u(z)),
        v = FT(profiles.v(z)),
        tke = FT(0),
    )
end

# ============================================================================
# Hydrostatic pressure solver
# ============================================================================

import ClimaComms
import ClimaCore.Domains as Domains
import ClimaCore.Meshes as Meshes
import ClimaCore.Operators as Operators
import ClimaCore.Topologies as Topologies
import ClimaCore.Spaces as Spaces

const FunctionOrSpline =
    Union{Function, APL.AbstractProfile, Intp.Extrapolation}

"""
    ColumnInterpolatableField(::Fields.ColumnField)

A column field object that can be interpolated
in the z-coordinate. For example:

!!! warn
    This function allocates and is not GPU-compatible
    so please avoid using this inside `step!` only use
    this for initialization.
"""
struct ColumnInterpolatableField{F, D}
    f::F
    data::D
    function ColumnInterpolatableField(f::Fields.ColumnField)
        zdata = vec(parent(Fields.Fields.coordinate_field(f).z))
        fdata = vec(parent(f))
        data = Intp.extrapolate(
            Intp.interpolate((zdata,), fdata, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
        return new{typeof(f), typeof(data)}(f, data)
    end
end
(f::ColumnInterpolatableField)(z) = Spaces.undertype(axes(f.f))(f.data(z))

"""
    column_indefinite_integral(f, ϕ₀, zspan; nelems = 100)

The column integral, returned as an interpolate-able field.
"""
function column_indefinite_integral(
    f::Function,
    ϕ₀::FT,
    zspan::Tuple{FT, FT};
    nelems = 100, # sets resolution for integration
) where {FT <: Real}
    # --- Make a space for integration:
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(first(zspan)),
        Geometry.ZPoint(last(zspan));
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain; nelems)
    context = ClimaComms.SingletonCommsContext()
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    cspace = Spaces.CenterFiniteDifferenceSpace(z_topology)
    fspace = Spaces.FaceFiniteDifferenceSpace(z_topology)
    # ---
    zc = Fields.coordinate_field(cspace)
    ᶠintegral = Fields.Field(FT, fspace)
    Operators.column_integral_indefinite!(f, ᶠintegral, ϕ₀)
    return ColumnInterpolatableField(ᶠintegral)
end

"""
    hydrostatic_pressure_profile(; thermo_params, p_0, [T, θ, q_tot, z_max])

Solves the initial value problem `p'(z) = -g * ρ(z)` for all `z ∈ [0, z_max]`,
given `p(0)`, either `T(z)` or `θ(z)`, and optionally also `q_tot(z)`. If
`q_tot(z)` is not given, it is assumed to be 0. If `z_max` is not given, it is
assumed to be 30 km. Note that `z_max` should be the maximum elevation to which
the specified profiles T(z), θ(z), and/or q_tot(z) are valid.
"""
function hydrostatic_pressure_profile(;
    thermo_params,
    p_0,
    T = nothing,
    θ = nothing,
    q_tot = nothing,
    z_max = 30000,
)
    FT = eltype(thermo_params)
    grav = TD.Parameters.grav(thermo_params)

    # Compute air density from (p, z) using either T(z) or θ(z), with optional q_tot(z)
    function ρ_from_profile(p, z, ::Nothing, ::Nothing, _)
        error("Either T or θ must be specified")
    end
    function ρ_from_profile(p, z, T::FunctionOrSpline, θ::FunctionOrSpline, _)
        error("Only one of T and θ can be specified")
    end
    function ρ_from_profile(p, z, T::FunctionOrSpline, ::Nothing, ::Nothing)
        TD.air_density(thermo_params, oftype(p, T(z)), p)
    end
    function ρ_from_profile(p, z, ::Nothing, θ::FunctionOrSpline, ::Nothing)
        T_val = TD.air_temperature(thermo_params, TD.pθ_li(), p, oftype(p, θ(z)))
        TD.air_density(thermo_params, T_val, p)
    end
    function ρ_from_profile(p, z, T::FunctionOrSpline, ::Nothing, q_tot::FunctionOrSpline)
        TD.air_density(thermo_params, oftype(p, T(z)), p, oftype(p, q_tot(z)), FT(0), FT(0))
    end
    function ρ_from_profile(p, z, ::Nothing, θ::FunctionOrSpline, q_tot::FunctionOrSpline)
        q = oftype(p, q_tot(z))
        T_val = TD.air_temperature(thermo_params, TD.pθ_li(), p, oftype(p, θ(z)), q)
        TD.air_density(thermo_params, T_val, p, q, FT(0), FT(0))
    end
    dp_dz(p, z) = -grav * ρ_from_profile(p, z, T, θ, q_tot)

    return column_indefinite_integral(dp_dz, p_0, (FT(0), FT(z_max)))
end
