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
  - `q_gas_A`: Passive gas tracer concentration (default 0)
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
    q_gas_A = zero(T),
)
    # Validate only for real states (T is finite). Placeholder states with
    # T = NaN (e.g. WeatherModel, AMIPFromERA5) skip validation because their
    # prognostic fields are overwritten after construction.
    !isnan(T) && isnan(p) && isnan(ρ) &&
        error("physical_state requires at least one of `p` or `ρ`")
    return (;
        T, p, ρ, u, v, q_tot, q_liq, q_ice, tke, draft_area,
        q_rai, q_sno, n_liq, n_rai, n_ice, q_rim, b_rim, q_gas_A,
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

Wrap a column field so it can be evaluated at any height `z` by linear
interpolation between grid levels, with flat extrapolation outside the column
bounds. Backed by `ClimaInterpolations.Interpolation1D.Interpolate1D` over the
column knot and value arrays on the field device.

For IC generation on a GPU space, call [`interpolate_column_profile_to_space`](@ref)
once and pass the result through `center_initial_condition(...; p_at_point=...)` rather
than calling the profile directly inside a device broadcast.
"""
struct ColumnInterpolatableField{I <: CI1D.Interpolate1D}
    itp::I
end

"""Materialize matching 1D knot/value arrays on the field device for `Interpolate1D`."""
function _device_column_vectors(f::Fields.ColumnField)
    zdata = vec(parent(Fields.coordinate_field(f).z))
    fdata = vec(parent(f))
    @assert length(zdata) == length(fdata)
    typeof(zdata) === typeof(fdata) && return zdata, fdata
    DA = ClimaComms.array_type(ClimaComms.device(parent(f)))
    FT = eltype(fdata)
    zout = DA{FT}(undef, length(zdata))
    fout = DA{FT}(undef, length(fdata))
    copyto!(zout, zdata)
    copyto!(fout, fdata)
    return zout, fout
end

function ColumnInterpolatableField(f::Fields.ColumnField)
    zdata, fdata = _device_column_vectors(f)
    itp = CI1D.Interpolate1D(
        zdata,
        fdata;
        interpolationorder = CI1D.Linear(),
        extrapolationorder = CI1D.Flat(),
    )
    return ColumnInterpolatableField(itp)
end
@inline (f::ColumnInterpolatableField)(z) = f.itp(convert(eltype(f.itp.xsource), z))

"""
    evaluate_pressure(p_profile, z; p_at_point = nothing)

Return a pre-interpolated pressure `p_at_point` when provided, otherwise evaluate
`p_profile` at height `z`.
"""
@inline evaluate_pressure(p_profile, z; p_at_point = nothing) =
    isnothing(p_at_point) ? p_profile(z) : p_at_point

"""
    interpolate_column_profile_to_space(cif::ColumnInterpolatableField, space)

Interpolate a column profile onto `space` using `ClimaInterpolations.interpolate1d!`.
Runs as a device-native column operation (no scalar indexing of GPU arrays).
"""
function interpolate_column_profile_to_space(
    cif::ColumnInterpolatableField,
    space::Spaces.AbstractSpace,
)
    ᶜz = Fields.coordinate_field(space).z
    FT = Spaces.undertype(space)
    ᶜout = Fields.Field(FT, space)
    CI1D.interpolate1d!(
        vec(parent(ᶜout)),
        vec(cif.itp.xsource),
        vec(parent(ᶜz)),
        vec(cif.itp.fsource),
        CI1D.Linear(),
        CI1D.Flat(),
    )
    return ᶜout
end

"""
    preinterpolated_hydrostatic_pressure(setup, center_space)

If `setup` carries a hydrostatic [`ColumnInterpolatableField`](@ref) pressure
profile, interpolate it onto `center_space` for use in `initial_state`.
"""
function preinterpolated_hydrostatic_pressure(setup, center_space)
    hasproperty(setup, :profiles) || return nothing
    profiles = getproperty(setup, :profiles)
    hasproperty(profiles, :p) || return nothing
    p = getproperty(profiles, :p)
    p isa ColumnInterpolatableField || return nothing
    return interpolate_column_profile_to_space(p, center_space)
end

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
    ρ_from_profile(thermo_params, p, z, T, θ, q_tot)

Compute air density at pressure `p` and height `z`, given either temperature
`T(z)` or potential temperature `θ(z)` and, optionally, total specific
humidity `q_tot(z)`. Exactly one of `T` or `θ` must be provided.
"""
ρ_from_profile(thermo_params, p, z, ::Nothing, ::Nothing, _) =
    error("Either T or θ must be specified")
ρ_from_profile(thermo_params, p, z, T::FunctionOrSpline, θ::FunctionOrSpline, _) =
    error("Only one of T and θ can be specified")
ρ_from_profile(thermo_params, p, z, T::FunctionOrSpline, ::Nothing, ::Nothing) =
    TD.air_density(thermo_params, oftype(p, T(z)), p)
function ρ_from_profile(thermo_params, p, z, ::Nothing, θ::FunctionOrSpline, ::Nothing)
    T_val = TD.air_temperature(thermo_params, TD.pθ_li(), p, oftype(p, θ(z)))
    return TD.air_density(thermo_params, T_val, p)
end
function ρ_from_profile(
    thermo_params, p, z, T::FunctionOrSpline, ::Nothing, q_tot::FunctionOrSpline,
)
    FT = eltype(thermo_params)
    return TD.air_density(
        thermo_params, oftype(p, T(z)), p, oftype(p, q_tot(z)), FT(0), FT(0),
    )
end
function ρ_from_profile(
    thermo_params, p, z, ::Nothing, θ::FunctionOrSpline, q_tot::FunctionOrSpline,
)
    FT = eltype(thermo_params)
    q = oftype(p, q_tot(z))
    T_val = TD.air_temperature(thermo_params, TD.pθ_li(), p, oftype(p, θ(z)), q)
    return TD.air_density(thermo_params, T_val, p, q, FT(0), FT(0))
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

    dp_dz(p, z) = -grav * ρ_from_profile(thermo_params, p, z, T, θ, q_tot)

    return column_indefinite_integral(dp_dz, p_0, (FT(0), FT(z_max)))
end
