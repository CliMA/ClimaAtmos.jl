# ============================================================================
# Physical state constructor
# ============================================================================

"""
    physical_state(;
        T, p = NaN, ρ = NaN, u = 0, v = 0, q_tot = 0, q_liq = 0, q_ice = 0,
        tke = 0, draft_area = 0, q_rai = 0, q_sno = 0, n_liq = 0, n_rai = 0,
        n_ice = 0, q_rim = 0, b_rim = 0, q_gas_A = 0,
    )

Construct the physical state at one grid point.

The return value of every setup's `center_initial_condition`: the
thermodynamic and kinematic state, with no knowledge of the model
configuration. The assembly layer in `prognostic_variables.jl` selects from it
the prognostic variables a given `AtmosModel` needs, so a setup may set fields
that the model ignores. The keyword list is closed — an unrecognized name is a
method error rather than a silently dropped field.

`p` and `ρ` default to `NaN` sentinels rather than `nothing` so that every field
has the same concrete float type; `air_density` fills in whichever was left
unset. Placeholder states with `T = NaN` (used by setups that overwrite the
state from a file) skip validation.

# Keyword Arguments

  - `T`: Temperature, required [K].
  - `p = NaN`: Pressure [Pa]. At least one of `p` and `ρ` is required.
  - `ρ = NaN`: Density [kg/m³].
  - `u`, `v`: Zonal and meridional velocity [m/s].
  - `q_tot`, `q_liq`, `q_ice`: Total, cloud liquid, and cloud ice specific
    humidities [kg/kg].
  - `tke`: Specific turbulent kinetic energy [m²/s²].
  - `draft_area`: Total EDMF draft area fraction, split evenly across the
    subdomains [-].
  - `q_rai`, `q_sno`: Rain and snow specific humidities [kg/kg].
  - `n_liq`, `n_rai`: Cloud droplet and raindrop number concentrations, for
    two-moment microphysics [1/kg].
  - `n_ice`, `q_rim`, `b_rim`: Ice number concentration [1/kg], rime specific
    content [kg/kg], and rime specific volume [m³/kg], for P3 microphysics.
  - `q_gas_A`: Passive gas tracer specific concentration [kg/kg].

# Examples

```julia
state = physical_state(; T = 300.0, p = 101500.0, q_tot = 0.017)
```
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
# Column profiles (shared by the file- and GCM-driven ForcingFromFile cases)
# ============================================================================

"""
    ColumnProfiles{F}

Vertical interpolators of the five variables needed to initialize a column,
shared by the setups that read their initial profiles from a file.

# Fields

  - `T`: Temperature profile `z -> T` [K].
  - `u`, `v`: Zonal and meridional velocity profiles [m/s].
  - `q_tot`: Total specific humidity profile [kg/kg].
  - `ρ`: Density profile [kg/m³].
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

Build `ColumnProfiles` from the height vector `z` [m] and the
corresponding value vectors, interpolating linearly in height and extrapolating
flat beyond the data.
"""
function ColumnProfiles(z, T, u, v, q_tot, ρ)
    interp(vals) = Intp.extrapolate(
        Intp.interpolate((z,), vals, Intp.Gridded(Intp.Linear())),
        Intp.Flat(),
    )
    return ColumnProfiles(interp(T), interp(u), interp(v), interp(q_tot), interp(ρ))
end

"""
    column_profiles_ic(profiles::ColumnProfiles, local_geometry)

Evaluate `profiles` at the height of `local_geometry` and return the resulting
[`physical_state`](@ref).

The shared `center_initial_condition` body of the file-initialized column
setups.
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
    Union{Function, APL.AbstractProfile, Intp.Extrapolation, CI1D.Interpolate1D}

"""
    column_indefinite_integral(f, ϕ₀, zspan; nelems = 1000)

Integrate `ϕ' = f(ϕ, z)` upward from `ϕ(first(zspan)) = ϕ₀`, and return the
solution as a callable profile of height.

The integral is computed with `Operators.column_integral_indefinite!` on a
dedicated column of `nelems` elements, then wrapped in a
`ClimaInterpolations.Interpolation1D.Interpolate1D` with linear interpolation
and flat extrapolation. The column is built on the host, so the returned
profile holds host arrays and must be evaluated on the host.
"""
function column_indefinite_integral(
    f::Function,
    ϕ₀::FT,
    zspan::Tuple{FT, FT};
    nelems = 1000,
) where {FT <: Real}
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(first(zspan)),
        Geometry.ZPoint(last(zspan));
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain; nelems)
    context = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    fspace = Spaces.FaceFiniteDifferenceSpace(z_topology)
    ᶠintegral = Fields.Field(FT, fspace)
    Operators.column_integral_indefinite!(f, ᶠintegral, ϕ₀)
    zdata = copy(vec(parent(Fields.coordinate_field(fspace).z)))
    fdata = copy(vec(parent(ᶠintegral)))
    return CI1D.Interpolate1D(
        zdata, fdata;
        interpolationorder = CI1D.Linear(),
        extrapolationorder = CI1D.Flat(),
    )
end

"""
    ρ_from_profile(thermo_params, p, z, T, θ, q_tot)

Compute the air density [kg/m³] at pressure `p` [Pa] and height `z` [m], given
either the temperature profile `T(z)` [K] or the liquid-ice potential
temperature profile `θ(z)` [K], and optionally the total specific humidity
profile `q_tot(z)` [kg/kg].

Exactly one of `T` and `θ` must be given; the other must be `nothing`. Passing
neither or both is an error. Any condensate is assumed absent, so the moist
density uses `q_tot` alone.
"""
ρ_from_profile(_, _, _, ::Nothing, ::Nothing, _) = error("Either T or θ must be specified")
ρ_from_profile(_, _, _, _::FunctionOrSpline, _::FunctionOrSpline, _) =
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
    return TD.air_density(thermo_params, FT(T(z)), p, FT(q_tot(z)), FT(0), FT(0))
end
function ρ_from_profile(
    thermo_params, p, z, ::Nothing, θ::FunctionOrSpline, q_tot::FunctionOrSpline,
)
    FT = eltype(thermo_params)
    q = FT(q_tot(z))
    T_val = TD.air_temperature(thermo_params, TD.pθ_li(), p, FT(θ(z)), q)
    return TD.air_density(thermo_params, T_val, p, q, FT(0), FT(0))
end

"""
    hydrostatic_pressure_profile(; thermo_params, p_0, [T, θ, q_tot, z_max])

Solve the hydrostatic balance `p'(z) = -g ρ(z)` on `z ∈ [0, z_max]` from the
surface pressure `p_0`, and return the pressure as a callable profile of height
[Pa].

# Keyword Arguments

  - `thermo_params`: Thermodynamics parameter set.
  - `p_0`: Pressure at `z = 0` [Pa].
  - `T`, `θ`: Temperature or liquid-ice potential temperature profile [K].
    Exactly one is required.
  - `q_tot = nothing`: Total specific humidity profile [kg/kg]. Taken as zero
    when omitted.
  - `z_max = 30000`: Top of the integration [m]. It should be the highest
    elevation at which the given profiles are valid, since the result is
    extrapolated flat above it.
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

    return column_indefinite_integral(dp_dz, FT(p_0), (FT(0), FT(z_max)))
end
