module RRTMGPInterface

import ..AbstractCloudInRadiation, ..InteractiveCloudInRadiation

import NCDatasets as NC
using RRTMGP
import RRTMGP.AtmosphericStates as AS
using ClimaCore: DataLayouts, Spaces, Fields
import Adapt
import ClimaComms
using NVTX
using Random
# TODO: Move this file to RRTMGP.jl, once the interface has been settled.
# It will be faster to do interface development in the same repo as experiment
# development, but, since this is just a user-friendly wrapper for RRTMGP.jl, we
# should move it there eventually.

abstract type AbstractRRTMGPMode end
@kwdef struct GrayRadiation <: AbstractRRTMGPMode
    add_isothermal_boundary_layer::Bool = true
    deep_atmosphere::Bool = true
end
@kwdef struct ClearSkyRadiation <: AbstractRRTMGPMode
    idealized_h2o::Bool = false
    add_isothermal_boundary_layer::Bool = true
    aerosol_radiation::Bool = false
    deep_atmosphere::Bool = true
end
@kwdef struct AllSkyRadiation{ACR <: Union{Nothing, AbstractCloudInRadiation}} <:
              AbstractRRTMGPMode
    idealized_h2o::Bool = false
    idealized_clouds::Bool = false
    cloud::ACR = InteractiveCloudInRadiation()
    add_isothermal_boundary_layer::Bool = true
    aerosol_radiation::Bool = false
    """
    Reset the RNG seed before calling RRTMGP to a known value (the timestep number).
    When modeling cloud optics, RRTMGP uses a random number generator.
    Resetting the seed every time RRTMGP is called to a deterministic value ensures that
    the simulation is fully reproducible and can be restarted in a reproducible way.
    Disable this option when running production runs.
    """
    reset_rng_seed::Bool = false
    deep_atmosphere::Bool = true
end
@kwdef struct AllSkyRadiationWithClearSkyDiagnostics{
    ACR <: Union{Nothing, AbstractCloudInRadiation},
} <: AbstractRRTMGPMode
    idealized_h2o::Bool = false
    idealized_clouds::Bool = false
    cloud::ACR = InteractiveCloudInRadiation()
    add_isothermal_boundary_layer::Bool = true
    aerosol_radiation::Bool = false
    """
    Reset the RNG seed before calling RRTMGP to a known value (the timestep number).
    When modeling cloud optics, RRTMGP uses a random number generator.
    Resetting the seed every time RRTMGP is called to a deterministic value ensures that
    the simulation is fully reproducible and can be restarted in a reproducible way.
    Disable this option when running production runs.
    """
    reset_rng_seed::Bool = false
    deep_atmosphere::Bool = true
end

"""
    abstract type AbstractInterpolation

A method for obtaining cell face pressures/temperatures from cell center
pressures/temperatures, or vice versa. The available options are `BestFit`,
`UniformZ`, `UniformP`, `GeometricMean`, and `ArithmeticMean`. `BestFit`
requires z-coordinates to be provided, while the other options do not. If both
cell center and cell face values are provided, `NoInterpolation` should be used.

To get cell face values from cell center values, an `AbstractInterpolation` is
used for interpolation on the interior faces and for extrapolation on the
boundary faces. To get cell center values from cell face values, it is only used
for interpolation.

For `BestFit`, we start by assuming that there is some constant lapse rate
    ∂T(z)/∂z = L.
This tells us that, for some constant T₀,
    T(z) = T₀ + L * z.
Since T(z₁) = T₁ and T(z₂) = T₂, for some z₁ != z₂, we have that
    T₁ = T₀ + L * z₁ and T₂ = T₀ + L * z₂ ==>
    T₂ - T₁ = L * (z₂ - z₁) ==>
    L = (T₂ - T₁) / (z₂ - z₁) ==>
    T₁ = T₀ + (T₂ - T₁) / (z₂ - z₁) * z₁ ==>
    T₀ = (T₁ * z₂ - T₂ * z₁) / (z₂ - z₁) ==>
    T(z) = T₁ + (T₂ - T₁) / (z₂ - z₁) * (z - z₁).
To get p(z), we do different things depending on whether or not T₁ == T₂ (i.e.,
whether or not L == 0).
If T₁ == T₂, we assume hydrostatic equilibrium, so that, for some constant C,
    ∂p(z)/∂z = C * p(z).
This tells us that, for some constant p₀,
    p(z) = p₀ * exp(C * z).
Since p(z₁) = p₁ and p(z₂) = p₂, we have that
    p₁ = p₀ * exp(C * z₁) and p₂ = p₀ * exp(C * z₂) ==>
    p₂ / p₁ = exp(C * (z₂ - z₁)) ==>
    C = log(p₂ / p₁) / (z₂ - z₁) ==>
    p₁ = p₀ * (p₂ / p₁)^(z₁ / (z₂ - z₁)) ==>
    p₀ = p₁ / (p₂ / p₁)^(z₁ / (z₂ - z₁)) ==>
    p(z) = p₁ * (p₂ / p₁)^((z - z₁) / (z₂ - z₁)).
If T₁ != T₂, we assume that p and T are governed by an isentropic process, so
that, for some constants A and B,
    p(z) = A * T(z)^B.
Since p(z₁) = p₁, p(z₂) = p₂, T(z₁) = T₁, and T(z₂) = T₂, we have that
    p₁ = A * T₁^B and p₂ = A * T₂^B ==>
    p₂ / p₁ = (T₂ / T₁)^B ==>
    B = log(p₂ / p₁) / log(T₂ / T₁) ==>
    p₁ = A * (p₂ / p₁)^(log(T₁) / log(T₂ / T₁)) ==>
    A = p₁ / (p₂ / p₁)^(log(T₁) / log(T₂ / T₁)) ==>
    p(z) = p₁ * (p₂ / p₁)^(log(T(z) / T₁) / log(T₂ / T₁)).
So, in conclusion, we have that
    T(z) = T₁ + (T₂ - T₁) / (z₂ - z₁) * (z - z₁) and
    p(z) = T₁ == T₂ ?
           p₁ * (p₂ / p₁)^((z - z₁) / (z₂ - z₁)) :
           p₁ * (p₂ / p₁)^(log(T(z) / T₁) / log(T₂ / T₁)).

`UniformZ`, `UniformP`, and `GeometricMean` are all special cases of `BestFit`
that assume a particular value for z in order to avoid requiring z-coordinates.
For `UniformZ`, we assume that
    z = (z₁ + z₂) / 2.
This tells us that
    T(z) = (T₁ + T₂) / 2 and
    p(z) = T₁ == T₂ ?
        sqrt(p₁ * p₂) :
        p₁ * (p₂ / p₁)^(log(T(z) / T₁) / log(T₂ / T₁)).
For `UniformP`, we assume that T₁ != T₂ and p₁ != p₂, and that
    z = z₁ + (z₂ - z₁) / (T₂ / T₁ - 1) *
        ((T₂ / T₁)^(log((1 + p₂ / p₁) / 2) / log(p₂ / p₁)) - 1).
This tells us that
    T(z) = T₁ * (T₂ / T₁)^(log(p(z) / p₁) / log(p₂ / p₁)) and
    p(z) = (p₁ + p₂) / 2.
For `GeometricMean`, we assume that
    z = z₁ + (z₂ - z₁) / (sqrt(T₂ / T₁) + 1).
This tells us that
    T(z) = sqrt(T₁ * T₂) and p(z) = sqrt(p₁ * p₂).

Finally, `ArithmeticMean` is the simplest possible interpolation method:
    T(z) = (T₁ + T₂) / 2 and
    p(z) = (p₁ + p₂) / 2.
"""
abstract type AbstractInterpolation end
struct NoInterpolation <: AbstractInterpolation end
struct ArithmeticMean <: AbstractInterpolation end
struct GeometricMean <: AbstractInterpolation end
struct UniformZ <: AbstractInterpolation end
struct UniformP <: AbstractInterpolation end
struct BestFit <: AbstractInterpolation end

"""
    abstract type AbstractBottomExtrapolation

A method for obtaining the bottom cell face pressure/temperature from the cell
center pressures/temperatures above it. The available options are
`SameAsInterpolation`, `UseSurfaceTempAtBottom`, and `HydrostaticBottom`.
`SameAsInterpolation` uses the interpolation method to extrapolate the bottom
cell face values, while `UseSurfaceTempAtBottom` and `HydrostaticBottom` provide
alternative methods. `HydrostaticBottom` requires z-coordinates to be provided,
while `UseSurfaceTempAtBottom` does not.

For `UseSurfaceTempAtBottom` and `HydrostaticBottom`, we assume that we have a
dry ideal gas undergoing an isentropic process, so that, for some constant A,
    p(z) = A * T(z)^(cₚ / R).
Since p(z⁺) = p⁺ and T(z⁺) = T⁺, where z⁺ is the z-coordinate of the first cell
center above the bottom cell face, we have that
    p⁺ = A * T⁺^(cₚ / R) ==>
    A = p⁺ / T⁺^(cₚ / R) ==>
    p(z) = p⁺ * (T(z) / T⁺)^(cₚ / R).

For `UseSurfaceTempAtBottom`, we assume that the air at the bottom cell face is
in thermal equilibrium with the surface, whose temperature is Tₛ, so that
    T(z) = Tₛ.

For `HydrostaticBottom`, we assume that the lapse rate in the bottom cell is
    ∂T(z)/∂z = g / cₚ.
This tells us that, for some constant T₀,
    T(z) = T₀ + g / cₚ * z.
Since T(z⁺) = T⁺, we have that
    T⁺ = T₀ + g / cₚ * z⁺ ==>
    T₀ = T⁺ - g / cₚ * z⁺ ==>
    T(z) = T⁺ + g / cₚ * (z - z⁺).
"""
abstract type AbstractBottomExtrapolation end
struct SameAsInterpolation <: AbstractBottomExtrapolation end
struct UseSurfaceTempAtBottom <: AbstractBottomExtrapolation end
struct HydrostaticBottom <: AbstractBottomExtrapolation end

requires_z(::Any) = false
requires_z(::Union{BestFit, HydrostaticBottom}) = true

uniform_z_p(T, p₁, T₁, p₂, T₂) =
    T₁ == T₂ ? sqrt(p₁ * p₂) : p₁ * (p₂ / p₁)^(log(T / T₁) / log(T₂ / T₁))
best_fit_p(T, z, p₁, T₁, z₁, p₂, T₂, z₂) =
    T₁ == T₂ ? p₁ * (p₂ / p₁)^((z - z₁) / (z₂ - z₁)) :
    p₁ * (p₂ / p₁)^(log(T / T₁) / log(T₂ / T₁))

function interp!(::ArithmeticMean, p, T, pꜜ, Tꜜ, pꜛ, Tꜛ)
    @. T = (Tꜜ + Tꜛ) / 2
    @. p = (pꜜ + pꜛ) / 2
end
function interp!(::GeometricMean, p, T, pꜜ, Tꜜ, pꜛ, Tꜛ)
    @. T = sqrt(Tꜜ * Tꜛ)
    @. p = sqrt(pꜜ * pꜛ)
end
function interp!(::UniformZ, p, T, pꜜ, Tꜜ, pꜛ, Tꜛ)
    @. T = (Tꜜ + Tꜛ) / 2
    @. p = uniform_z_p(T, pꜜ, Tꜜ, pꜛ, Tꜛ)
end
function interp!(::UniformP, p, T, pꜜ, Tꜜ, pꜛ, Tꜛ)
    @. p = (pꜜ + pꜛ) / 2
    @. T = Tꜜ * (Tꜛ / Tꜜ)^(log(p / pꜜ) / log(pꜛ / pꜜ)) # assume that pꜜ != pꜛ
end
function interp!(::BestFit, p, T, z, pꜜ, Tꜜ, zꜜ, pꜛ, Tꜛ, zꜛ)
    @. T = Tꜜ + (Tꜛ - Tꜜ) * (z - zꜜ) / (zꜛ - zꜜ)
    @. p = best_fit_p(T, z, pꜜ, Tꜜ, zꜜ, pꜛ, Tꜛ, zꜛ)
end

function extrap!(::ArithmeticMean, p, T, p⁺, T⁺, p⁺⁺, T⁺⁺, Tₛ, params)
    @. T = (3 * T⁺ - T⁺⁺) / 2
    @. p = (3 * p⁺ - p⁺⁺) / 2
end
function extrap!(::GeometricMean, p, T, p⁺, T⁺, p⁺⁺, T⁺⁺, Tₛ, params)
    @. T = sqrt(T⁺^3 / T⁺⁺)
    @. p = sqrt(p⁺^3 / p⁺⁺)
end
function extrap!(::UniformZ, p, T, p⁺, T⁺, p⁺⁺, T⁺⁺, Tₛ, params)
    @. T = (3 * T⁺ - T⁺⁺) / 2
    @. p = uniform_z_p(T, p⁺, T⁺, p⁺⁺, T⁺⁺)
end
function extrap!(::UniformP, p, T, p⁺, T⁺, p⁺⁺, T⁺⁺, Tₛ, params)
    @. p = (3 * p⁺ - p⁺⁺) / 2
    @. T = T⁺ * (T⁺⁺ / T⁺)^(log(p / p⁺) / log(p⁺⁺ / p⁺)) # assume that p⁺ != p⁺⁺
end
function extrap!(::BestFit, p, T, z, p⁺, T⁺, z⁺, p⁺⁺, T⁺⁺, z⁺⁺, Tₛ, params)
    @. T = T⁺ + (T⁺⁺ - T⁺) * (z - z⁺) / (z⁺⁺ - z⁺)
    @. p = best_fit_p(T, z, p⁺, T⁺, z⁺, p⁺⁺, T⁺⁺, z⁺⁺)
end
function extrap!(::UseSurfaceTempAtBottom, p, T, p⁺, T⁺, p⁺⁺, T⁺⁺, Tₛ, params)
    cₚ = RRTMGP.Parameters.cp_d(params)
    R = RRTMGP.Parameters.R_d(params)
    @. T = Tₛ
    @. p = p⁺ * (T / T⁺)^(cₚ / R)
end
function extrap!(
    ::HydrostaticBottom,
    p,
    T,
    z,
    p⁺,
    T⁺,
    z⁺,
    p⁺⁺,
    T⁺⁺,
    z⁺⁺,
    Tₛ,
    params,
)
    FT = eltype(p)
    g = FT(RRTMGP.Parameters.grav(params))
    cₚ = FT(RRTMGP.Parameters.cp_d(params))
    R = FT(RRTMGP.Parameters.R_d(params))
    @. T = T⁺ + g / cₚ * (z⁺ - z)
    @. p = p⁺ * (T / T⁺)^(cₚ / R)
end

struct RRTMGPModel{R, I, B, L, P, LWS, SWS, AS, V, M}
    radiation_mode::R
    interpolation::I
    bottom_extrapolation::B
    lookups::L
    params::P
    lw_solver::LWS
    sw_solver::SWS
    as::AS  # Atmospheric state
    views::V  # user-friendly views into the solver
    metric_scaling::M
end

# Allow cache to be moved on the CPU. Used by ClimaCoupler to save checkpoints
Adapt.@adapt_structure RRTMGPModel

function Base.getproperty(model::RRTMGPModel, s::Symbol)
    if s in fieldnames(typeof(model))
        return getfield(model, s)
    else
        return getproperty(getfield(model, :views), s)
    end
end

function Base.propertynames(model::RRTMGPModel, private::Bool = false)
    names = propertynames(getfield(model, :views))
    return private ? (names..., fieldnames(typeof(model))...) : names
end

get_radiation_method(m::GrayRadiation) = RRTMGP.GrayRadiation()
get_radiation_method(m::ClearSkyRadiation) =
    RRTMGP.ClearSkyRadiation(m.aerosol_radiation)
get_radiation_method(m::AllSkyRadiation) =
    RRTMGP.AllSkyRadiation(m.aerosol_radiation, m.reset_rng_seed)
get_radiation_method(m::AllSkyRadiationWithClearSkyDiagnostics) =
    RRTMGP.AllSkyRadiationWithClearSkyDiagnostics(
        m.aerosol_radiation,
        m.reset_rng_seed,
    )

"""
    RRTMGPModel(params; kwargs...)

A user-friendly interface for `RRTMGP.jl`. Stores an `RRTMGP.RTE.Solver`, along
with all of the data required to use it. Provides easy access to `RRTMGP`'s
inputs and outputs (e.g., `model.center_temperature` and `model.face_flux`).

After constructing an `RRTMGPModel`, use it as follows:
- update all the inputs that have changed since it was constructed; e.g.,
  `model.center_temperature .= field2array(current_center_temperature_field)`
- call `update_fluxes!(model)`
- use the values of any fluxes of interest; e.g.,
  `field2array(face_flux_field) .= model.face_flux`

The `RRTMGPModel` assumes that pressure and temperature live on cell centers,
and internally interpolates data to cell faces when needed.

Every keyword argument that corresponds to an array of cell center or cell face
values can be specified as a scalar (corresponding to a constant value
throughout the atmosphere), or as a 1D array (corresponding to the values in
each column), or as a 2D array with a single row (corresponding to the values in
each level), or as the full 2D array specifying the value at every point.
Similarly, every keyword argument that corresponds to an array of values at the
top/bottom of the atmosphere can be specified as a scalar, or as the full 1D
array.

# Positional Arguments
- Artifacts stored in `RRTMGPReferenceData/<file_name>`.
   Should be callable with filenames:
  - `lookup_tables/clearsky_lw.nc`
  - `lookup_tables/cloudysky_lw.nc`
  - `lookup_tables/clearsky_sw.nc`
  - `lookup_tables/cloudysky_sw.nc`
- `FT`: floating-point number type (performance with `Float32` is questionable)
- `DA`: array type (defaults to `CuArray` when a compatible GPU is available)

# Keyword Arguments
- `ncol`: number of vertical columns in the domain/extension
- `domain_nlay`: number of cells (layers) in the domain
- `radiation_mode`: overall mode for running `RRTMGP`; available options are
    - `GrayRadiation`: uniform absorption across all frequencies
    - `ClearSkyRadiation`: full RRTMGP model, but without clouds
    - `AllSkyRadiation`: full RRTMGP model
    - `AllSkyRadiationWithClearSkyDiagnostics`: computes the fluxes for both
      `AllSkyRadiation` and `ClearSkyRadiation`
- `interpolation`: method for determining implied values (if there are any);
  see documentation for `AbstractInterpolation` for available options
- `bottom_extrapolation`: method for determining implied values at the bottom
  cell face (only used when the cell face values are implied); see documentation
  for `AbstractInterpolation` for available options
- `use_global_means_for_well_mixed_gases`: whether to use a scalar value to
  represent the volume mixing ratio of each well-mixed gas (i.e., a gas that is
  not water vapor or ozone), instead of using an array that represents a
  spatially varying volume mixing ratio
- `center_pressure` and/or `face_pressure`: air pressure in Pa on cell centers
  and on cell faces (either one or both of these must be specified)
- `center_temperature` and/or `face_temperature`: air temperature in K on cell
  centers and on cell faces (if `center_pressure` is specified, then
  `center_temperature` must also be specified, and, if `face_pressure` is
  specified, then `face_temperature` must also be specified)
- `surface_temperature`: temperature of the surface in K (required)
- `surface_emissivity`: longwave emissivity of the surface (required)
- `top_of_atmosphere_lw_flux_dn`: incoming longwave radiation in W/m^2
  (assumed to be 0 by default)
- `direct_sw_surface_albedo`: direct shortwave albedo of the surface
  (required)
- `diffuse_sw_surface_albedo`: diffuse shortwave albedo of the surface
  (required)
- `cos_zenith`: cosine of the zenith angle of sun in radians (required)
- `toa_flux`: irradiance of sun in W/m^2 (required); the incoming
  direct shortwave radiation is given by
  `model.toa_flux .* model.cos_zenith`
- `top_of_atmosphere_diffuse_sw_flux_dn`: incoming diffuse shortwave
  radiation in W/m^2 (assumed to be 0 by default)
- arguments only available when `radiation_mode isa GrayRadiation`:
    - `lapse_rate`: a scalar value that specifies the lapse rate throughout the
      atmosphere (required); this is a constant that can't be modified after the
      model is constructed
    - `optical_thickness_parameter`: the longwave optical depth at the surface
      (required)
- arguments only available when `!(radiation_mode isa GrayRadiation)`:
    - `center_volume_mixing_ratio_h2o`: volume mixing ratio of water vapor on
      cell centers (required)
    - `center_volume_mixing_ratio_o3`: volume mixing ratio of ozone on cell
      centers (required)
    - arguments only available when `use_global_means_for_well_mixed_gases`:
        - `volume_mixing_ratio_<gas_name>` for `gas_name` in `co2`, `n2o`, `co`,
          `ch4`, `o2`, `n2`, `ccl4`, `cfc11`, `cfc12`, `cfc22`, `hfc143a`,
          `hfc125`, `hfc23`, `hfc32`, `hfc134a`, `cf4`, `no2`: a scalar value
          that specifies the volume mixing ratio of each well-mixed gas
          throughout the atmosphere (required)
    - arguments only available when `!use_global_means_for_well_mixed_gases`:
        - `center_volume_mixing_ratio_<gas_name>` for `gas_name` in `co2`,
          `n2o`, `co`,`ch4`, `o2`, `n2`, `ccl4`, `cfc11`, `cfc12`, `cfc22`,
          `hfc143a`, `hfc125`, `hfc23`, `hfc32`, `hfc134a`, `cf4`, `no2`: volume
          mixing ratio of each well-mixed gas on cell centers (required)
    - arguments only available when `!(radiation_mode isa ClearSkyRadiation)`:
        - `center_cloud_liquid_effective_radius`: effective radius of cloud
        liquid water in m on cell centers (required)
        - `center_cloud_ice_effective_radius`: effective radius of cloud ice
        water in m on cell centers (required)
        - `center_cloud_liquid_water_path`: mean path length of cloud liquid
        water in m on cell centers (required)
        - `center_cloud_ice_water_path`: mean path length of cloud ice water in
        m on cell centers (required)
        - `center_cloud_fraction`: cloud fraction on cell centers (required)
        - `ice_roughness`: either 1, 2, or 3, with 3 corresponding to the
        roughest ice (required); this is a constant that can't be modified after
        the model is constructed
    - `latitude`: latitude in degrees (assumed to be 45 by default); used for
      computing the concentration of air in molecules/cm^2
- arguments only available when
  `requires_z(interpolation) || requires_z(bottom_extrapolation)`:
    - `center_z`: z-coordinate in m at cell centers
    - `face_z`: z-coordinate in m at cell faces
    - `planet_radius`: planet radius (used to compute metric scaling factor)
"""
RRTMGPModel(
    params::RRTMGP.Parameters.ARP,
    context;
    ncol::Int,
    domain_nlay::Int,
    radiation_mode::AbstractRRTMGPMode = ClearSkyRadiation(),
    interpolation::AbstractInterpolation = NoInterpolation(),
    bottom_extrapolation::AbstractBottomExtrapolation = SameAsInterpolation(),
    use_global_means_for_well_mixed_gases::Bool = false,
    kwargs...,
) = _RRTMGPModel(
    params,
    context,
    radiation_mode;
    ncol,
    domain_nlay,
    interpolation,
    bottom_extrapolation,
    use_global_means_for_well_mixed_gases,
    kwargs...,
)

# TODO: make this the new interface for the next breaking release.
function _RRTMGPModel(
    params::RRTMGP.Parameters.ARP,
    context,
    radiation_mode::AbstractRRTMGPMode = ClearSkyRadiation();
    ncol::Int,
    domain_nlay::Int,
    interpolation::AbstractInterpolation = NoInterpolation(),
    bottom_extrapolation::AbstractBottomExtrapolation = SameAsInterpolation(),
    use_global_means_for_well_mixed_gases::Bool = false,
    kwargs...,
)
    device = ClimaComms.device(context)
    DA = ClimaComms.array_type(device)
    FT = typeof(params.grav)
    # turn kwargs into a Dict, so that values can be dynamically popped from it
    dict = Dict(kwargs)
    nlay = domain_nlay + Int(radiation_mode.add_isothermal_boundary_layer)
    grid_params = RRTMGP.RRTMGPGridParams(
        FT;
        context,
        nlay,
        ncol,
        isothermal_boundary_layer = radiation_mode.add_isothermal_boundary_layer,
    )
    op = RRTMGP.Optics.TwoStream(grid_params)
    if use_global_means_for_well_mixed_gases && radiation_mode isa GrayRadiation
        @warn "use_global_means_for_well_mixed_gases is ignored when using \
               GrayRadiation"
    end

    if interpolation isa NoInterpolation
        error("interpolation cannot be NoInterpolation if only center \
               pressures/temperatures are specified")
    end

    radiation_method = get_radiation_method(radiation_mode)
    (; lookups, lu_kwargs) = RRTMGP.lookup_tables(grid_params, radiation_method)
    views = []

    t = (views, domain_nlay)

    src_lw = if op isa RRTMGP.Optics.OneScalar
        RRTMGP.SourceLWNoScat(grid_params; params)
    else
        RRTMGP.SourceLW2Str(grid_params; params)
    end
    flux_lw = RRTMGP.Fluxes.FluxLW(grid_params)
    fluxb_lw =
        radiation_mode isa GrayRadiation ? nothing :
        RRTMGP.Fluxes.FluxLW(grid_params)
    set_and_save!(flux_lw.flux_up, "face_lw_flux_up", t...)
    set_and_save!(flux_lw.flux_dn, "face_lw_flux_dn", t...)
    set_and_save!(flux_lw.flux_net, "face_lw_flux", t...)
    if radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics
        flux_lw2 = RRTMGP.Fluxes.FluxLW(grid_params)
        set_and_save!(flux_lw2.flux_up, "face_clear_lw_flux_up", t...)
        set_and_save!(flux_lw2.flux_dn, "face_clear_lw_flux_dn", t...)
        set_and_save!(flux_lw2.flux_net, "face_clear_lw_flux", t...)
    end

    sfc_emis = DA{FT}(undef, lu_kwargs.nbnd_lw, ncol)
    set_and_save!(sfc_emis, "surface_emissivity", t..., dict)
    name = "top_of_atmosphere_lw_flux_dn"
    if Symbol(name) in keys(dict)
        inc_flux = DA{FT}(undef, ncol)
        set_and_save!(transpose(inc_flux), name, t..., dict)
    else
        inc_flux = nothing
    end
    bcs_lw = RRTMGP.BCs.LwBCs(sfc_emis, inc_flux)
    src_sw =
        op isa RRTMGP.Optics.OneScalar ? nothing :
        RRTMGP.SourceSW2Str(grid_params)
    flux_sw = RRTMGP.Fluxes.FluxSW(grid_params)
    fluxb_sw =
        radiation_mode isa GrayRadiation ? nothing :
        RRTMGP.Fluxes.FluxSW(grid_params)
    set_and_save!(flux_sw.flux_up, "face_sw_flux_up", t...)
    set_and_save!(flux_sw.flux_dn, "face_sw_flux_dn", t...)
    set_and_save!(flux_sw.flux_net, "face_sw_flux", t...)
    set_and_save!(flux_sw.flux_dn_dir, "face_sw_direct_flux_dn", t...)
    if radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics
        flux_sw2 = RRTMGP.Fluxes.FluxSW(grid_params)
        set_and_save!(flux_sw2.flux_up, "face_clear_sw_flux_up", t...)
        set_and_save!(flux_sw2.flux_dn, "face_clear_sw_flux_dn", t...)
        set_and_save!(
            flux_sw2.flux_dn_dir,
            "face_clear_sw_direct_flux_dn",
            t...,
        )
        set_and_save!(flux_sw2.flux_net, "face_clear_sw_flux", t...)
    end

    cos_zenith = DA{FT}(undef, ncol)
    set_and_save!(cos_zenith, "cos_zenith", t..., dict)
    toa_flux = DA{FT}(undef, ncol)
    set_and_save!(toa_flux, "toa_flux", t..., dict)
    sfc_alb_direct = DA{FT}(undef, lu_kwargs.nbnd_sw, ncol)
    set_and_save!(sfc_alb_direct, "direct_sw_surface_albedo", t..., dict)
    sfc_alb_diffuse = DA{FT}(undef, lu_kwargs.nbnd_sw, ncol)
    set_and_save!(sfc_alb_diffuse, "diffuse_sw_surface_albedo", t..., dict)
    name = "top_of_atmosphere_diffuse_sw_flux_dn"
    if Symbol(name) in keys(dict)
        @warn "incoming diffuse shortwave fluxes are not yet implemented \
               in RRTMGP.jl; the value of $name will be ignored"
        inc_flux_diffuse = DA{FT}(undef, ncol, ngpt_sw)
        set_and_save!(transpose(inc_flux_diffuse), name, t..., dict)
    else
        inc_flux_diffuse = nothing
    end
    bcs_sw = RRTMGP.BCs.SwBCs(
        cos_zenith,
        toa_flux,
        sfc_alb_direct,
        inc_flux_diffuse,
        sfc_alb_diffuse,
    )

    set_and_save!(similar(flux_lw.flux_net), "face_flux", t...)
    if radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics
        set_and_save!(similar(flux_lw2.flux_net), "face_clear_flux", t...)
    end

    if !(:latitude in keys(dict))
        lon = lat = nothing
    else
        lon = DA{FT}(undef, ncol) # TODO: lon required but unused
        lat = DA{FT}(undef, ncol)
        set_and_save!(lat, "latitude", t..., dict)
    end

    p_lev = DA{FT}(undef, nlay + 1, ncol)
    if radiation_mode.deep_atmosphere && :planet_radius in keys(dict)
        metric_scaling = DA{FT}(undef, nlay + 1, ncol)
    else
        metric_scaling = nothing
    end
    t_lev = DA{FT}(undef, nlay + 1, ncol)
    t_sfc = DA{FT}(undef, ncol)
    set_and_save!(t_sfc, "surface_temperature", t..., dict)

    if radiation_mode isa GrayRadiation
        p_lay = DA{FT}(undef, nlay, ncol)
        t_lay = DA{FT}(undef, nlay, ncol)
        set_and_save!(p_lay, "center_pressure", t..., dict)
        set_and_save!(t_lay, "center_temperature", t..., dict)

        z_lev = DA{FT}(undef, nlay + 1, ncol) # TODO: z_lev required but unused

        # lapse_rate is a constant, so don't use set_and_save! to get it
        :lapse_rate in keys(dict) || throw(UndefKeywordError(:lapse_rate))
        α = pop!(dict, :lapse_rate)
        α isa Real || error("lapse_rate must be a Real")

        d0 = DA{FT}(undef, ncol)
        set_and_save!(d0, "optical_thickness_parameter", t..., dict)
        otp = RRTMGP.AtmosphericStates.GrayOpticalThicknessOGorman2008(FT)
        as = RRTMGP.AtmosphericStates.GrayAtmosphericState(
            lat,
            p_lay,
            p_lev,
            t_lay,
            t_lev,
            z_lev,
            t_sfc,
            otp,
        )
    else
        layerdata = DA{FT}(undef, 4, nlay, ncol)
        p_lay = view(layerdata, 2, :, :)
        t_lay = view(layerdata, 3, :, :)
        rh_lay = view(layerdata, 4, :, :)
        set_and_save!(p_lay, "center_pressure", t..., dict)
        set_and_save!(t_lay, "center_temperature", t..., dict)
        set_and_save!(rh_lay, "center_relative_humidity", t..., dict)
        vmr_str = "volume_mixing_ratio_"
        gas_names = filter(
            gas_name ->
                !(gas_name in ("h2o", "h2o_frgn", "h2o_self", "o3")),
            RRTMGP.gas_names_sw(),
        )
        # TODO: This gives the wrong types for CUDA 3.4 and above.
        # gm = use_global_means_for_well_mixed_gases
        # vmr = RRTMGP.Vmrs.init_vmr(ngas, nlay, ncol, FT, DA; gm)
        if use_global_means_for_well_mixed_gases
            vmr = RRTMGP.Vmrs.VmrGM(
                DA{FT}(undef, nlay, ncol),
                DA{FT}(undef, nlay, ncol),
                DA{FT}(undef, lu_kwargs.ngas_sw),
            )
            vmr.vmr .= 0 # TODO: do we need this?
            set_and_save!(vmr.vmr_h2o, "center_$(vmr_str)h2o", t..., dict)
            set_and_save!(vmr.vmr_o3, "center_$(vmr_str)o3", t..., dict)
            for gas_name in gas_names
                gas_view = view(vmr.vmr, lookups.idx_gases_sw[gas_name])
                set_and_save!(gas_view, "$vmr_str$gas_name", t..., dict)
            end
        else
            vmr = RRTMGP.Vmrs.Vmr(DA{FT}(undef, lu_kwargs.ngas_sw, nlay, ncol))
            for gas_name in ["h2o", "o3", gas_names...]
                gas_view = view(vmr.vmr, lookups.idx_gases_sw[gas_name], :, :)
                set_and_save!(gas_view, "center_$vmr_str$gas_name", t..., dict)
            end
        end

        if radiation_mode isa ClearSkyRadiation
            cloud_state = nothing
        else
            cld_r_eff_liq = DA{FT}(undef, nlay, ncol)
            name = "center_cloud_liquid_effective_radius"
            set_and_save!(cld_r_eff_liq, name, t..., dict)
            cld_r_eff_ice = DA{FT}(undef, nlay, ncol)
            name = "center_cloud_ice_effective_radius"
            set_and_save!(cld_r_eff_ice, name, t..., dict)
            cld_path_liq = DA{FT}(undef, nlay, ncol)
            name = "center_cloud_liquid_water_path"
            set_and_save!(cld_path_liq, name, t..., dict)
            cld_path_ice = DA{FT}(undef, nlay, ncol)
            name = "center_cloud_ice_water_path"
            set_and_save!(cld_path_ice, name, t..., dict)
            cld_frac = DA{FT}(undef, nlay, ncol)
            set_and_save!(cld_frac, "center_cloud_fraction", t..., dict)
            cld_mask_lw = DA{Bool}(undef, nlay, ncol)
            cld_mask_sw = DA{Bool}(undef, nlay, ncol)
            cld_overlap = RRTMGP.AtmosphericStates.MaxRandomOverlap()

            # ice_roughness is a constant, so don't use set_and_save! to get it
            if !(:ice_roughness in keys(dict))
                throw(UndefKeywordError(:ice_roughness))
            end
            ice_rgh = pop!(dict, :ice_roughness)
            if !(ice_rgh in (1, 2, 3))
                error("ice_roughness must be either 1, 2, or 3")
            end

            cloud_state = RRTMGP.AtmosphericStates.CloudState(
                cld_r_eff_liq,
                cld_r_eff_ice,
                cld_path_liq,
                cld_path_ice,
                cld_frac,
                cld_mask_lw,
                cld_mask_sw,
                cld_overlap,
                ice_rgh,
            )
        end

        if radiation_mode.aerosol_radiation
            aod_sw_ext = DA{FT}(undef, ncol)
            aod_sw_sca = DA{FT}(undef, ncol)
            aero_mask = DA{Bool}(undef, nlay, ncol)
            set_and_save!(aod_sw_ext, "aod_sw_extinction", t..., dict)
            set_and_save!(aod_sw_sca, "aod_sw_scattering", t..., dict)

            n_aerosol_sizes = maximum(values(lookups.idx_aerosize_sw)) # TODO: verify correctness
            n_aerosols = length(lookups.idx_aerosol_sw) # TODO: verify correctness
            # See the lookup table in RRTMGP for the order of aerosols
            aero_size = DA{FT}(undef, n_aerosol_sizes, nlay, ncol)
            aero_mass = DA{FT}(undef, n_aerosols, nlay, ncol)

            aerosol_names = [
                "dust1",
                "ss1",
                "so4",
                "bcpi",
                "bcpo",
                "ocpi",
                "ocpo",
                "dust2",
                "dust3",
                "dust4",
                "dust5",
                "ss2",
                "ss3",
                "ss4",
                "ss5",
            ]
            for (i, name) in enumerate(aerosol_names)
                if occursin("dust", name) || occursin("ss", name)
                    set_and_save!(
                        view(aero_size, i, :, :),
                        "center_$(name)_radius",
                        t...,
                        dict,
                    )
                end
            end
            for (i, name) in enumerate(aerosol_names)
                set_and_save!(
                    view(aero_mass, i, :, :),
                    "center_$(name)_column_mass_density",
                    t...,
                    dict,
                )
            end
            aerosol_state = RRTMGP.AtmosphericStates.AerosolState(
                aod_sw_ext,
                aod_sw_sca,
                aero_mask,
                aero_size,
                aero_mass,
            )
        else
            aerosol_state = nothing
        end
        as = RRTMGP.AtmosphericStates.AtmosphericState(
            lon,
            lat,
            # layerdata contains `col_dry`, `p_lay`, and `t_lay`
            layerdata,
            p_lev,
            t_lev,
            t_sfc,
            vmr,
            cloud_state,
            aerosol_state,
        )
    end

    sw_solver = RRTMGP.RTE.TwoStreamSWRTE(
        context,
        op,
        src_sw,
        bcs_sw,
        fluxb_sw,
        flux_sw,
    )
    lw_solver = RRTMGP.RTE.TwoStreamLWRTE(
        context,
        op,
        src_lw,
        bcs_lw,
        fluxb_lw,
        flux_lw,
    )

    if requires_z(interpolation) || requires_z(bottom_extrapolation)
        z_lay = DA{FT}(undef, nlay, ncol)
        set_and_save!(z_lay, "center_z", t..., dict)
        z_lev = DA{FT}(undef, nlay + 1, ncol)
        set_and_save!(z_lev, "face_z", t..., dict)
        if radiation_mode.deep_atmosphere && :planet_radius in keys(dict)
            planet_radius = pop!(dict, :planet_radius)
            # Area ratio appears in denominator of RRTMGP scaling functions,
            # we therefore pass the multiplicative inverse from ClimaAtmos to
            # use mult ops instead of div in RRTMGP GPU kernels.
            metric_scaling .=
                inv.(((z_lev .+ planet_radius) ./ planet_radius) .^ (FT(2)))
        end
    end

    if length(dict) > 0
        @warn string(
            "unused keyword argument",
            length(dict) == 1 ? " " : "s ",
            join(keys(dict), ", ", length(dict) == 2 ? " and " : ", and "),
        )
    end

    return RRTMGPModel(
        radiation_mode,
        interpolation,
        bottom_extrapolation,
        lookups,
        params,
        lw_solver,
        sw_solver,
        as,
        NamedTuple(views),
        metric_scaling,
    )
end

# This sets `array .= value`, but it allows `array` to be to be a `CuArray`
# while `value` is an `Array` (in which case broadcasting throws an error).
set_array!(array, value::Real, symbol) = fill!(array, value)
function set_array!(array, value::AbstractArray{<:Real}, symbol)
    if ndims(array) == 2
        if size(value) == size(array)
            copyto!(array, value)
        elseif size(value) == (size(array, 1),)
            for col in eachcol(array)
                copyto!(col, value)
            end
        elseif size(value) == (1, size(array, 2))
            for (icol, col) in enumerate(eachcol(array))
                fill!(col, value[1, icol])
            end
        else
            error("expected $symbol to be an array of size $(size(array)), \
                   ($(size(array, 1)),), or (1, $(size(array, 2))); received \
                   an array of size $(size(value))")
        end
    else
        if size(value) == size(array)
            copyto!(array, value)
        else
            error("expected $symbol to be an array of size $(size(array)); \
                   received an array of size $(size(value))")
        end
    end
end

function set_and_save!(array, name, views, domain_nlay, dict = nothing)
    domain_symbol = Symbol(name)

    if isnothing(dict)
        domain_value = NaN
    else
        if !(domain_symbol in keys(dict))
            throw(UndefKeywordError(domain_symbol))
        end
        domain_value = pop!(dict, domain_symbol)
    end

    if startswith(name, "center_") || startswith(name, "face_")
        domain_range =
            startswith(name, "center_") ? (1:domain_nlay) :
            (1:(domain_nlay + 1))
        domain_view = view(array, domain_range, :)
        set_array!(domain_view, domain_value, domain_symbol)
        push!(views, (domain_symbol, domain_view))
    else
        set_array!(array, domain_value, domain_symbol)
        push!(views, (domain_symbol, array))
    end
end

"""
    update_fluxes!(model, seedval)

Updates the fluxes in the `RRTMGPModel` based on its internal state. Returns the
net flux at cell faces in the domain, `model.face_flux`. The full set of fluxes
available in the model after calling this function is
- `face_flux`
- `face_lw_flux`, `face_lw_flux_dn`, `face_lw_flux_up`
- `face_sw_flux`, `face_sw_flux_dn`, `face_sw_flux_up`, `face_sw_direct_flux_dn`
If `radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics`, the set of
available fluxes also includes
- `face_clear_flux`
- `face_clear_lw_flux`, `face_clear_lw_flux_dn`, `face_clear_lw_flux_up`
- `face_clear_sw_flux`, `face_clear_sw_flux_dn`, `face_clear_sw_flux_up`,
  `face_clear_sw_direct_flux_dn`
"""
NVTX.@annotate function update_fluxes!(model, seedval)
    (; radiation_mode) = model
    if (
        radiation_mode isa AllSkyRadiation ||
        radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics
    )
        radiation_mode.reset_rng_seed && Random.seed!(seedval)
    end
    update_implied_values!(model)
    model.radiation_mode.add_isothermal_boundary_layer &&
        update_boundary_layer!(model)
    clip_values!(model)
    # TODO : update_concentrations as part of update_atmospheric_state!
    update_concentrations!(model.radiation_mode, model)
    update_lw_fluxes!(model.radiation_mode, model)
    update_sw_fluxes!(model.radiation_mode, model)
    update_net_fluxes!(model.radiation_mode, model)
    return model.face_flux
end

get_p_min(model) = get_p_min(model.as, model.lookups)
get_p_min(as::RRTMGP.AtmosphericStates.GrayAtmosphericState, lookups) =
    zero(eltype(as.p_lay))
get_p_min(as::RRTMGP.AtmosphericStates.AtmosphericState, lookups) =
    lookups.lookup_lw.p_ref_min # TODO: verify correctness

function update_implied_values!(model)
    (; p_lev, t_lev, t_sfc) = model.as
    p_lay = AS.getview_p_lay(model.as)
    t_lay = AS.getview_t_lay(model.as)
    nlay =
        size(p_lay, 1) - Int(model.radiation_mode.add_isothermal_boundary_layer)
    if requires_z(model.interpolation) || requires_z(model.bottom_extrapolation)
        z_lay = parent(model.center_z)
        z_lev = parent(model.face_z)
    end
    mode = model.interpolation
    outs = requires_z(mode) ? (p_lev, t_lev, z_lev) : (p_lev, t_lev)
    ins = requires_z(mode) ? (p_lay, t_lay, z_lay) : (p_lay, t_lay)
    update_views(interp!, mode, outs, ins, (), 2:nlay, 1:(nlay - 1), 2:nlay)
    others = (t_sfc, model.params)
    update_views(extrap!, mode, outs, ins, others, nlay + 1, nlay, nlay - 1)
    mode =
        model.bottom_extrapolation isa SameAsInterpolation ?
        model.interpolation : model.bottom_extrapolation
    outs = requires_z(mode) ? (p_lev, t_lev, z_lev) : (p_lev, t_lev)
    ins = requires_z(mode) ? (p_lay, t_lay, z_lay) : (p_lay, t_lay)
    update_views(extrap!, mode, outs, ins, others, 1, 1, 2)
end

update_views(f, mode, outs, ins, others, out_range, in_range1, in_range2) = f(
    mode,
    map(out -> view(out, out_range, :), outs)...,
    map(in -> view(in, in_range1, :), ins)...,
    map(in -> view(in, in_range2, :), ins)...,
    others...,
)

"""
    update_boundary_layer!(model)

Adds an isothermal boundary layer above the domain and extension (the pressure at the top of this layer
is the minimum pressure supported by RRTMGP, while the temperature and volume mixing ratios in this layer
are the same as in the layer below it)
"""
function update_boundary_layer!(model)
    as = model.as
    p_min = get_p_min(model)
    @views AS.getview_p_lay(model.as)[end, :] .=
        (as.p_lev[end - 1, :] .+ p_min) ./ 2
    @views as.p_lev[end, :] .= p_min
    @views AS.getview_t_lay(model.as)[end, :] .= as.t_lev[end - 1, :]
    @views as.t_lev[end, :] .= as.t_lev[end - 1, :]
    if !(model.radiation_mode isa GrayRadiation)
        @views AS.getview_rel_hum(model.as)[end, :] .=
            AS.getview_rel_hum(model.as)[end - 1, :]
    end
    update_boundary_layer_vmr!(model.radiation_mode, as)
    update_boundary_layer_cloud!(model.radiation_mode, as)
    update_boundary_layer_aerosol!(model.radiation_mode, as)
end
update_boundary_layer_vmr!(::GrayRadiation, as) = nothing
update_boundary_layer_vmr!(radiation_mode, as) =
    update_boundary_layer_vmr!(as.vmr)
function update_boundary_layer_vmr!(vmr::RRTMGP.Vmrs.VmrGM)
    @views vmr.vmr_h2o[end, :] .= vmr.vmr_h2o[end - 1, :]
    @views vmr.vmr_o3[end, :] .= vmr.vmr_o3[end - 1, :]
end
update_boundary_layer_vmr!(vmr::RRTMGP.Vmrs.Vmr) =
    @views vmr.vmr[:, end, :] .= vmr.vmr[:, end - 1, :]

update_boundary_layer_cloud!(::Union{GrayRadiation, ClearSkyRadiation}, _) =
    nothing
update_boundary_layer_cloud!(
    ::Union{AllSkyRadiation, AllSkyRadiationWithClearSkyDiagnostics},
    as,
) = update_boundary_layer_cloud!(as.cloud_state)

function update_boundary_layer_cloud!(cloud_state)
    @views cloud_state.cld_r_eff_liq[end, :] .=
        cloud_state.cld_r_eff_liq[end - 1, :]
    @views cloud_state.cld_r_eff_ice[end, :] .=
        cloud_state.cld_r_eff_ice[end - 1, :]
    @views cloud_state.cld_path_liq[end, :] .=
        cloud_state.cld_path_liq[end - 1, :]
    @views cloud_state.cld_path_ice[end, :] .=
        cloud_state.cld_path_ice[end - 1, :]
    @views cloud_state.cld_frac[end, :] .= cloud_state.cld_frac[end - 1, :]
end

update_boundary_layer_aerosol!(::GrayRadiation, _) = nothing
update_boundary_layer_aerosol!(::AbstractRRTMGPMode, as) =
    update_boundary_layer_aerosol!(as.aerosol_state)
update_boundary_layer_aerosol!(::Nothing) = nothing
function update_boundary_layer_aerosol!(
    aerosol_state::RRTMGP.AtmosphericStates.AerosolState,
)
    @views aerosol_state.aero_size[:, end, :] .=
        aerosol_state.aero_size[:, end - 1, :]
    @views aerosol_state.aero_mass[:, end, :] .=
        aerosol_state.aero_mass[:, end - 1, :]
end

function clip_values!(model)
    (; p_lev) = model.as
    p_lay = AS.getview_p_lay(model.as)
    if !(model.radiation_mode isa GrayRadiation)
        (; vmr_h2o) = model.as.vmr
        @. vmr_h2o = max(vmr_h2o, 0)
    end
    p_min = get_p_min(model)
    @. p_lay = max(p_lay, p_min)
    @. p_lev = max(p_lev, p_min)
end
update_concentrations!(::GrayRadiation, model) = nothing

update_concentrations!(radiation_mode, model) = RRTMGP.Optics.compute_col_gas!(
    ClimaComms.device(model.sw_solver.context),
    model.as.p_lev,
    AS.getview_col_dry(model.as),
    model.params,
    get_vmr_h2o(model.as.vmr, model.lookups.idx_gases_sw),
    model.as.lat,
)
get_vmr_h2o(vmr::RRTMGP.Vmrs.VmrGM, idx_gases_sw) = vmr.vmr_h2o
get_vmr_h2o(vmr::RRTMGP.Vmrs.Vmr, idx_gases_sw) =
    view(vmr.vmr, idx_gases_sw["h2o"], :, :)

NVTX.@annotate update_lw_fluxes!(::GrayRadiation, model) =
    RRTMGP.RTESolver.solve_lw!(model.lw_solver, model.as, model.metric_scaling)
NVTX.@annotate update_lw_fluxes!(::ClearSkyRadiation, model) =
    RRTMGP.RTESolver.solve_lw!(
        model.lw_solver,
        model.as,
        model.lookups.lookup_lw,
        nothing,
        model.lookups.lookup_lw_aero,
        model.metric_scaling,
    )
NVTX.@annotate update_lw_fluxes!(::AllSkyRadiation, model) =
    RRTMGP.RTESolver.solve_lw!(
        model.lw_solver,
        model.as,
        model.lookups.lookup_lw,
        model.lookups.lookup_lw_cld,
        model.lookups.lookup_lw_aero,
        model.metric_scaling,
    )
NVTX.@annotate function update_lw_fluxes!(
    ::AllSkyRadiationWithClearSkyDiagnostics,
    model,
)
    RRTMGP.RTESolver.solve_lw!(
        model.lw_solver,
        model.as,
        model.lookups.lookup_lw,
        nothing,
        model.lookups.lookup_lw_aero,
        model.metric_scaling,
    )
    parent(model.face_clear_lw_flux_up) .= parent(model.face_lw_flux_up)
    parent(model.face_clear_lw_flux_dn) .= parent(model.face_lw_flux_dn)
    parent(model.face_clear_lw_flux) .= parent(model.face_lw_flux)
    RRTMGP.RTESolver.solve_lw!(
        model.lw_solver,
        model.as,
        model.lookups.lookup_lw,
        model.lookups.lookup_lw_cld,
        model.lookups.lookup_lw_aero,
        model.metric_scaling,
    )
end

NVTX.@annotate update_sw_fluxes!(::GrayRadiation, model) =
    RRTMGP.RTESolver.solve_sw!(model.sw_solver, model.as, model.metric_scaling)
NVTX.@annotate update_sw_fluxes!(::ClearSkyRadiation, model) =
    RRTMGP.RTESolver.solve_sw!(
        model.sw_solver,
        model.as,
        model.lookups.lookup_sw,
        nothing,
        model.lookups.lookup_sw_aero,
        model.metric_scaling,
    )
NVTX.@annotate update_sw_fluxes!(::AllSkyRadiation, model) =
    RRTMGP.RTESolver.solve_sw!(
        model.sw_solver,
        model.as,
        model.lookups.lookup_sw,
        model.lookups.lookup_sw_cld,
        model.lookups.lookup_sw_aero,
        model.metric_scaling,
    )
NVTX.@annotate function update_sw_fluxes!(
    ::AllSkyRadiationWithClearSkyDiagnostics,
    model,
)
    RRTMGP.RTESolver.solve_sw!(
        model.sw_solver,
        model.as,
        model.lookups.lookup_sw,
        nothing,
        model.lookups.lookup_sw_aero,
        model.metric_scaling,
    )
    parent(model.face_clear_sw_flux_up) .= parent(model.face_sw_flux_up)
    parent(model.face_clear_sw_flux_dn) .= parent(model.face_sw_flux_dn)
    parent(model.face_clear_sw_direct_flux_dn) .=
        parent(model.face_sw_direct_flux_dn)
    parent(model.face_clear_sw_flux) .= parent(model.face_sw_flux)
    RRTMGP.RTESolver.solve_sw!(
        model.sw_solver,
        model.as,
        model.lookups.lookup_sw,
        model.lookups.lookup_sw_cld,
        model.lookups.lookup_sw_aero,
        model.metric_scaling,
    )
end

function update_net_fluxes!(_, model)
    model.face_flux .= model.face_lw_flux .+ model.face_sw_flux

end
function update_net_fluxes!(::AllSkyRadiationWithClearSkyDiagnostics, model)
    model.face_clear_flux .=
        model.face_clear_lw_flux .+ model.face_clear_sw_flux
    model.face_flux .= model.face_lw_flux .+ model.face_sw_flux
end

include("update_inputs.jl")
end # end module
