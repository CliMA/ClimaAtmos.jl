using Pkg
using NCDatasets
using RRTMGP
using CLIMAParameters: AbstractEarthParameterSet, Planet
using ClimaCore: Fields, DataLayouts

# TODO: Move this file to RRTMGP.jl, once the interface has been settled.
# It will be faster to do interface development in the same repo as experiment
# development, but, since this is just a user-friendly wrapper for RRTMGP.jl, we
# should move it there eventually.

"""
    field2array(field)

Extracts a view of a `ClimaCore` `Field`'s underlying array. Can be used to
simplify the process of getting and setting values in an `RRTMGPModel`; e.g.
```
    model.center_temperature .= field2array(center_temperature_field)
    field2array(face_flux_field) .= model.face_flux
```
"""
function field2array(field::Fields.Field)
    if sizeof(eltype(field)) != sizeof(eltype(parent(field)))
        f_axis_size = sizeof(eltype(parent(field))) ÷ sizeof(eltype(field))
        error("unable to use field2array because each Field element is \
               represented by $f_axis_size array elements (must be 1)")
    end
    return data2array(Fields.field_values(field))
end

const DataLayoutWithV =
    Union{DataLayouts.VF, DataLayouts.VIFH, DataLayouts.VIJFH}
const DataLayoutWithoutV =
    Union{DataLayouts.IF, DataLayouts.IFH, DataLayouts.IJF, DataLayouts.IJFH}
data2array(data::DataLayoutWithV) =
    reshape(parent(data), size(parent(data), 1), :)
data2array(data::DataLayoutWithoutV) = reshape(parent(data), :)
data2array(data::DataLayouts.AbstractData) =
    error("data2array not yet implemented for $(typeof(data).name.wrapper)")

"""
    rrtmgp_artifact(subfolder, file_name)

Constructs an `NCDataset` from the `RRTMGP.jl` artifact stored in
`RRTMGPReferenceData/<subfolder>/<file_name>`.
"""
function rrtmgp_artifact(subfolder, file_name)
    artifact_name = "RRTMGPReferenceData"
    artifacts_file =
        joinpath(dirname(dirname(pathof(RRTMGP))), "Artifacts.toml")
    data_folder = joinpath(
        Pkg.Artifacts.ensure_artifact_installed(artifact_name, artifacts_file),
        artifact_name,
    )
    return Dataset(joinpath(data_folder, subfolder, file_name), "r")
end

abstract type AbstractRadiationMode end
struct GrayRadiation <: AbstractRadiationMode end
struct ClearSkyRadiation <: AbstractRadiationMode end
struct AllSkyRadiation <: AbstractRadiationMode end
struct AllSkyRadiationWithClearSkyDiagnostics <: AbstractRadiationMode end

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
    FT = eltype(p)
    cₚ = FT(Planet.cp_d(params))
    R = FT(Planet.R_d(params))
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
    g = FT(Planet.grav(params))
    cₚ = FT(Planet.cp_d(params))
    R = FT(Planet.R_d(params))
    @. T = T⁺ + g / cₚ * (z⁺ - z)
    @. p = p⁺ * (T / T⁺)^(cₚ / R)
end

struct RRTMGPModel{R, I, B, L, P, S, V}
    radiation_mode::R
    interpolation::I
    bottom_extrapolation::B
    implied_values::Symbol
    add_isothermal_boundary_layer::Bool
    disable_longwave::Bool
    disable_shortwave::Bool
    lookups::L
    params::P
    max_threads::Int
    solver::S
    views::V # user-friendly views into the solver
end

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

To construct the `RRTMGPModel`, one must specify `center_pressure` and
`center_temperature`, or `face_pressure` and `face_temperature`, or all four
values. If only the values at cell centers are specified, the values at cell
faces are "implied values". Likewise, if only the values at faces are specified,
the values at centers are "implied values".

If the keyword argument `extension_nlay` is not 0, then, for every keyword
argument of the form `center_...`/`face_...`, one must also specify a
corresponding keyword argument `extension_center_...`/`extension_face_...`.

Every keyword argument that corresponds to an array of cell center or cell face
values can be specified as a scalar (corresponding to a constant value
throughout the atmosphere), or as a 1D array (corresponding to the values in
each column), or as a 2D array with a single row (corresponding to the values in
each level), or as the full 2D array specifying the value at every point.
Similarly, every keyword argument that corresponds to an array of values at the
top/bottom of the atmosphere can be specified as a scalar, or as the full 1D
array.

# Keyword Arguments
- `FT`: floating-point number type (performance with `Float32` is questionable)
- `DA`: array type (defaults to `CuArray` when a compatible GPU is available)
- `ncol`: number of vertical columns in the domain/extension
- `domain_nlay`: number of cells (layers) in the domain
- `extension_nlay`: number of cells (layers) in the extension
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
- `disable_longwave`: whether to ignore longwave fluxes
- `disable_shortwave`: whether to ignore shortwave fluxes
- `use_one_scalar_mode`: whether to use the one-scalar method for computing
  optical properties instead of the default two-stream method
- `use_pade_cloud_optics_mode`: whether to use PADE interpolation for computing
  cloud optical properties instead of the default LUT interpolation
- `use_global_means_for_well_mixed_gases`: whether to use a scalar value to
  represent the volume mixing ratio of each well-mixed gas (i.e., a gas that is
  not water vapor or ozone), instead of using an array that represents a
  spatially varying volume mixing ratio
- `add_isothermal_boundary_layer`: whether to add an isothermal boundary layer
  above the domain and extension (the pressure at the top of this layer is
  the minimum pressure supported by RRTMGP, while the temperature and volume
  mixing ratios in this layer are the same as in the layer below it)
- `max_threads`: the maximum number of GPU threads to use for `update_fluxes!`
- `center_pressure` and/or `face_pressure`: air pressure in Pa on cell centers
  and on cell faces (either one or both of these must be specified)
- `center_temperature` and/or `face_temperature`: air temperature in K on cell
  centers and on cell faces (if `center_pressure` is specified, then
  `center_temperature` must also be specified, and, if `face_pressure` is
  specified, then `face_temperature` must also be specified)
- `surface_temperature`: temperature of the surface in K (required)
- arguments only available when `!disable_longwave`:
    - `surface_emissivity`: longwave emissivity of the surface (required)
    - `top_of_atmosphere_lw_flux_dn`: incoming longwave radiation in W/m^2
      (assumed to be 0 by default)
- arguments only available when `!disable_shortwave`:
    - `direct_sw_surface_albedo`: direct shortwave albedo of the surface
      (required)
    - `diffuse_sw_surface_albedo`: diffuse shortwave albedo of the surface
      (required)
    - `solar_zenith_angle`: zenith angle of sun in radians (required)
    - `weighted_irradiance`: irradiance of sun in W/m^2 (required); the incoming
      direct shortwave radiation is given by
      `model.weighted_irradiance .* cos.(model.solar_zenith_angle)`
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
        - `center_cloud_boolean_mask`: a boolean at every cell center that
        indicates whether there is a cloud at that location (required)
        - `ice_roughness`: either 1, 2, or 3, with 3 corresponding to the
        roughest ice (required); this is a constant that can't be modified after
        the model is constructed
    - `latitude`: latitude in degrees (assumed to be 45 by default); used for
      computing the concentration of air in molecules/cm^2
- arguments only available when
  `requires_z(interpolation) || requires_z(bottom_extrapolation)`:
    - `center_z`: z-coordinate in m at cell centers
    - `face_z`: z-coordinate in m at cell faces
"""
function RRTMGPModel(
    params::AbstractEarthParameterSet;
    FT::Type{<:AbstractFloat} = Float64,
    DA::Type{<:AbstractArray} = RRTMGP.Device.array_type(),
    ncol::Int,
    domain_nlay::Int,
    extension_nlay::Int = 0,
    radiation_mode::AbstractRadiationMode = ClearSkyRadiation(),
    interpolation::AbstractInterpolation = NoInterpolation(),
    bottom_extrapolation::AbstractBottomExtrapolation = SameAsInterpolation(),
    disable_longwave::Bool = false,
    disable_shortwave::Bool = false,
    use_one_scalar_mode::Bool = false,
    use_pade_cloud_optics_mode::Bool = false,
    use_global_means_for_well_mixed_gases::Bool = false,
    add_isothermal_boundary_layer::Bool = false,
    max_threads::Int = 256,
    kwargs...,
)
    # turn kwargs into a Dict, so that values can be dynamically popped from it
    dict = Dict(kwargs)

    if disable_longwave && disable_shortwave
        error("either longwave or shortwave fluxes must be enabled")
    end
    if use_one_scalar_mode && !disable_shortwave
        @warn "upward shortwave fluxes are not computed when \
               use_one_scalar_mode is true"
    end
    if use_pade_cloud_optics_mode && (
        radiation_mode isa GrayRadiation ||
        radiation_mode isa ClearSkyRadiation
    )
        @warn "use_pade_cloud_optics_mode is ignored when using GrayRadiation \
               or ClearSkyRadiation"
    end
    if use_global_means_for_well_mixed_gases && radiation_mode isa GrayRadiation
        @warn "use_global_means_for_well_mixed_gases is ignored when using \
               GrayRadiation"
    end

    if (
        :center_pressure in keys(dict) &&
        :center_temperature in keys(dict) &&
        :face_pressure in keys(dict) &&
        :face_temperature in keys(dict)
    )
        if !(interpolation isa NoInterpolation)
            @warn "interpolation is ignored if both center and face pressures/\
                   temperatures are specified"
        end
        implied_values = :none
    elseif (
        :center_pressure in keys(dict) &&
        :center_temperature in keys(dict) &&
        !(:face_pressure in keys(dict)) &&
        !(:face_temperature in keys(dict))
    )
        if interpolation isa NoInterpolation
            error("interpolation cannot be NoInterpolation if only center \
                   pressures/temperatures are specified")
        end
        implied_values = :face
    elseif (
        !(:center_pressure in keys(dict)) &&
        !(:center_temperature in keys(dict)) &&
        :face_pressure in keys(dict) &&
        :face_temperature in keys(dict)
    )
        if interpolation isa NoInterpolation
            error("interpolation cannot be NoInterpolation if only face \
                   pressures/temperatures are specified")
        end
        implied_values = :center
    else
        error("please specify either center_pressure and center_temperature, \
               or face_pressure and face_temperature, or all four values")
    end

    if implied_values != :face
        if !(bottom_extrapolation isa SameAsInterpolation)
            @warn "bottom_extrapolation is ignored if face_pressure and \
                   face_temperature are specified"
        end
    end

    lookups = NamedTuple()
    views = []

    nlay = domain_nlay + extension_nlay + Int(add_isothermal_boundary_layer)
    op_symbol = use_one_scalar_mode ? :OneScalar : :TwoStream
    t = (views, domain_nlay, extension_nlay)

    if disable_longwave
        src_lw = flux_lw = fluxb_lw = bcs_lw = nothing
    else
        if radiation_mode isa GrayRadiation
            nbnd_lw = ngpt_lw = 1
        else
            ds_lw = rrtmgp_artifact("lookup_tables", "clearsky_lw.nc")
            lookup_lw, idx_gases =
                RRTMGP.LookUpTables.LookUpLW(ds_lw, Int, FT, DA)
            close(ds_lw)
            lookups = (; lookups..., lookup_lw, idx_gases)

            nbnd_lw = lookup_lw.n_bnd
            ngpt_lw = lookup_lw.n_gpt
            ngas = lookup_lw.n_gases

            if !(radiation_mode isa ClearSkyRadiation)
                ds_lw_cld = rrtmgp_artifact("lookup_tables", "cloudysky_lw.nc")
                lookup_lw_cld = RRTMGP.LookUpTables.LookUpCld(
                    ds_lw_cld,
                    Int,
                    FT,
                    DA,
                    !use_pade_cloud_optics_mode,
                )
                close(ds_lw_cld)
                lookups = (; lookups..., lookup_lw_cld)
            end
        end

        src_lw =
            RRTMGP.Sources.source_func_longwave(FT, ncol, nlay, op_symbol, DA)
        flux_lw = RRTMGP.Fluxes.FluxLW(ncol, nlay, FT, DA)
        fluxb_lw = radiation_mode isa GrayRadiation ? nothing :
            RRTMGP.Fluxes.FluxLW(ncol, nlay, FT, DA)
        set_and_save!(flux_lw.flux_up, "face_lw_flux_up", t...)
        set_and_save!(flux_lw.flux_dn, "face_lw_flux_dn", t...)
        set_and_save!(flux_lw.flux_net, "face_lw_flux", t...)
        if radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics
            flux_lw2 = RRTMGP.Fluxes.FluxLW(ncol, nlay, FT, DA)
            set_and_save!(flux_lw2.flux_up, "face_clear_lw_flux_up", t...)
            set_and_save!(flux_lw2.flux_dn, "face_clear_lw_flux_dn", t...)
            set_and_save!(flux_lw2.flux_net, "face_clear_lw_flux", t...)
        end

        sfc_emis = DA{FT}(undef, nbnd_lw, ncol)
        set_and_save!(sfc_emis, "surface_emissivity", t..., dict)
        name = "top_of_atmosphere_lw_flux_dn"
        if Symbol(name) in keys(dict)
            inc_flux = DA{FT}(undef, ncol, ngpt_lw)
            set_and_save!(transpose(inc_flux), name, t..., dict)
        else
            inc_flux = nothing
        end
        bcs_lw = RRTMGP.BCs.LwBCs(sfc_emis, inc_flux)
    end

    if disable_shortwave
        src_sw = flux_sw = fluxb_sw = bcs_sw = nothing
    else
        if radiation_mode isa GrayRadiation
            nbnd_sw = ngpt_sw = 1
        else
            ds_sw = rrtmgp_artifact("lookup_tables", "clearsky_sw.nc")
            lookup_sw, idx_gases =
                RRTMGP.LookUpTables.LookUpSW(ds_sw, Int, FT, DA)
            close(ds_sw)
            lookups = (; lookups..., lookup_sw, idx_gases)

            nbnd_sw = lookup_sw.n_bnd
            ngpt_sw = lookup_sw.n_gpt
            ngas = lookup_sw.n_gases

            if !(radiation_mode isa ClearSkyRadiation)
                ds_sw_cld = rrtmgp_artifact("lookup_tables", "cloudysky_sw.nc")
                lookup_sw_cld = RRTMGP.LookUpTables.LookUpCld(
                    ds_sw_cld,
                    Int,
                    FT,
                    DA,
                    !use_pade_cloud_optics_method,
                )
                close(ds_sw_cld)
                lookups = (; lookups..., lookup_sw_cld)
            end
        end

        src_sw =
            RRTMGP.Sources.source_func_shortwave(FT, ncol, nlay, op_symbol, DA)
        flux_sw = RRTMGP.Fluxes.FluxSW(ncol, nlay, FT, DA)
        fluxb_sw = radiation_mode isa GrayRadiation ? nothing :
            RRTMGP.Fluxes.FluxSW(ncol, nlay, FT, DA)
        set_and_save!(flux_sw.flux_up, "face_sw_flux_up", t...)
        set_and_save!(flux_sw.flux_dn, "face_sw_flux_dn", t...)
        set_and_save!(flux_sw.flux_dn_dir, "face_sw_direct_flux_dn", t...)
        set_and_save!(flux_sw.flux_net, "face_sw_flux", t...)
        if radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics
            flux_sw2 = RRTMGP.Fluxes.FluxSW(ncol, nlay, FT, DA)
            set_and_save!(flux_sw2.flux_up, "face_clear_sw_flux_up", t...)
            set_and_save!(flux_sw2.flux_dn, "face_clear_sw_flux_dn", t...)
            set_and_save!(
                flux_sw2.flux_dn_dir,
                "face_clear_sw_direct_flux_dn",
                t...,
            )
            set_and_save!(flux_sw2.flux_net, "face_clear_sw_flux", t...)
        end

        zenith = DA{FT}(undef, ncol)
        set_and_save!(zenith, "solar_zenith_angle", t..., dict)
        toa_flux = DA{FT}(undef, ncol)
        set_and_save!(toa_flux, "weighted_irradiance", t..., dict)
        sfc_alb_direct = DA{FT}(undef, nbnd_sw, ncol)
        set_and_save!(sfc_alb_direct, "direct_sw_surface_albedo", t..., dict)
        sfc_alb_diffuse = DA{FT}(undef, nbnd_sw, ncol)
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
            zenith,
            toa_flux,
            sfc_alb_direct,
            inc_flux_diffuse,
            sfc_alb_diffuse,
        )
    end

    if disable_longwave
        set_and_save!(flux_sw.flux_net, "face_flux", t...)
        if radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics
            set_and_save!(flux_sw2.flux_net, "face_clear_flux", t...)
        end
    elseif disable_shortwave
        set_and_save!(flux_lw.flux_net, "face_flux", t...)
        if radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics
            set_and_save!(flux_lw2.flux_net, "face_clear_flux", t...)
        end
    else
        set_and_save!(similar(flux_lw.flux_net), "face_flux", t...)
        if radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics
            set_and_save!(similar(flux_lw2.flux_net), "face_clear_flux", t...)
        end
        if !(radiation_mode isa GrayRadiation)
            @assert lookup_lw.n_gases == lookup_sw.n_gases
            @assert lookup_lw.p_ref_min == lookup_sw.p_ref_min
        end
    end

    p_lay = DA{FT}(undef, nlay, ncol)
    p_lev = DA{FT}(undef, nlay + 1, ncol)
    t_lay = DA{FT}(undef, nlay, ncol)
    t_lev = DA{FT}(undef, nlay + 1, ncol)
    if implied_values != :center
        set_and_save!(p_lay, "center_pressure", t..., dict)
        set_and_save!(t_lay, "center_temperature", t..., dict)
    end
    if implied_values != :face
        set_and_save!(p_lev, "face_pressure", t..., dict)
        set_and_save!(t_lev, "face_temperature", t..., dict)
    end
    t_sfc = DA{FT}(undef, ncol)
    set_and_save!(t_sfc, "surface_temperature", t..., dict)

    if radiation_mode isa GrayRadiation
        z_lev = DA{FT}(undef, nlay + 1, ncol) # TODO: z_lev required but unused

        # lapse_rate is a constant, so don't use set_and_save! to get it
        if !(:lapse_rate in keys(dict))
            throw(UndefKeywordError(:lapse_rate))
        end
        α = pop!(dict, :lapse_rate)
        if !(α isa Real)
            error("lapse_rate must be a Real")
        end

        d0 = DA{FT}(undef, ncol)
        set_and_save!(d0, "optical_thickness_parameter", t..., dict)
        as = RRTMGP.AtmosphericStates.GrayAtmosphericState(
            p_lay,
            p_lev,
            t_lay,
            t_lev,
            z_lev,
            t_sfc,
            α,
            d0,
            nlay,
            ncol,
        )
    else
        if !(:latitude in keys(dict))
            lon = lat = nothing
        else
            lon = DA{FT}(undef, ncol) # TODO: lon required but unused
            lat = DA{FT}(undef, ncol)
            set_and_save!(lat, "latitude", t..., dict)
        end

        vmr_str = "volume_mixing_ratio_"
        gas_names = filter(
            gas_name -> !(gas_name in ("h2o", "h2o_frgn", "h2o_self", "o3")),
            keys(idx_gases),
        )
        # TODO: This gives the wrong types for CUDA 3.4 and above.
        # gm = use_global_means_for_well_mixed_gases
        # vmr = RRTMGP.Vmrs.init_vmr(ngas, nlay, ncol, FT, DA; gm)
        if use_global_means_for_well_mixed_gases
            vmr = RRTMGP.Vmrs.VmrGM(
                DA{FT}(undef, nlay, ncol),
                DA{FT}(undef, nlay, ncol),
                DA{FT}(undef, ngas),
            )
            vmr.vmr .= 0 # TODO: do we need this?
            set_and_save!(vmr.vmr_h2o, "center_$(vmr_str)h2o", t..., dict)
            set_and_save!(vmr.vmr_o3, "center_$(vmr_str)o3", t..., dict)
            for gas_name in gas_names
                gas_view = view(vmr.vmr, idx_gases[gas_name])
                set_and_save!(gas_view, "$vmr_str$gas_name", t..., dict)
            end
        else
            vmr = RRTMGP.Vmrs.Vmr(DA{FT}(undef, nlay, ncol, ngas))
            for gas_name in ["h2o", "o3", gas_names...]
                gas_view = view(vmr.vmr, :, :, idx_gases[gas_name])
                set_and_save!(gas_view, "center_$vmr_str$gas_name", t..., dict)
            end
        end

        if radiation_mode isa ClearSkyRadiation
            cld_r_eff_liq = cld_r_eff_ice = nothing
            cld_path_liq = cld_path_ice = cld_mask = nothing
            ice_rgh = 1
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
            cld_mask = DA{Bool}(undef, nlay, ncol)
            set_and_save!(cld_mask, "center_cloud_boolean_mask", t..., dict)

            # ice_roughness is a constant, so don't use set_and_save! to get it
            if !(:ice_roughness in keys(dict))
                throw(UndefKeywordError(:ice_roughness))
            end
            ice_rgh = pop!(dict, :ice_roughness)
            if !(ice_rgh in (1, 2, 3))
                error("ice_roughness must be either 1, 2, or 3")
            end
        end

        as = RRTMGP.AtmosphericStates.AtmosphericState(
            lon,
            lat,
            p_lay,
            p_lev,
            t_lay,
            t_lev,
            t_sfc,
            DA{FT}(undef, nlay, ncol), # col_dry array is for internal use only
            vmr,
            cld_r_eff_liq,
            cld_r_eff_ice,
            cld_path_liq,
            cld_path_ice,
            cld_mask,
            ice_rgh,
            nlay,
            ncol,
            ngas,
        )
    end

    op_type =
        use_one_scalar_mode ? RRTMGP.Optics.OneScalar : RRTMGP.Optics.TwoStream
    solver = RRTMGP.RTE.Solver(
        as,
        op_type(FT, ncol, nlay, DA),
        src_lw,
        src_sw,
        bcs_lw,
        bcs_sw,
        fluxb_lw,
        fluxb_sw,
        flux_lw,
        flux_sw,
    )

    if requires_z(interpolation) || requires_z(bottom_extrapolation)
        z_lay = DA{FT}(undef, nlay, ncol)
        set_and_save!(z_lay, "center_z", t..., dict)
        z_lev = DA{FT}(undef, nlay + 1, ncol)
        set_and_save!(z_lev, "face_z", t..., dict)
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
        implied_values,
        add_isothermal_boundary_layer,
        disable_longwave,
        disable_shortwave,
        lookups,
        params,
        max_threads,
        solver,
        NamedTuple(views),
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

function set_and_save!(
    array,
    name,
    views,
    domain_nlay,
    extension_nlay,
    dict = nothing,
)
    domain_symbol = Symbol(name)
    extension_symbol = Symbol("extension_$name")

    if isnothing(dict)
        domain_value = NaN
    else
        if !(domain_symbol in keys(dict))
            throw(UndefKeywordError(domain_symbol))
        end
        domain_value = pop!(dict, domain_symbol)
    end

    if startswith(name, "center_") || startswith(name, "face_")
        domain_range = startswith(name, "center_") ? (1:domain_nlay) :
            (1:(domain_nlay + 1))
        domain_view = view(array, domain_range, :)
        set_array!(domain_view, domain_value, domain_symbol)
        push!(views, (domain_symbol, domain_view))

        if extension_nlay > 0
            if isnothing(dict)
                extension_value = NaN
            else
                if !(extension_symbol in keys(dict))
                    if domain_value isa Real
                        extension_value = domain_value
                    else
                        throw(UndefKeywordError(extension_symbol))
                    end
                end
                extension_value = pop!(dict, extension_symbol)
            end

            extension_range = startswith(name, "center_") ?
                ((domain_nlay + 1):(domain_nlay + extension_nlay)) :
                ((domain_nlay + 2):(domain_nlay + extension_nlay + 1))
            extension_view = view(array, extension_range, :)
            set_array!(extension_view, extension_value, extension_symbol)
            push!(views, (extension_symbol, extension_view))
        end
    else
        set_array!(array, domain_value, domain_symbol)
        push!(views, (domain_symbol, array))
    end
end

"""
    update_fluxes!(model)

Updates the fluxes in the `RRTMGPModel` based on its internal state. Returns the
net flux at cell faces in the domain, `model.face_flux`. The full set of fluxes
available in the model after calling this function is
- `face_flux`
- `face_lw_flux`, `face_lw_flux_dn`, `face_lw_flux_up` (if `!disable_longwave`)
- `face_sw_flux`, `face_sw_flux_dn`, `face_sw_flux_up`, `face_sw_direct_flux_dn`
  (if `!disable_shortwave`)
If `radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics`, the set of
available fluxes also includes
- `face_clear_flux`
- `face_clear_lw_flux`, `face_clear_lw_flux_dn`, `face_clear_lw_flux_up` (if
  `!disable_longwave`)
- `face_clear_sw_flux`, `face_clear_sw_flux_dn`, `face_clear_sw_flux_up`,
  `face_clear_sw_direct_flux_dn` (if `!disable_shortwave`)
If `extension_nlay > 0`, the set of available fluxes also includes all of the
aforementioned values prefixed by `extension_`, corresponding to the values at
cell faces in the extension.
"""
function update_fluxes!(model)
    model.implied_values != :none && update_implied_values!(model)
    model.add_isothermal_boundary_layer && update_boundary_layer!(model)
    clip_values!(model)
    update_concentrations!(model.radiation_mode, model)
    !model.disable_longwave && update_lw_fluxes!(model.radiation_mode, model)
    !model.disable_shortwave && update_sw_fluxes!(model.radiation_mode, model)
    !(model.disable_longwave || model.disable_shortwave) &&
    update_net_fluxes!(model.radiation_mode, model)
    return model.face_flux
end

get_p_min(model) = get_p_min(model.solver.as, model.lookups)
get_p_min(as::RRTMGP.AtmosphericStates.GrayAtmosphericState, lookups) =
    zero(eltype(as.p_lay))
get_p_min(as::RRTMGP.AtmosphericStates.AtmosphericState, lookups) =
    lookups[1].p_ref_min

function update_implied_values!(model)
    (; p_lay, p_lev, t_lay, t_lev, t_sfc) = model.solver.as
    nlay = size(p_lay, 1) - Int(model.add_isothermal_boundary_layer)
    if requires_z(model.interpolation) || requires_z(model.bottom_extrapolation)
        z_lay = parent(model.center_z)
        z_lev = parent(model.face_z)
    end
    if model.implied_values == :center
        mode = model.interpolation
        outs = requires_z(mode) ? (p_lay, t_lay, z_lay) : (p_lay, t_lay)
        ins = requires_z(mode) ? (p_lev, t_lev, z_lev) : (p_lev, t_lev)
        update_views(interp!, mode, outs, ins, (), 1:nlay, 1:nlay, 2:(nlay + 1))
    else # model.implied_values == :face
        mode = model.interpolation
        outs = requires_z(mode) ? (p_lev, t_lev, z_lev) : (p_lev, t_lev)
        ins = requires_z(mode) ? (p_lay, t_lay, z_lay) : (p_lay, t_lay)
        update_views(interp!, mode, outs, ins, (), 2:nlay, 1:(nlay - 1), 2:nlay)
        others = (t_sfc, model.params)
        update_views(extrap!, mode, outs, ins, others, nlay + 1, nlay, nlay - 1)
        mode = model.bottom_extrapolation isa SameAsInterpolation ?
            model.interpolation : model.bottom_extrapolation
        outs = requires_z(mode) ? (p_lev, t_lev, z_lev) : (p_lev, t_lev)
        ins = requires_z(mode) ? (p_lay, t_lay, z_lay) : (p_lay, t_lay)
        update_views(extrap!, mode, outs, ins, others, 1, 1, 2)
    end
end
update_views(f, mode, outs, ins, others, out_range, in_range1, in_range2) = f(
    mode,
    map(out -> view(out, out_range, :), outs)...,
    map(in -> view(in, in_range1, :), ins)...,
    map(in -> view(in, in_range2, :), ins)...,
    others...,
)

function update_boundary_layer!(model)
    as = model.solver.as
    p_min = get_p_min(model)
    @views as.p_lay[end, :] .= (as.p_lev[end - 1, :] .+ p_min) ./ 2
    @views as.p_lev[end, :] .= p_min
    @views as.t_lay[end, :] .= as.t_lev[end - 1, :]
    @views as.t_lev[end, :] .= as.t_lev[end - 1, :]
    update_boundary_layer_vmr!(model.radiation_mode, as)
end
update_boundary_layer_vmr!(::GrayRadiation, as) = nothing
update_boundary_layer_vmr!(radiation_mode, as) =
    update_boundary_layer_vmr!(as.vmr)
function update_boundary_layer_vmr!(vmr::RRTMGP.Vmrs.VmrGM)
    @views vmr.vmr_h2o[end, :] .= vmr.vmr_h2o[end - 1, :]
    @views vmr.vmr_o3[end, :] .= vmr.vmr_o3[end - 1, :]
end
update_boundary_layer_vmr!(vmr::RRTMGP.Vmrs.Vmr) =
    @views vmr.vmr[end, :, :] .= vmr.vmr[end - 1, :, :]

function clip_values!(model)
    (; p_lay, p_lev) = model.solver.as
    if !(model.radiation_mode isa GrayRadiation)
        (; vmr_h2o) = model.solver.as.vmr
        @. vmr_h2o = max(vmr_h2o, 0)
    end
    p_min = get_p_min(model)
    @. p_lay = max(p_lay, p_min)
    @. p_lev = max(p_lev, p_min)
end
update_concentrations!(::GrayRadiation, model) = nothing
update_concentrations!(radiation_mode, model) = RRTMGP.Optics.compute_col_dry!(
    model.solver.as.p_lev,
    model.solver.as.col_dry,
    model.params,
    get_vmr_h2o(model.solver.as.vmr, model.lookups.idx_gases),
    model.solver.as.lat,
    model.max_threads,
)
get_vmr_h2o(vmr::RRTMGP.Vmrs.VmrGM, idx_gases) = vmr.vmr_h2o
get_vmr_h2o(vmr::RRTMGP.Vmrs.Vmr, idx_gases) =
    view(vmr.vmr, :, :, idx_gases["h2o"])

update_lw_fluxes!(::GrayRadiation, model) =
    RRTMGP.RTESolver.solve_lw!(model.solver, model.max_threads)
update_lw_fluxes!(::ClearSkyRadiation, model) = RRTMGP.RTESolver.solve_lw!(
    model.solver,
    model.max_threads,
    model.lookups.lookup_lw,
)
update_lw_fluxes!(::AllSkyRadiation, model) = RRTMGP.RTESolver.solve_lw!(
    model.solver,
    model.max_threads,
    model.lookups.lookup_lw,
    model.lookups.lookup_lw_cld,
)
function update_lw_fluxes!(::AllSkyRadiationWithClearSkyDiagnostics, model)
    RRTMGP.RTESolver.solve_lw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_lw,
    )
    parent(model.face_clear_lw_flux_up) .= parent(model.face_lw_flux_up)
    parent(model.face_clear_lw_flux_dn) .= parent(model.face_lw_flux_dn)
    parent(model.face_clear_lw_flux) .= parent(model.face_lw_flux)
    RRTMGP.RTESolver.solve_lw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_lw,
        model.lookups.lookup_lw_cld,
    )
end

update_sw_fluxes!(::GrayRadiation, model) =
    RRTMGP.RTESolver.solve_sw!(model.solver, model.max_threads)
update_sw_fluxes!(::ClearSkyRadiation, model) = RRTMGP.RTESolver.solve_sw!(
    model.solver,
    model.max_threads,
    model.lookups.lookup_sw,
)
update_sw_fluxes!(::AllSkyRadiation, model) = RRTMGP.RTESolver.solve_sw!(
    model.solver,
    model.max_threads,
    model.lookups.lookup_sw,
    model.lookups.lookup_sw_cld,
)
function update_sw_fluxes!(::AllSkyRadiationWithClearSkyDiagnostics, model)
    RRTMGP.RTESolver.solve_sw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_sw,
    )
    parent(model.face_clear_sw_flux_up) .= parent(model.face_sw_flux_up)
    parent(model.face_clear_sw_flux_dn) .= parent(model.face_sw_flux_dn)
    parent(model.face_clear_sw_direct_flux_dn) .=
        parent(model.face_sw_direct_flux_dn)
    parent(model.face_clear_sw_flux) .= parent(model.face_sw_flux)
    RRTMGP.RTESolver.solve_sw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_sw,
        model.lookups.lookup_sw_cld,
    )
end

update_net_fluxes!(radiation_mode, model) =
    parent(model.face_flux) .=
        parent(model.face_lw_flux) .+ parent(model.face_sw_flux)
function update_net_fluxes!(::AllSkyRadiationWithClearSkyDiagnostics, model)
    parent(model.face_clear_flux) .=
        parent(model.face_clear_lw_flux) .+ parent(model.face_clear_sw_flux)
    parent(model.face_flux) .=
        parent(model.face_lw_flux) .+ parent(model.face_sw_flux)
end

# Overriding the definition of optical depth for GrayRadiation.
function τ_lw_gray(p, pꜜ, pꜛ, p₀, τ₀)
    FT = eltype(p)
    f = FT(0.2)
    return τ₀ * (f / p₀ + 4 * (1 - f) / p₀ * (p / p₀)^3) * (pꜜ - pꜛ)
end
τ_sw_gray(p, pꜜ, pꜛ, p₀, τ₀) = 2 * τ₀ * (p / p₀) / p₀ * (pꜜ - pꜛ)
# Note: the original value for both of these functions was
#     abs(α * τ₀ * (p / p₀)^α / p * (pꜛ - pꜜ)),
# where α is the lapse rate and τ₀ is the optical thickness parameter.

import RRTMGP.Optics: compute_optical_props_kernel!
function compute_optical_props_kernel!(
    op::RRTMGP.Optics.AbstractOpticalProps{FT},
    as::RRTMGP.AtmosphericStates.GrayAtmosphericState{FT},
    glaycol,
    source::RRTMGP.Sources.AbstractSourceLW{FT},
) where {FT <: AbstractFloat}
    compute_optical_props_kernel_lw!(op, as, glaycol)
    RRTMGP.Optics.compute_sources_gray_kernel!(source, as, glaycol)
end
function compute_optical_props_kernel_lw!(
    op::RRTMGP.Optics.AbstractOpticalProps{FT},
    as::RRTMGP.AtmosphericStates.GrayAtmosphericState{FT},
    glaycol,
) where {FT <: AbstractFloat}
    glay, gcol = glaycol
    (; p_lay, p_lev, d0) = as
    @inbounds op.τ[glay, gcol] = τ_lw_gray(
        p_lay[glay, gcol],
        p_lev[glay, gcol],
        p_lev[glay + 1, gcol],
        p_lev[1, gcol],
        d0[gcol],
    )
    if op isa RRTMGP.Optics.TwoStream
        op.ssa[glaycol...] = FT(0)
        op.g[glaycol...] = FT(0)
    end
end
function compute_optical_props_kernel!(
    op::RRTMGP.Optics.AbstractOpticalProps{FT},
    as::RRTMGP.AtmosphericStates.GrayAtmosphericState{FT},
    glaycol,
) where {FT <: AbstractFloat}
    glay, gcol = glaycol
    (; p_lay, p_lev) = as
    @inbounds op.τ[glay, gcol] = τ_sw_gray(
        p_lay[glay, gcol],
        p_lev[glay, gcol],
        p_lev[glay + 1, gcol],
        p_lev[1, gcol],
        FT(0.22), # hardcode the value of τ₀ for shortwave gray radiation
    )
    if op isa RRTMGP.Optics.TwoStream
        op.ssa[glaycol...] = FT(0)
        op.g[glaycol...] = FT(0)
    end
end
