"""
    SurfaceTemperature

Abstract supertype for the sources of the surface temperature `T_sfc` [K] used
when computing surface conditions.

Subtypes:

  - [`AnalyticTemperature`](@ref): a function `(coordinates, params, t) -> T_sfc`.
    A spatially and temporally constant `T_sfc` is constructed as
    `AnalyticTemperature(Returns(T_sfc))`.
  - [`ExternalTemperature`](@ref): a time-varying input read from a cached `Field`.
  - [`SlabOceanTemperature`](@ref): prognostic, reads `Y.sfc.T`; carries the slab
    parameters.
  - [`CoupledTemperature`](@ref): a `Field` owned by an external driver (the coupler).

Each subtype extends `surface_temperature(temperature, Y, p, t_time)`, which
returns the value that [`update_surface_conditions!`](@ref) broadcasts across
the surface: either a `DataLayout` of per-cell temperatures, or the temperature
object itself when it must be evaluated per coordinate (see `resolve_T_sfc`).
"""
abstract type SurfaceTemperature end

Base.broadcastable(t::SurfaceTemperature) = tuple(t)

"""
    AnalyticTemperature(f)

A surface temperature given by `f(coordinates, surface_temp_params, t)` [K].

Used for the analytic SST formulas (zonally symmetric, RCEMIPII), time-varying
setups (e.g. GABLS), and spatially uniform constants
(`AnalyticTemperature(Returns(T))`). `f` is evaluated per coordinate inside the
surface-update broadcast, so it must be GPU-compatible; if the formula does not
depend on time, ignore the `t` argument.

# Fields

  - `f`: Callable `(coordinates, surface_temp_params, t) -> T_sfc` [K].

# Examples

```julia
temperature = AnalyticTemperature(Returns(300.0))
```
"""
struct AnalyticTemperature{F} <: SurfaceTemperature
    f::F
end

"""
    ExternalTemperature()

A surface temperature read from a time-varying external input [K].

`surface_temperature(::ExternalTemperature, Y, p, t_time)` evaluates
`p.external_forcing.surface_timevaryinginputs.ts` into
`p.external_forcing.surface_fields.ts`, so this temperature requires a setup
that populates `external_forcing.surface_fields` from a file carrying the `ts`
variable (e.g. `ForcingFromFile`).
"""
struct ExternalTemperature <: SurfaceTemperature end

"""
    SlabOceanTemperature{FT}(; depth_ocean, ρ_ocean, cp_ocean, q_flux, Q₀, ϕ₀)

Prognostic slab-ocean surface temperature, read from `Y.sfc.T` [K].

The only [`SurfaceTemperature`](@ref) that adds surface prognostic state
(`Y.sfc.T` and `Y.sfc.water`); its fields are the slab parameters used by
`surface_temp_tendency!` and by the conservation diagnostics. The optional
Q-flux is an idealized meridional profile of ocean heat-flux divergence.

# Fields

  - `depth_ocean = 40`: Ocean mixed-layer depth [m].
  - `ρ_ocean = 1020`: Ocean density [kg/m³].
  - `cp_ocean = 4184`: Ocean specific heat capacity [J/kg/K].
  - `q_flux = false`: Whether to apply the idealized Q-flux [-].
  - `Q₀ = -20`: Q-flux amplitude [W/m²].
  - `ϕ₀ = 16`: Q-flux meridional scale [degrees].
"""
@kwdef struct SlabOceanTemperature{FT} <: SurfaceTemperature
    depth_ocean::FT = 40        # ocean mixed-layer depth (m)
    ρ_ocean::FT = 1020          # ocean density (kg/m³)
    cp_ocean::FT = 4184         # ocean heat capacity (J/(kg·K))
    q_flux::Bool = false        # use Q-flux (horizontal ocean energy flux div.)
    Q₀::FT = -20                # Q-flux amplitude (W/m²)
    ϕ₀::FT = 16                 # Q-flux meridional scale (deg)
end

"""
    CoupledTemperature(field)

A surface temperature owned by an external driver (the coupler) [K].

The driver writes into `field` between steps; ClimaAtmos only reads from it.

# Fields

  - `field`: Surface `Field` of temperatures [K].
"""
struct CoupledTemperature{F} <: SurfaceTemperature
    field::F
end

# ============================================================================
# surface_temperature: dispatch from temperature type to the value used in
# update_surface_conditions!.
# ============================================================================

"""
    surface_temperature(temperature, Y, p, t_time)

Return the surface temperature [K] in the form consumed by
[`update_surface_conditions!`](@ref).

For an [`ExternalTemperature`](@ref), [`SlabOceanTemperature`](@ref), or
[`CoupledTemperature`](@ref) this is a `DataLayout` of per-cell values (the
external input is evaluated at `t_time` first). For an
[`AnalyticTemperature`](@ref) it is the temperature object itself, which
`resolve_T_sfc` evaluates per coordinate inside the surface broadcast.
"""
surface_temperature(t::AnalyticTemperature, Y, p, _) = t

function surface_temperature(::ExternalTemperature, Y, p, t_time)
    (; surface_fields, surface_timevaryinginputs) = p.external_forcing
    evaluate!(surface_fields.ts, surface_timevaryinginputs.ts, t_time)
    return Fields.field_values(surface_fields.ts)
end

surface_temperature(::SlabOceanTemperature, Y, p, _) =
    Fields.field_values(Y.sfc.T)

surface_temperature(t::CoupledTemperature, Y, p, _) =
    Fields.field_values(t.field)
