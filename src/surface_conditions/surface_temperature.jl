"""
    SurfaceTemperature

Abstract supertype for ways of obtaining the surface temperature `T_sfc` used
when computing surface conditions. Concrete subtypes:

  - [`AnalyticTemperature`](@ref): a function `(coordinates, params, t) -> T_sfc`. A
    spatially and temporally constant `T_sfc` is simply constructed as
    `AnalyticTemperature(Returns(T_sfc))`.
  - [`ExternalTemperature`](@ref): time-varying input read from a cached Field.
  - [`SlabOceanTemperature`](@ref): prognostic, reads `Y.sfc.T`. Carries slab params.
  - [`CoupledTemperature`](@ref): a Field owned by an external driver (e.g. coupler).

`surface_temperature(t, Y, p, t_time)` produces the value(s) that
[`update_surface_conditions!`](@ref) will broadcast across the surface.
"""
abstract type SurfaceTemperature end

Base.broadcastable(t::SurfaceTemperature) = tuple(t)

"""
    AnalyticTemperature(f)

A surface temperature given by `f(coordinates, params, t)`. Used for the
analytic SST formulas (zonally-symmetric, RCEMIPII), time-varying setups
(e.g. GABLS), and spatially-uniform constants (`AnalyticTemperature(Returns(T))`).
`f` is broadcast per-coordinate inside the surface update; if the formula does
not depend on time, simply ignore the `t` argument.
"""
struct AnalyticTemperature{F} <: SurfaceTemperature
    f::F
end

"""
    ExternalTemperature()

A surface temperature read from a time-varying external input.
`surface_temperature(::ExternalTemperature, Y, p, t)` evaluates the field
from `p.external_forcing.surface_inputs.ts` (driven by
`surface_timevaryinginputs.ts`), so this temperature is only valid when the
setup populates `external_forcing.surface_inputs` (e.g. `InterpolatedColumnProfile`).
"""
struct ExternalTemperature <: SurfaceTemperature end

"""
    SlabOceanTemperature{FT}()

Prognostic slab-ocean surface temperature, read from `Y.sfc.T`. Carries the
slab-ocean parameters used by `surface_temp_tendency!` and conservation
diagnostics.
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

A surface temperature owned by an external driver (the coupler). The driver
writes into `field` between steps; ClimaAtmos reads from it.
"""
struct CoupledTemperature{F} <: SurfaceTemperature
    field::F
end

"""
    ARMVARANALTimeVaryingSST()

Surface temperature read from ARM VARANAL forcing data.
`surface_temperature(::ARMVARANALTimeVaryingSST, Y, p, t)` evaluates the
time-varying interpolator from `p.external_forcing.T_sfc_tvi` into
`p.external_forcing.surface_ts`.
"""
struct ARMVARANALTimeVaryingSST <: SurfaceTemperature end

# ============================================================================
# surface_temperature: dispatch from temperature type to the value used in
# update_surface_conditions!. Returns either the temperature struct itself
# (AnalyticTemperature, evaluated per-coordinate downstream via resolve_T_sfc)
# or a DataLayout of per-cell values that broadcasts across the surface.
# ============================================================================

# AnalyticTemperature must be evaluated per-coordinate downstream; we return
# the temperature struct so update_surface_conditions! can dispatch on it.
surface_temperature(t::AnalyticTemperature, Y, p, _) = t

function surface_temperature(::ExternalTemperature, Y, p, t_time)
    (; surface_inputs, surface_timevaryinginputs) = p.external_forcing
    evaluate!(surface_inputs.ts, surface_timevaryinginputs.ts, t_time)
    return Fields.field_values(surface_inputs.ts)
end

surface_temperature(::SlabOceanTemperature, Y, p, _) =
    Fields.field_values(Y.sfc.T)

surface_temperature(t::CoupledTemperature, Y, p, _) =
    Fields.field_values(t.field)

function surface_temperature(::ARMVARANALTimeVaryingSST, Y, p, t_time)
    (; T_sfc_tvi, surface_ts) = p.external_forcing
    evaluate!(surface_ts, T_sfc_tvi, t_time)
    return Fields.field_values(surface_ts)
end
