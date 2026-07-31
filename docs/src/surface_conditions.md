# Surface Conditions

The lower boundary is where the atmosphere exchanges momentum, heat, and
moisture with whatever lies beneath it (ocean, land, sea ice, or an idealized
slab). ClimaAtmos collects everything controlling this boundary into one object,
`AtmosSurface`, stored as `atmos.surface`. It is read at each step to fill
`p.precomputed.sfc_conditions`, the surface fluxes and values consumed as
boundary conditions by the dynamical core, radiation, and turbulence schemes.

The [User Guide](#User-Guide) covers the options and how to choose; the
[Developer Guide](#Developer-Guide) covers the design, data flow, and how to
extend or debug it.

## User Guide

### The four knobs

[`AtmosSurface`](@ref ClimaAtmos.AtmosSurface) has four fields, each with one
purpose:

  - `flux_scheme`: computes turbulent fluxes from air–surface differences in
    temperature, humidity etc.
  - `temperature`: sets the surface temperature `T_sfc`.
  - `boundary_overrides`: pins surface properties at user-specified values.
  - `surface_albedo`: sets the shortwave reflectivity seen by radiation (distinct
    direct and diffuse components).

Set these directly when building a model, or let them be chosen by a
[setup](@ref "Setups") or by [YAML keys](#Configuring-from-YAML).

### Flux scheme (`flux_scheme`)

The closure turning the surface–to–lowest-level difference into turbulent
fluxes of momentum, heat, and moisture:

  - **[`MoninObukhov`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov)**:
    Monin–Obukhov Similarity Theory (MOST); fluxes follow from roughness length and
    near-surface stability. Heat fluxes (`shf`/`lhf` or `θ_flux`/`q_flux`) or
    `ustar` may instead be prescribed (common for LES). For *time-varying*
    prescribed fluxes, pass `fluxes` as a callable
    `(t, FT) -> HeatFluxes/θAndQFluxes`; it is resolved once per update (e.g.
    TRMM_LBA's diurnal SHF/LHF), while `z0`/`ustar` stay constant.
  - **[`ExchangeCoefficients`](@ref ClimaAtmos.SurfaceConditions.ExchangeCoefficients)**:
    bulk fluxes with fixed `Cd`/`Ch`; simpler and cheaper, for idealized constant
    exchange coefficients (rather than coefficients determined by MOST).
  - **`nothing`**: no atmos-side computation; an external driver supplies the
    conditions (see [Coupling](#Coupling-to-an-external-driver)).

### Temperature source (`temperature`)

What `T_sfc` is; the flux scheme then uses it (and surface humidity) for the
air–surface gradients:

  - **[`AnalyticTemperature`](@ref ClimaAtmos.SurfaceConditions.AnalyticTemperature)**:
    `T_sfc = f(coordinates, params, t)`, per point. Covers a uniform constant
    (`AnalyticTemperature(Returns(FT(300)))`), a zonally-symmetric SST, or a
    time-varying profile (e.g., GABLS).
  - **[`SlabOceanTemperature`](@ref ClimaAtmos.SurfaceConditions.SlabOceanTemperature)**:
    *prognostic*; `T_sfc` read from `Y.sfc.T`, evolved by a slab-ocean energy
    budget. The only type that adds a prognostic state.
  - **[`ExternalTemperature`](@ref ClimaAtmos.SurfaceConditions.ExternalTemperature)**:
    read from a time-varying external input; valid only when the setup populates
    `external_forcing.surface_fields`.
  - **[`CoupledTemperature`](@ref ClimaAtmos.SurfaceConditions.CoupledTemperature)**:
    read from a `Field` the coupler writes into (see
    [Coupling](#Coupling-to-an-external-driver)).

!!! note "Constant temperature"

    There is no dedicated constant type. Use
    `AnalyticTemperature(Returns(FT(300)))`, wrapping the value in `FT(...)` to
    keep the broadcast type-stable.

### Boundary overrides (`boundary_overrides`)

By default, surface values come from physics (pressure hydrostatically
extrapolated, humidity saturated at `T_sfc`, zero winds, unit gustiness/moisture
availability).
[`SurfaceBoundaryOverrides`](@ref ClimaAtmos.SurfaceConditions.SurfaceBoundaryOverrides)
pins a value to a fixed override; each field defaults to `nothing` (use the
physical default). Currently only `q_vap`, `u`, `v`, and `gustiness` are
consumed by `surface_state_to_conditions`; the `p` and `beta` fields are
accepted and stored but not yet applied (the surface density comes from
`SurfaceFluxes.surface_density`). Many idealized setups nevertheless set `p`
for future use.

### Albedo (`surface_albedo`)

Sets the shortwave reflectivity passed to the radiation scheme. Three models:

  - **[`ConstantAlbedo`](@ref ClimaAtmos.ConstantAlbedo)**: a single value applied
    to both direct and diffuse shortwave.
  - **[`RegressionFunctionAlbedo`](@ref ClimaAtmos.RegressionFunctionAlbedo)**: the
    Jin et al. (2011) ocean parameterization, a solar-zenith-angle-dependent
    *direct* albedo plus a separate *diffuse* albedo, with wind-speed-dependent
    surface roughness.
  - **[`CouplerAlbedo`](@ref ClimaAtmos.CouplerAlbedo)**: albedo supplied by an
    external driver (the coupler).

**Direct vs. diffuse** The model carries distinct
`direct_sw_surface_albedo` and `diffuse_sw_surface_albedo` fields.
`ConstantAlbedo` sets them equal, `RegressionFunctionAlbedo` computes them
separately.

**Spectral** Both atmosphere-side models write a single value across every shortwave band,
and the [`RegressionFunctionAlbedo`](@ref ClimaAtmos.RegressionFunctionAlbedo)
scheme treats the refractive index as wavelength-independent. The RRTMGP
interface arrays are band-resolved (`(nbnd_sw, ncol)`), so per-band albedo is a
supported extension point but it would require a model that fills bands with
distinct values.

**Longwave surface reflectivity** Albedo is shortwave-only; longwave surface reflectivity is handled
separately through `surface_emissivity`.

See the [Ocean Surface Albedo](@ref "Ocean Surface Albedo") page for the
Jin (2011) [`RegressionFunctionAlbedo`](@ref ClimaAtmos.RegressionFunctionAlbedo) formulation.

### Choosing

`flux_scheme` and `temperature` are independent axes, and you set **both** (the
other two fields take defaults). Each row below is a compatible *pair*, not an
either/or:

| If you want…                                      | `flux_scheme`                                                                                                                                                                  | `temperature`                                                                       |
|:------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |:----------------------------------------------------------------------------------- |
| Stability-dependent fluxes over a prescribed SST  | [`MoninObukhov(; z0 = …)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov)                                                                                                     | [`AnalyticTemperature(…)`](@ref ClimaAtmos.SurfaceConditions.AnalyticTemperature)   |
| Fixed-coefficient bulk fluxes                     | [`ExchangeCoefficients(; Cd, Ch)`](@ref ClimaAtmos.SurfaceConditions.ExchangeCoefficients)                                                                                     | [`AnalyticTemperature(…)`](@ref ClimaAtmos.SurfaceConditions.AnalyticTemperature)   |
| Prescribed heat fluxes (constant or time-varying) | [`MoninObukhov(; z0, shf, lhf)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov) or [`MoninObukhov(; z0, fluxes = (t,FT)->…)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov) | [`AnalyticTemperature(…)`](@ref ClimaAtmos.SurfaceConditions.AnalyticTemperature)   |
| An interactive slab ocean surface                 | [`MoninObukhov(…)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov)                                                                                                            | [`SlabOceanTemperature(…)`](@ref ClimaAtmos.SurfaceConditions.SlabOceanTemperature) |
| Surface temperature from data                     | [`MoninObukhov(…)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov)                                                                                                            | [`ExternalTemperature(…)`](@ref ClimaAtmos.SurfaceConditions.ExternalTemperature)   |
| Coupler owns the surface (atmos skips fluxes)     | `nothing`                                                                                                                                                                      | unused; coupler writes `sfc_conditions`                                             |
| Coupler sets SST; atmos computes fluxes           | [`MoninObukhov(…)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov)                                                                                                            | [`CoupledTemperature(field)`](@ref ClimaAtmos.SurfaceConditions.CoupledTemperature) |

!!! note "Prescribed fluxes do not use MOST"

    When you set `shf`/`lhf` (or `θ_flux`/`q_flux`), those fluxes are used **as
    prescribed**: MOST does not compute them. They appear under `MoninObukhov`
    only because the prescribed-flux path currently lives inside that type (a
    historical conflation; see the Developer Guide). The required `z0` is used
    solely for the *momentum* closure, and only when `ustar` is not also
    prescribed: when both fluxes and `ustar` are given (as in every idealized
    LES setup), MOST does nothing and the surface is fully prescribed.

### Setting the surface in a runscript

Build an [`AtmosSurface`](@ref ClimaAtmos.AtmosSurface) and hand it to
`AtmosModel`. For example, Monin–Obukhov fluxes over a fixed 290 K sea surface
with a constant albedo:

```julia
import ClimaAtmos as CA
import ClimaAtmos.SurfaceConditions as SC
FT = Float64

surface = CA.AtmosSurface(;
    flux_scheme = SC.MoninObukhov(; z0 = FT(1e-4)),
    temperature = SC.AnalyticTemperature(Returns(FT(290))),
    surface_albedo = CA.ConstantAlbedo{FT}(; α = FT(0.07)),
    # boundary_overrides defaults to all-`nothing` (physical defaults)
)

model = CA.AtmosModel(; surface, microphysics_model = CA.DryModel())
```

Omitted fields take their defaults. You can also pass the surface fields
directly to `AtmosModel` (`CA.AtmosModel(; flux_scheme = …, temperature = …)`),
which assembles the `AtmosSurface` for you. To swap in an interactive slab
ocean use `temperature = SC.SlabOceanTemperature{FT}()`; for prescribed heat
fluxes, `flux_scheme = SC.MoninObukhov(; z0 = FT(1e-4), shf = …, lhf = …)`.

#### File-driven surface

`ForcingFromFile` assembles the existing independent `temperature` and
`flux_scheme` components described above (`ExternalTemperature` from the
file's `ts`, with an interactive `MoninObukhov` flux scheme) rather than introducing
another surface-policy hierarchy.

### Configuring from YAML

Three of the four `AtmosSurface` fields are YAML-configurable (resolved by
`AtmosSurface(::AtmosConfig, params, FT; setup_type)`). Setup-provided pieces
take precedence for `flux_scheme` and `boundary_overrides`; for `temperature`,
`prognostic_surface: SlabOceanSST` overrides the setup, while `PrescribedSST`
falls back to the setup piece:

  - `surface_setup` sets [`flux_scheme`](@ref "Flux scheme (flux_scheme)"):
    `"DefaultExchangeCoefficients"` (default), `"DefaultMoninObukhov"`, or
    `"PrescribedSurface"` (→ `nothing`).
  - `prognostic_surface` sets [`temperature`](@ref "Temperature source (temperature)"):
    `"PrescribedSST"` (default) or `"SlabOceanSST"` (→ `SlabOceanTemperature`).
  - `albedo_model` sets [`surface_albedo`](@ref "Albedo (surface_albedo)"):
    `"ConstantAlbedo"` (default), `"RegressionFunctionAlbedo"`, or `"CouplerAlbedo"`.

For example:

```yaml
surface_setup: "DefaultMoninObukhov"   # flux_scheme
prognostic_surface: "PrescribedSST"    # temperature
albedo_model: "ConstantAlbedo"         # surface_albedo
```

The fourth field, [`boundary_overrides`](@ref "Boundary overrides (boundary_overrides)"),
has no YAML key: it is populated by a setup's
[`surface_condition`](@ref ClimaAtmos.Setups.surface_condition) (its `overrides`
field), or left at the all-`nothing` default.

The two `surface_setup` markers,
[`DefaultMoninObukhov`](@ref ClimaAtmos.SurfaceConditions.DefaultMoninObukhov) and
[`DefaultExchangeCoefficients`](@ref ClimaAtmos.SurfaceConditions.DefaultExchangeCoefficients),
are lightweight placeholders that the config-driven constructor resolves into a
concrete `flux_scheme` against `params` (a default roughness length or exchange
coefficient).

### Coupling to an external driver

The coupler still builds a complete `AtmosSurface` (all four fields are present);
the two patterns differ only in the `flux_scheme`/`temperature` pair:

 1. **Atmosphere skips surface computation**: `flux_scheme = nothing` (YAML
    `"PrescribedSurface"`). `update_surface_conditions!` early-returns, so
    `temperature` is never read (leave it at its default).
    `init_sfc_conditions_zero!` pre-fills safe defaults at cache-build so RRTMGP /
    diagnostic EDMF never see uninitialized memory, and the coupler overwrites
    `sfc_conditions` directly.
 2. **Atmosphere computes fluxes from a coupler-supplied SST**: a real
    `flux_scheme` (e.g. `MoninObukhov(…)`) *together with*
    `temperature = CoupledTemperature(field)`. The coupler writes `T_sfc` into
    `field` between steps; the atmosphere reads it and computes the surface
    fluxes. Per-cell boundary overrides can be a
    `Fields.Field{<:SurfaceBoundaryOverrides}` on the cache. See
    `test/coupler_compatibility.jl`.

## Developer guide

The design rationale, data flow, dispatch chains, extension points, and
debugging checklist are in
[Surface Conditions Internals](surface_conditions_internals.md) in the
Developer Guide.
