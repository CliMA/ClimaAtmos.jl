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

---

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
  `(t, FT) -> HeatFluxes/θAndQFluxes` — it is resolved once per update (e.g.
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
  `external_forcing.surface_inputs`.
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
pins any of these to a fixed value;
each field defaults to `nothing` (use the physical default). Most idealized LES
setups override `p` and `q_vap`.

### Albedo (`surface_albedo`)

Sets the shortwave reflectivity passed to the radiation scheme. Three models:

- **[`ConstantAlbedo`](@ref ClimaAtmos.ConstantAlbedo)**: a single value applied
  to both direct and diffuse shortwave.
- **[`RegressionFunctionAlbedo`](@ref ClimaAtmos.RegressionFunctionAlbedo)**: the
  Jin et al. (2011) ocean parameterization — a solar-zenith-angle-dependent
  *direct* albedo plus a separate *diffuse* albedo, with wind-speed-dependent
  surface roughness.
- **[`CouplerAlbedo`](@ref ClimaAtmos.CouplerAlbedo)**: albedo supplied by an
  external driver (the coupler).

**Direct vs. diffuse** The model carries distinct
`direct_sw_surface_albedo` and `diffuse_sw_surface_albedo` fields.
`ConstantAlbedo` sets them equal, `RegressionFunctionAlbedo` computes them
separately.

**Spectral** All three models write a single value across every shortwave band,
and the [`RegressionFunctionAlbedo`](@ref ClimaAtmos.RegressionFunctionAlbedo)
scheme treats the refractive index as wavelength-independent. The RRTMGP
interface arrays are band-resolved (`(nbnd_sw, ncol)`), so per-band albedo is a
supported extension point but it would require a model that fills bands with
distinct values.

**Longwave surface reflectivity** Albedo is shortwave-only, longwave surface reflectivity is handled
separately through `surface_emissivity`. 

See the [Ocean Surface Albedo](@ref "Ocean Surface Albedo") page for the
Jin (2011) [`RegressionFunctionAlbedo`](@ref ClimaAtmos.RegressionFunctionAlbedo) formulation..

### Choosing

`flux_scheme` and `temperature` are independent axes, and you set **both** (the
other two fields take defaults). Each row below is a compatible *pair*, not an
either/or:

| If you want… | `flux_scheme` | `temperature` |
|---|---|---|
| Stability-dependent fluxes over a prescribed SST  | [`MoninObukhov(; z0 = …)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov)                    | [`AnalyticTemperature(…)`](@ref ClimaAtmos.SurfaceConditions.AnalyticTemperature)                                                                                     |
| Fixed-coefficient bulk fluxes                     | [`ExchangeCoefficients(; Cd, Ch)`](@ref ClimaAtmos.SurfaceConditions.ExchangeCoefficients)    | [`AnalyticTemperature(…)`](@ref ClimaAtmos.SurfaceConditions.AnalyticTemperature)                                                                                     |
| Prescribed heat fluxes (constant or time-varying) | [`MoninObukhov(; z0, shf, lhf)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov) or [`MoninObukhov(; z0, fluxes = (t,FT)->…)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov) | [`AnalyticTemperature(…)`](@ref ClimaAtmos.SurfaceConditions.AnalyticTemperature) |
| An interactive slab ocean surface                 | [`MoninObukhov(…)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov)                           | [`SlabOceanTemperature(…)`](@ref ClimaAtmos.SurfaceConditions.SlabOceanTemperature)                                                                                   |
| Surface temperature from data                     | [`MoninObukhov(…)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov)                           | [`ExternalTemperature(…)`](@ref ClimaAtmos.SurfaceConditions.ExternalTemperature)                                                                                     |
| Coupler owns the surface (atmos skips fluxes)     | `nothing`                                                                                     | unused — coupler writes `sfc_conditions`                                                                                                                              |
| Coupler sets SST; atmos computes fluxes           | [`MoninObukhov(…)`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov)                           | [`CoupledTemperature(field)`](@ref ClimaAtmos.SurfaceConditions.CoupledTemperature)                                                                                   |

!!! note "Prescribed fluxes do not use MOST"
    When you set `shf`/`lhf` (or `θ_flux`/`q_flux`), those fluxes are used **as
    prescribed**: MOST does not compute them. They appear under `MoninObukhov`
    only because the prescribed-flux path currently lives inside that type (a
    historical conflation; see the Developer Guide). The required `z0` is used
    solely for the *momentum* closure, and only when `ustar` is not also
    prescribed — when both fluxes and `ustar` are given (as in every idealized
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

### Configuring from YAML

Three of the four `AtmosSurface` fields are YAML-configurable (resolved by
`AtmosSurface(::AtmosConfig, params, FT; setup_type)`); setup-provided pieces
take precedence over these defaults:

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

---

## Developer Guide

### Design: one source of truth

Surface behavior lives entirely on `atmos.surface`. Principles:

- **Orthogonality**: `flux_scheme`, `temperature`, `boundary_overrides`, and
  `surface_albedo` are independent axes. Adding an option on one shouldn't touch
  the others.
- **Dispatch over branching**: behavior is selected by dispatch on concrete
  types, not `if/elseif` on config strings.
- **Eager resolution**: YAML markers and `Default*` placeholders resolve to
  concrete structs at construction, so the hot path sees only concrete types.

### Data flow

The entry point
[`update_surface_conditions!`](@ref ClimaAtmos.SurfaceConditions.update_surface_conditions!)
(called from `set_precomputed_quantities!`) does four things: (1) early-return if
`isnothing(flux_scheme)`; (2) resolve the temperature via `surface_temperature`;
(3) resolve the flux scheme via `resolve_flux_scheme` (once per update);
(4) broadcast
[`surface_state_to_conditions`](@ref ClimaAtmos.SurfaceConditions.surface_state_to_conditions)
over every surface point.

!!! note "Why a `DataLayout` broadcast"
    The kernel mixes *surface*-space and *lowest-interior-level* values, which
    live on different spaces, so a normal `Field` broadcast would error. The
    code drops to `Fields.field_values(...)` (raw `DataLayout`s) so the values
    broadcast as plain same-shape arrays.

### Dispatch chains

Three small families cover all behavior:

**`surface_temperature`** (`surface_temperature.jl`): temperature type → value:

| Type | Returns |
|---|---|
| `AnalyticTemperature` | the struct itself (deferred) |
| `ExternalTemperature` | `field_values` of the evaluated input |
| `SlabOceanTemperature` | `field_values(Y.sfc.T)` |
| `CoupledTemperature` | `field_values(t.field)` |

**`resolve_T_sfc`** (`surface_conditions.jl`): in the per-cell kernel, an
`AnalyticTemperature` is evaluated as `t.f(coordinates, surface_temp_params,
t_time)`; scalars and `DataLayout`s pass through. This two-step design lets
analytic formulas see each cell's local coordinates while field-valued
temperatures resolve once up front.

**Flux scheme → flux specs** (in `surface_state_to_conditions`): branches on
`ExchangeCoefficients` vs `MoninObukhov`, and within `MoninObukhov` on whether
fluxes are prescribed (`HeatFluxes`/`θAndQFluxes`) or derived from roughness.

### Constraints

- **Scalars must broadcast.** `Base.broadcastable(x) = tuple(x)` is defined once
  on the abstract supertypes `SurfaceParameterization` and `SurfaceTemperature`,
  so every concrete subtype inherits it for free. A new subtype needs nothing
  extra; the only ways to break this are introducing a parallel hierarchy that
  isn't a subtype, or removing the supertype method.
- **`surface_temperature` returns a `DataLayout`, an `AnalyticTemperature`, or a
  scalar**: nothing else. Return `Fields.field_values(...)`, not a `Field`. A
  scalar is permitted — it passes through `resolve_T_sfc` unchanged — but no
  built-in type currently returns one; the four in-tree types return either the
  struct (`AnalyticTemperature`) or `field_values(...)`.
- **Time-varying fluxes resolve per-update, not per-cell**: a `MoninObukhov`
  with a callable `fluxes` has it evaluated once by `resolve_flux_scheme`, then
  the resulting numeric scheme is broadcast everywhere.
- **`isnothing(flux_scheme)` is a supported state**: any reader of
  `atmos.surface.flux_scheme` must handle it.
- **Only `SlabOceanTemperature` adds prognostic state**: `Y.sfc` exists only for
  slab runs, so guard `Y.sfc.T` access on that type.

### Extending

Both extension points follow the same shape: define a concrete subtype, then add
the handful of methods the pipeline dispatches on. Because
`Base.broadcastable(::SurfaceTemperature)` and
`Base.broadcastable(::SurfaceParameterization)` are defined on the *abstract*
supertypes, your subtype inherits broadcastability for free — you do not need to
redefine it.

#### A new temperature source

1. **Define the type** as a subtype of `SurfaceConditions.SurfaceTemperature`.
   Store whatever it needs (a function, a `Field`, parameters):

   ```julia
   struct MyTemperature{F} <: SurfaceConditions.SurfaceTemperature
       data::F
   end
   ```

2. **Add a `surface_temperature` method**, the per-update resolver. It must
   return one of the three broadcastable shapes: a scalar, a
   `Fields.DataLayout` of per-cell values, or the struct itself (deferred to the
   per-cell kernel):

   ```julia
   # field-valued: resolve once per update
   SurfaceConditions.surface_temperature(t::MyTemperature, Y, p, t_time) =
       Fields.field_values(t.data)
   ```

3. **(Optional) Add a `resolve_T_sfc` method** if you returned the struct in
   step 2 because `T_sfc` depends on each cell's coordinates (this is how
   `AnalyticTemperature` works). It runs inside the broadcast kernel and receives
   the local coordinates:

   ```julia
   SurfaceConditions.surface_temperature(t::MyTemperature, Y, p, _) = t  # defer
   SurfaceConditions.resolve_T_sfc(t::MyTemperature, coords, surface_temp_params, t_time) =
       t.data(coords, surface_temp_params, t_time)
   ```

4. **(Optional) Wire in prognostic state** if `T_sfc` should evolve, mirroring
   `SlabOceanTemperature`: add a `surface_prognostic_variables(local_geometry,
   ::MyTemperature)` initializer and a `surface_kwargs(surface_space,
   ::MyTemperature)` method (so `Y.sfc` is allocated), a `surface_temp_tendency!`
   method for the time evolution, and any conservation-diagnostic dispatch in
   `diagnostics/conservation_diagnostics.jl`.

5. **(Optional) Expose it to configs** by extending
   `AtmosSurface(::AtmosConfig, ...)` in `src/config/model_getters.jl` (or have a
   setup return it from `surface_condition`).

#### A new flux scheme

1. **Define the type** as a subtype of
   `SurfaceConditions.SurfaceParameterization{FT}` (the `{FT}` parameter lets
   `float_type` recover the element type):

   ```julia
   struct MyScheme{FT} <: SurfaceConditions.SurfaceParameterization{FT}
       coefficient::FT
   end
   ```

2. **Handle it in `surface_state_to_conditions`**: extend the
   `parameterization isa …` branch that maps the scheme onto the `SurfaceFluxes`
   call (building the appropriate `FluxSpecs`/`SurfaceFluxConfig`). This is the
   one place flux schemes are interpreted.

3. **(Optional) Add a `resolve_flux_scheme` method** if the scheme varies in
   time, mirroring how `MoninObukhov` resolves a callable `fluxes`. It runs once
   per update (not per-cell) and must return a concrete, time-independent scheme:

   ```julia
   SurfaceConditions.resolve_flux_scheme(p::MyScheme, t, ::Type{FT}) where {FT} =
       MyScheme{FT}(p.coefficient * cos(t))
   ```

4. **(Optional) Expose it to configs/setups** as in step 5 above.

### Config and cache wiring

- `AtmosSurface(::AtmosConfig, params, FT; setup_type)`
  (`src/config/model_getters.jl`) maps YAML keys + setup pieces into a concrete
  `AtmosSurface`; setup pieces win via `@something`.
- `build_cache` (`src/cache/cache.jl`) stores `p.sfc_setup =
  atmos.surface.boundary_overrides` (a scalar, or a `Field` for the coupler) and
  calls
  [`init_sfc_conditions_zero!`](@ref ClimaAtmos.SurfaceConditions.init_sfc_conditions_zero!)
  when `isnothing(flux_scheme)`.

```@docs
ClimaAtmos.SurfaceConditions.init_sfc_conditions_zero!
```

### Debugging checklist

- **`sfc_conditions` NaN/uninitialized under the coupler**: `init_sfc_conditions_zero!`
  only fires when `isnothing(flux_scheme)`.
- **`T_sfc` uniform when it should vary**: the temperature must return per-cell
  values, or be an `AnalyticTemperature` whose `f` actually reads `coordinates`.
- **Space-mismatch error in `update_surface_conditions!`**: something returned a
  `Field` instead of a `DataLayout`/scalar, or a type is missing `broadcastable`.
- **`Y.sfc` not found**: not a `SlabOceanTemperature` run; guard slab-only code.
- **Time-varying flux not updating**: `MoninObukhov.fluxes` must be a callable
  `(t, FT) -> PrescribedFluxes` (resolved each update), not a fixed `HeatFluxes`
  captured at construction.
