# Surface Conditions Internals

Design rationale, data flow, dispatch chains, extension points, and
debugging for the surface-conditions subsystem. For the user-facing guide —
the four configuration knobs and how to set them, see
[Surface Conditions](surface_conditions.md).

## Design: one source of truth

Surface behavior lives entirely on `atmos.surface`. Principles:

  - **Orthogonality**: `flux_scheme`, `temperature`, `boundary_overrides`, and
    `surface_albedo` are independent axes. Adding an option on one shouldn't touch
    the others.
  - **Dispatch over branching**: behavior is selected by dispatch on concrete
    types, not `if/elseif` on config strings.
  - **Eager resolution**: YAML markers and `Default*` placeholders resolve to
    concrete structs at construction, so the hot path sees only concrete types.

## Data flow

The entry point
[`update_surface_conditions!`](@ref ClimaAtmos.SurfaceConditions.update_surface_conditions!)
(called from `set_explicit_precomputed_quantities!`) does four things: (1) early-return if
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

## Dispatch chains

Three small families cover all behavior:

**`surface_temperature`** (`surface_temperature.jl`): temperature type → value:

| Type                   | Returns                               |
|:---------------------- |:------------------------------------- |
| `AnalyticTemperature`  | the struct itself (deferred)          |
| `ExternalTemperature`  | `field_values` of the evaluated input |
| `SlabOceanTemperature` | `field_values(Y.sfc.T)`               |
| `CoupledTemperature`   | `field_values(t.field)`               |

**`resolve_T_sfc`** (`surface_conditions.jl`): in the per-cell kernel, an
[`AnalyticTemperature`](@ref ClimaAtmos.SurfaceConditions.AnalyticTemperature)
is evaluated as `t.f(coordinates, surface_temp_params, t_time)`; scalars and
`DataLayout`s pass through. This two-step design lets analytic formulas see each
cell's local coordinates while field-valued temperatures resolve once up front.

**Flux scheme → flux specs** (in `surface_state_to_conditions`): branches on
[`ExchangeCoefficients`](@ref ClimaAtmos.SurfaceConditions.ExchangeCoefficients)
vs [`MoninObukhov`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov), and within
`MoninObukhov` on whether fluxes are prescribed
([`HeatFluxes`](@ref ClimaAtmos.SurfaceConditions.HeatFluxes)/`θAndQFluxes`) or
derived from roughness.

## Constraints

  - **Scalars must broadcast.** `Base.broadcastable(x) = tuple(x)` is defined once
    on the abstract supertypes [`SurfaceParameterization`](@ref ClimaAtmos.SurfaceConditions.SurfaceParameterization) and [`SurfaceTemperature`](@ref ClimaAtmos.SurfaceConditions.SurfaceTemperature),
    so every concrete subtype inherits it for free. A new subtype needs nothing
    extra; the only ways to break this are introducing a parallel hierarchy that
    isn't a subtype, or removing the supertype method.
  - **`surface_temperature` returns a `DataLayout`, an `AnalyticTemperature`, or a
    scalar**: nothing else. Return `Fields.field_values(...)`, not a `Field`. A
    scalar is permitted (it passes through `resolve_T_sfc` unchanged), but no
    built-in type currently returns one; the four in-tree types return either the
    struct (`AnalyticTemperature`) or `field_values(...)`.
  - **Time-varying fluxes resolve per-update, not per-cell**: a `MoninObukhov`
    with a callable `fluxes` has it evaluated once by `resolve_flux_scheme`, then
    the resulting numeric scheme is broadcast everywhere.
  - **`isnothing(flux_scheme)` is a supported state**: any reader of
    `atmos.surface.flux_scheme` must handle it.
  - **Only [`SlabOceanTemperature`](@ref ClimaAtmos.SurfaceConditions.SlabOceanTemperature) adds prognostic state**: `Y.sfc` exists only for
    slab runs, so guard `Y.sfc.T` access on that type.

## Extending

Both extension points follow the same shape: define a concrete subtype, then add
the handful of methods the pipeline dispatches on. Because
`Base.broadcastable(::SurfaceTemperature)` and
`Base.broadcastable(::SurfaceParameterization)` are defined on the *abstract*
supertypes, your subtype inherits broadcastability for free; you do not need to
redefine it.

### A new temperature source

 1. **Define the type** as a subtype of [`SurfaceConditions.SurfaceTemperature`](@ref ClimaAtmos.SurfaceConditions.SurfaceTemperature).
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
    `SlabOceanTemperature`: add a `surface_prognostic_variables(local_geometry, ::MyTemperature)` initializer and a `surface_kwargs(surface_space, ::MyTemperature)` method (so `Y.sfc` is allocated), a `surface_temp_tendency!`
    method for the time evolution, and any conservation-diagnostic dispatch in
    `diagnostics/conservation_diagnostics.jl`.

 5. **(Optional) Expose it to configs** by extending
    `AtmosSurface(::AtmosConfig, ...)` in `src/config/model_getters.jl` (or have a
    setup return it from [`surface_condition`](@ref ClimaAtmos.Setups.surface_condition)).

### A new flux scheme

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

## Config and cache wiring

  - `AtmosSurface(::AtmosConfig, params, FT; setup_type)`
    (`src/config/model_getters.jl`) maps YAML keys + setup pieces into a concrete
    [`AtmosSurface`](@ref ClimaAtmos.AtmosSurface); setup pieces win via `@something`.
  - `build_cache` (`src/cache/cache.jl`) stores `p.sfc_setup = atmos.surface.boundary_overrides` (a scalar, or a `Field` for the coupler) and
    calls
    [`init_sfc_conditions_zero!`](@ref ClimaAtmos.SurfaceConditions.init_sfc_conditions_zero!)
    when `isnothing(flux_scheme)`.

```@docs
ClimaAtmos.SurfaceConditions.init_sfc_conditions_zero!
```

## Debugging checklist

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
