# Sea Salt + Diagnostic EDMF Implementation Summary

Branch: `zg/aerosol-playground`

## Overview

Five prognostic sea salt mass bins (SSLT01–SSLT05, radii 0.03–10 µm) are advected as tracers and coupled to the Diagnostic EDMF scheme for sub-grid-scale vertical transport. This feature is gated on `atmos.prognostic_aerosols` (a `Val{names}` of bin symbols) and `atmos.edmfx_model.prognostic_aerosols` (a `Val{true/false}` boolean, separate field).

---

## Key Files

| File | Role |
|------|------|
| `src/parameterized_tendencies/aerosols/sea_salt.jl` | Emission, deposition, MOST wind extrapolation |
| `src/cache/tracer_cache.jl` | Cache allocation for prescribed and prognostic aerosols |
| `src/cache/cache.jl` | `build_cache` — wires `tracer_cache` and `precomputing_arguments` |
| `src/cache/diagnostic_edmf_precomputed_quantities.jl` | EDMF bottom BC + column march for sea salt |
| `src/prognostic_equations/edmfx_sgs_flux.jl` | SGS mass-flux tendency for sea salt bins |
| `src/diagnostics/tracer_diagnostics.jl` | `emiss`, `emisslt01`–`emisslt05`, `loadss`, `mmrss` diagnostics |

---

## Emission Scheme (Gong 2003)

- **Wind**: 10 m wind extrapolated from the lowest model level via MOST profile ratio (`monin_obukhov_wind_extrapolated`). Roughness length from COARE3.
- **Flux**: `number_flux = bin_integral * |u₁₀|^3.41`, converted to mass via `(4/3)π r³ ρ_salt` per particle.
- **SST adjustment**: Available via `SST_adj=true` keyword but off by default.
- **Ocean masking**: Multiplied by `p.ocean_fraction` (set by coupler before first step).
- **Timing**: Computed in `set_sea_salt_emission_flux!`, called from `set_explicit_precomputed_quantities!` after surface conditions are available but before the EDMF column-march, so fluxes are ready as updraft surface BCs.
- **Cache**: Per-bin fluxes stored in `p.tracers.prognostic_aerosols_field` (a surface NamedTuple keyed by bin symbol).

## Deposition

Simple exponential decay: `d(ρSSLTxx)/dt = -λ ρSSLTxx` with a shared half-life of 0.55 days. **TODO**: replace with size-dependent dry deposition; add Stokes settling for coarse bins (SSLT04–05).

---

## Cache Architecture

`tracer_cache` builds two sub-caches merged into `aerosol_cache`:

```
prescribed_aerosol_cache  →  prescribed_aerosols_field (3D NamedTuple, updated by radiation callback)
                              prescribed_aerosol_timevaryinginputs

prognostic_aerosol_cache  →  prognostic_aerosols_field (2D surface NamedTuple, one field per bin)
```

**Important**: `tracers` and `ocean_fraction` are created **before** `set_precomputed_quantities!` is called in `build_cache` and included in `precomputing_arguments`. This is required because `set_sea_salt_emission_flux!` accesses both during the initial cache-building call.

---

## Diagnostic EDMF Integration

Three integration points, all gated on `p.atmos.edmfx_model.prognostic_aerosols isa Val{true}`:

### 1. Bottom BC (`diagnostic_edmf_precomputed_quantities.jl` ~line 410)
Updraft sea salt at level 1 initialized to grid-mean concentration via `sgs_scalar_first_interior_bc` with zero kinematic surface flux (emission enters the grid-mean, not the updraft directly).

### 2. Column March (~line 1211)
Per-level updraft advection using `diag_edmf_advection` + `entr_detr`. Uses a materialized scratch field (`p.scratch.ᶜtemp_scalar`) to compute grid-mean specific mixing ratio before slicing with `Fields.level` — **do not use `@. lazy(...)` here**, it returns a `(Broadcasted, Int)` tuple that `Fields.field_values` cannot accept.

### 3. Kill-updraft reset (~line 1274)
When the updraft is killed (weak vertical velocity), updraft sea salt is reset to grid-mean.

### 4. SGS Mass-Flux Tendency (`edmfx_sgs_flux.jl` ~line 354)
Vertical transport tendency applied to `ρSSLTxx` via `vertical_transport`, using draft-area-weighted `(χʲ - χ̄)` flux divergence. Capped by `0.02 / max(w³_diff, ε)` to prevent numerical blowup.

---

## Known Issues (as of branch state)

1. **Negative tracers + aphysical masses**: The model produces negative sea salt concentrations and unphysically large masses. Root cause not yet diagnosed. Suspected: SGS mass-flux transport in EDMF is producing spurious sources/sinks. The `vertical_transport` cap may be insufficient, or the updraft initialization at level 1 (grid-mean, not zero) is feeding back incorrectly.

2. **`lazy` misuse**: Three instances fixed in `diagnostic_edmf_precomputed_quantities.jl` (lines ~422, ~1222, ~1284) where `Fields.level(@. lazy(...), i)` returned a `(Broadcasted, Int)` tuple. Fixed by materializing into `p.scratch.ᶜtemp_scalar` first. The `lazy` usages in `edmfx_sgs_flux.jl` are fine — they stay inside `@.` broadcast expressions.

3. **Deposition is a placeholder**: Single shared exponential half-life, not size-dependent.

---

## Design Decisions

- `_aerosol_names(::Val{names})` unwraps the `Val{names}` type. Defined in `sea_salt.jl`, used in `tracer_cache.jl` and `cache.jl` (both included after `sea_salt.jl` in `ClimaAtmos.jl`).
- `prognostic_aerosols_field` naming mirrors `prescribed_aerosols_field` — both are NamedTuples keyed by bin symbol, just at different vertical extents (surface 2D vs full 3D column).
- Two separately named config keys both map to a field called `prognostic_aerosols` on different structs:
  - Config `"edmfx_aerosols"` → `atmos.edmfx_model.prognostic_aerosols` (`Val{true/false}`) — gates EDMF transport
  - Config `"prognostic_aerosols"` → `atmos.prognostic_aerosols` (`Val{(:SSLT01,...)}`) — names the tracer bins
