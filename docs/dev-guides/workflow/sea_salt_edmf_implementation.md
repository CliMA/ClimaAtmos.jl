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
Per-level updraft advection using `diag_edmf_advection` + `entr_detr`. Grid-mean specific mixing ratio follows the same pattern as `ᶜq_tot` in this function: bind `ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))` to a variable, then slice with `Fields.level(ᶜχ, i)`. (The earlier "(Broadcasted, Int) tuple" failure was a macro-precedence pitfall, not a ClimaCore limitation: inlining `Fields.level(@. lazy(...), i)` makes `@.` swallow the `, i` argument. Bind the lazy first.)

### 3. Kill-updraft reset (~line 1274)
When the updraft is killed (weak vertical velocity), updraft sea salt is reset to grid-mean.

### 4. SGS Mass-Flux Tendency (`edmfx_sgs_flux.jl` ~line 354)
Vertical transport tendency applied to `ρSSLTxx` via `vertical_transport`, using draft-area-weighted `(χʲ - χ̄)` flux divergence. Capped by `0.02 / max(w³_diff, ε)` to prevent numerical blowup.

---

## Blowup Diagnosis (2026-06-09)

The "negative tracers + aphysical masses" issue (alternating ± values growing to
huge magnitudes) was diagnosed by code analysis. Two changes were made:

1. **Root cause: missing eddy-diffusion (ED) term.** Sea salt bins received the
   EDMF mass-flux (MF) tendency in `edmfx_sgs_mass_flux_tendency!` but were
   absent from the tracer loop in `edmfx_sgs_diffusive_flux_tendency!`. The
   deviatoric MF flux `ρa·(wʲ−w̄)(χʲ−χ̄)` contains a `−ρa·(wʲ−w̄)·χ̄` component:
   an advection of χ̄ with *downward* velocity `−a_eff·Δw`, but `ᶠupwind1`
   samples against `+Δw`, i.e. from the *downwind* side — a negative-diffusion
   term on χ̄ (ν ≈ −a_eff·Δw·Δz/2, up to a few m²/s). For energy/q_tot/
   microphysics this is overwhelmed by their ED flux (K_h ~ 10–100 m²/s damps
   2Δz modes at 4K/Δz²); sea salt had **no vertical mixing at all**
   (hyperdiffusion is horizontal-only, vert_diff is off in EDMFX configs).
   Net result: exponential growth of column-mass-conserving, alternating-sign
   2Δz noise — exactly the observed symptom (e-folding ~ Δz²/4ν ~ 10³ s).
   **Fix**: SSLT bins added to the diffusion loop in
   `edmfx_sgs_diffusive_flux_tendency!` (α = 1). Because vertical diffusion is
   stiff (the very reason `implicit_diffusion: true` exists, and box EDMFX
   configs use it), the SSLT bins were also given tridiagonal
   `∂(∂ₜρχ)/∂(ρχ) = divᵥ ∘ ρK_h gradᵥ ∘ 1/ρ` blocks in
   `manual_sparse_jacobian.jl`, replacing their identity blocks whenever the
   diffusion derivative flag is active — the same approximation used for
   `ρq_tot`, `ρtke`, and the microphysics tracers (∂/∂ρ and ∂K_h/∂tke
   neglected, exactly as upstream does).

2. **The "blind EDMF fix" environment branch was removed** (was added in commit
   `1fb1427d1`). Its premise was wrong: algebraically,
   `ρa⁰(w⁰−w̄)(χ⁰−χ̄) = +(ρaʲ/ρa⁰)·ρaʲ(wʲ−w̄)(χʲ−χ̄)` — the env flux has the
   *same sign* as the updraft flux (verified numerically), so it amplified
   rather than balanced the transport, and it bypassed the `0.02/Δw` stability
   cap. Additionally, reconstructing `χ⁰ = (ρχ−ρaʲχʲ)/ρa⁰` is invalid in
   DiagnosticEDMFX because the diagnosed updraft is not constrained to satisfy
   the grid-mean decomposition (upstream's `ᶜspecific_env_value` explicitly
   errors for DiagnosticEDMFX for this reason); χ⁰ goes strongly negative
   wherever the plume carries surface concentrations aloft.

Notes from the analysis:
- The hull clamp in the column march (commit `3d99b9adb`) is sound and was
  kept: a passive tracer with no in-updraft source must stay in the convex
  hull of its inputs, and the SGS tendency is column-conservative regardless
  (zero-flux `ᶜadvdivᵥ` BCs).
- The `ratio_max` diagnostic (4.1) is misleading: with the hull clamp, a plume
  legitimately carries surface χ̄ (~1e-9) to levels where local χ̄ ~ 1e-20, so
  O(1e10) ratios do not localize a bug. The `SSLT_DIAG_COUNTER` instrumentation
  in `edmfx_sgs_flux.jl` can be removed once runs are clean.
- Emission sign convention, Gong-2003 units (radius in m from ClimaParams),
  ocean masking, bottom BC (χʲ₁ = χ̄₁ via zero kinematic flux), kill-updraft
  resets, and scratch-field usage in the column march were all checked and are
  correct.

## Known Issues (as of branch state)

1. **Deposition is a placeholder**: Single shared exponential half-life, not size-dependent.

2. **Performance (fixed 2026-06-09)**: the per-bin full-column `specific(ᶜρχ, Y.c.ρ)` scratch broadcast inside the per-level column-march loop (O(nlevels² × nbins)) was removed. All three sea-salt blocks now use the upstream `ᶜq_tot` pattern — a variable-bound `@. lazy(specific(ᶜρχ, Y.c.ρ))` sliced per level with `Fields.level` — which materializes only the level being written and frees `p.scratch.ᶜtemp_scalar` from this file entirely (it is reused as `ᶜa_scalar` in `edmfx_sgs_flux.jl`, so this also removes an aliasing hazard).

3. **MF positivity**: even with the ED term restored, the deviatoric MF flux is not positivity-preserving for a tracer with ~zero background aloft; small negatives may still appear and may warrant clipping or a limiter on `ᶜa_scalar` later.

---

## Design Decisions

- `_aerosol_names(::Val{names})` unwraps the `Val{names}` type. Defined in `sea_salt.jl`, used in `tracer_cache.jl` and `cache.jl` (both included after `sea_salt.jl` in `ClimaAtmos.jl`).
- `prognostic_aerosols_field` naming mirrors `prescribed_aerosols_field` — both are NamedTuples keyed by bin symbol, just at different vertical extents (surface 2D vs full 3D column).
- Two separately named config keys both map to a field called `prognostic_aerosols` on different structs:
  - Config `"edmfx_aerosols"` → `atmos.edmfx_model.prognostic_aerosols` (`Val{true/false}`) — gates EDMF transport
  - Config `"prognostic_aerosols"` → `atmos.prognostic_aerosols` (`Val{(:SSLT01,...)}`) — names the tracer bins
