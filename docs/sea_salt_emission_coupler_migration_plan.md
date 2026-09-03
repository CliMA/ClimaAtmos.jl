# Migration plan: sea salt emission moves to the coupler

*Reviewer direction: the land–ocean mask should never live in the ClimaAtmos
cache; the coupler owns the surface, so the coupler should drive emission.
Surveyed 2026-08-31 against `zg/ssa-emission-transport` (post
Prescribed-struct removal) and the local `~/ClimaCoupler.jl` checkout on
`zg/ssa-ocean-mask` (c273d7ae). Revised same day: emission is computed
per-surface in the coupler's flux loop (using the ocean's own `u★`), not from
the combined atmos `sfc_conditions` — the ocean-arm inputs already exist, so
there is no reason to route through ClimaAtmos first, and no explicit ocean
mask survives anywhere.*

## The design in one paragraph

Sea salt emission becomes a per-surface coupler product, mirroring the
turbulent-flux pipeline: during the surface loop, the **ocean** simulation's
arm computes the per-bin Gong flux from the ocean's own `u★`/`L_MO` (which
`SF.surface_fluxes` already solves for per surface, per coupling step — the
values are currently discarded in `get_surface_fluxes`' return), weights it
by the ocean `area_fraction` exactly as `F_lh` etc. are weighted, and
accumulates it into new coupler fields; after the loop, a dedicated push —
**parallel to, not inside,** `update_turbulent_fluxes!` — writes the
combined per-bin fluxes into `p.tracers.sslt_sfc_fluxes`. The Gong
*physics* stays in ClimaAtmos as exported pure functions; ClimaAtmos keeps
the flux cache and the unchanged `aerosol_emission_tendency!`. The ocean
mask disappears as a concept: the ocean arm's standard area weighting *is*
the mask (and automatically suppresses emission under sea ice, since ice
area is not ocean area).

Physics consequences vs. the current implementation: in mixed (coastal)
cells the emission wind is the ocean surface's own MOST wind rather than the
land-contaminated combined `u★` — a real improvement given the `u₁₀^3.41`
nonlinearity. Aquaplanet coupled runs are bit-equivalent (one surface,
`area_fraction ≡ 1`, ocean `u★` == combined `u★`), so existing GPU
validation runs remain comparable.

## ClimaAtmos changes (amend `zg/ssa-emission-transport`)

1. **Keep the physics, export it as pure functions** in `sea_salt.jl`:
   `wind_at_height(z, ustar, L, sfp)` (already pure; COARE roughness — correct
   over ocean, which is now the only place it is used) and a new pure per-bin
   flux `sslt_bin_emission_flux(u₁₀, mass_flux_scale, u_ref, wind_exp)`
   = `mass_flux_scale * (u₁₀/u_ref)^wind_exp` [kg m⁻² s⁻¹, upward]. Both
   documented as coupler-facing.
2. **Replace the driver with a setter**, called once per coupling step:
   `set_sslt_surface_fluxes!(Y, p, bin_fluxes)` takes a tuple of per-bin
   *scalar* surface fields (already area-combined, physical magnitude,
   positive up; surface fields are 2D, so holding one remap target per bin
   is cheap) and does the `C3`/`unit_basis_vector_data` wrap into
   `p.tracers.sslt_sfc_fluxes` — the covariant convention stays inside
   atmos, so the coupler never handles flux components
   ([[ssa-covariant-flux-bug]]). `::Nothing` seasalt dispatch no-ops so the
   coupler can call unconditionally. The old `sfc_conditions`-reading driver,
   its `u₁₀` scratch use, and the `L_uninitialized = 1e-4` first-step
   sentinel (and its `TODO`) are deleted.
3. **Delete the precompute call** in `set_explicit_precomputed_quantities!`.
4. **Delete `ocean_fraction`** from `AtmosCache` (`OFRAC` type param, the
   `ones(...)` init in `build_cache`, the `precomputing_arguments` entry).
5. **Delete the mask-availability gating**: `requires_ocean_mask`,
   `check_ocean_mask_availability` (types.jl), `Setups.is_aquaplanet` + its
   `AtmosSimulation` call. Standalone runs then have zero emission (cache
   stays zero-init) — acceptable: the standalone aquaplanet path was already
   effectively unused, and every production prognostic run is coupled.
   Optionally warn at setup when prognostic aerosols run uncoupled.
6. **Tests**: emission becomes more testable standalone — the column
   assembly test feeds synthetic per-bin scalar fluxes through the new
   setter (exactly what the coupler will do) and checks the applied
   tendency; the pure `wind_at_height`/bin-flux functions get direct unit
   tests (monotonicity in `u★`, the `u₁₀^3.41` scaling, table consistency).

## ClimaCoupler changes (successor of `zg/ssa-ocean-mask`)

1. **Surface `u★`/`L_MO` out of the solver**: extend `get_surface_fluxes`
   (`src/FluxCalculator.jl:228`) to include `ustar` and `L_MO` from the
   `SF.surface_fluxes` `outputs` in its returned NamedTuple (they are
   computed and currently discarded at `:276`). Generally useful beyond this
   feature.
2. **Coupler fields**: register per-bin emission fields (e.g.
   `F_seasalt_emission_01 … _05`) via the extensible coupler-fields
   mechanism, gated on the atmos simulation actually carrying prognostic sea
   salt (`hasproperty(p.tracers, :sslt_sfc_fluxes)`). Zero them at the
   start of each exchange, wherever the `F_*` combined fields are reset.
3. **Per-surface hook, mirroring `update_flux_fields!`**: after
   `update_flux_fields!` in `compute_surface_fluxes!`
   (`src/FluxCalculator.jl:368`), call
   `accumulate_seasalt_emission!(csf, sim, fluxes, atmos_sim)` with
   - `::Interfacer.AbstractSurfaceSimulation` → no-op,
   - ocean surfaces → compute. GOTCHA (caught in adversarial review): the
     prescribed-SST ocean used by AMIP/subseasonal runs is
     `Models.PrescribedOceanSimulation <: AbstractSurfaceStub`, NOT an
     `AbstractOceanSimulation` — the dispatch must be the union of both
     (`SeaSaltEmittingSimulation`), else those runs silently emit zero.
     For `::SeaSaltEmittingSimulation` → compute
     `u₁₀ = CA.wind_at_height(10, fluxes.ustar, fluxes.L_MO, sfp)`, then for
     each bin `csf.F_seasalt_emission_xx .+= area_fraction .*
     CA.sslt_bin_emission_flux.(u₁₀, scale, u_ref, wind_exp)` with the
     `ifelse(area_fraction ≈ 0, zero, ...)` NaN guard copied from
     `update_flux_fields!`. Gong parameters come from
     `atmos_sim.integrator.p.params.prognostic_aerosol_params`.
4. **Dedicated push, parallel to the turbulent-flux push**: in
   `step_model_sims!` (`src/FieldExchanger.jl:309`), next to — not inside —
   `FluxCalculator.update_turbulent_fluxes!`, call
   `Interfacer.update_field!(atmos_sim, Val(:seasalt_emission_fluxes), csf)`;
   the ext method (`ClimaCouplerClimaAtmosExt.jl`) remaps the five bin
   fields to the atmos surface space and calls
   `CA.set_sslt_surface_fluxes!(u, p, bin_fluxes)`. Emission is thereby
   held constant over a coupling step, exactly like `ρ_flux_q_tot`.
   Scope note: `update_turbulent_fluxes!` stays a momentum/heat/moisture
   function; aerosols get the same *structure*, not the same function — and
   the structure is where land dust emission will slot in later (the land
   arm of the same per-surface hook).
5. **Slow surfaces**: `update_flux_fields!` updates the area-weighted `csf`
   contribution unconditionally even when a `FluxAccumulator` defers the
   surface-side push; the emission accumulation piggybacks on the same
   unconditional `csf` update, so no accumulator changes are needed.
6. **Delete the ocean-fraction push**: `Val(:ocean_fraction)` registration
   (`src/Interfacer.jl:344`), the ext `update_field!` method
   (`ClimaCouplerClimaAtmosExt.jl:411–415`), and the `update_sim!` line
   (`src/FieldExchanger.jl:231`).
7. **Compat**: land the ClimaAtmos change first; the coupler PR bumps its
   ClimaAtmos compat (or, on the server checkout during transition, guards
   with `isdefined(CA, :sslt_bin_emission_flux)`).

## Behavior / validation

- **Aquaplanet coupled runs**: bit-equivalent after the first coupling step
  (single all-ocean surface ⇒ ocean `u★` == combined `u★`); the first step
  improves (emission from the initial exchange instead of sentinel-gated to
  zero). Existing wxquest segments remain the reference.
- **AMIP mixed cells**: answers change by design (ocean-specific wind,
  under-ice suppression via ocean area fraction). Validate with an A/B
  server segment comparing SSLT burdens/`emiss`-diagnostics rather than
  bit-diffs; coastal cells are where differences concentrate.
- **Standalone runs**: zero emission. No repro configs exercise prognostic
  aerosols, so ClimaAtmos CI is unaffected; unit tests drive the setter
  directly.
- **Deposition stays in atmos**: dry (Zhang, water category everywhere) and
  wet deposition need no per-surface information yet, so their precomputes
  are untouched. When per-land-use Zhang categories arrive, they can follow
  this same per-surface pattern.

## Sequencing

1. Amend `zg/ssa-emission-transport` (items 1–6), restack the five branches
   above (trivial context conflicts expected where the growth branch adds
   `set_sslt_growth_factor!` next to the deleted precompute call).
2. Revalidate: aerosol suite at the stack tip + dual-source script + the new
   setter-driven emission assembly block.
3. Coupler branch replacing `zg/ssa-ocean-mask` (drop the ocean-fraction
   interfacer commit; add items 1–6); server-side aquaplanet bit-check, then
   AMIP A/B.
