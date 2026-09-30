# DG dynamics backend inside ClimaAtmos (Tier B integration plan)

Goal: run the full ClimaAtmos physics stack (RRTMGP, EDMF, microphysics,
surface fluxes) on DG horizontal dynamics — aquaplanet first, then
topography/AMIP-like. Physics is column-local, so the integration replaces
only the horizontal dynamics; everything vertical and every physics
tendency is reused verbatim.

## Vehicle: the vector-invariant DG core

`src/vector_invariant.jl` (not the Cartesian flux-form core) is the
integration vehicle because its state already matches ClimaAtmos:
`Y.c = (; ρ, ρe, uₕ::Covariant12)`, `Y.f = (; w::Covariant3)`, and it uses
the same `ᶠu³ = CT3(w) + ρJ-weighted CT3(uₕ)` machinery. The `:kep` face
set closes horizontal advective KE to roundoff, so `κ₄ = 0` and
`filter_Nc = 0` are admissible (no CG hyperdiffusion needed for the dry
core; physics adds its own diffusion).

## Verified seams in ClimaAtmos

`remaining_tendency!` (src/prognostic_equations/remaining_tendency.jl:47)
accumulates, in order:

1. `horizontal_tracer_advection_tendency!`   (advection.jl:113)  → REPLACE
2. `horizontal_dynamics_tendency!`           (advection.jl:36)   → REPLACE
3. `hyperdiffusion_tendency!`                                    → DISABLE (kep)
4. `explicit_vertical_advection_tendency!`                       → KEEP
5. `additional_tendency!` (physics, sponges, EDMFX)              → KEEP

- The implicit block (`implicit_tendency!`, Jacobians, Newton) is untouched —
  the DG vertical treatment developed in this experiment already mirrors it.
- `dss!` (wired in `ClimaODEFunction`, simulation/integrator.jl:219) → no-op
  under DG; inter-element coupling enters through the DG interface fluxes in
  the replaced tendencies 1–2.
- CG limiters (`Yₜ_lim` / `limiters_func!`) → disabled; DG tracers rely on
  Rusanov/Roe interface dissipation (as in this experiment). Revisit tracer
  positivity under EDMF sources later (Zalesak-style DG limiter if needed).

## Work packages

1. **ClimaCore dependency alignment** (gating everything). Assessed
   2026-09-01 — REFRAMED after inspecting ClimaCore main:
   - **Main already carries the DG base infrastructure**, under active
     upstream development: `Grids.CG()`/`Grids.DG()` singleton grid type
     parameters with model-level CG↔DG switching via *tendency completions*
     (commit 1e815e350, 2026-09-01), `GhostFaceExchange` (MPI DG),
     GPU-coalesced face kernels, outflow BCs, and a 1,460-line
     numericalflux.jl.
   - **The `as/moisture-0M-mpi` branch uniquely holds the physics flux
     library** (~1,700 diff lines): the Cartesian Kennedy–Gruber /
     Ranocha / Waruszewski / Roe / Rusanov / ES two-point flux family,
     advective (pressure-stripped) variants, tracer fluxes, and lifting
     corrections. These are pure flux functions — the upstreaming unit is
     this library, rebased onto main's optimized kernel structure (face
     operators were renamed/optimized on main; expect mechanical conflicts
     in numericalflux.jl, none in concept).
   - **Integration seam**: target main's tendency-completion mechanism, NOT
     a hand-rolled replacement of `remaining_tendency!` internals. First
     step of any prototype: study the completion API (what it completes and
     how interface fluxes plug in) and coordinate with the upstream CG↔DG
     switching work — this is on the CliMA roadmap and duplicating it would
     be wasted effort.
   - ClimaAtmos pins ClimaCore 0.14.51 vs main's 0.15.3 (breaking series).
     The CA upgrade is an independent workstream; don't block on it —
     prototype on a CA dev branch pinned to ClimaCore main once the flux
     library lands there.
2. **`DGDynamics` module in ClimaAtmos**: port the VI-DG horizontal
   operators (flux-differencing ρ/ρe with the kep face set, vector-invariant
   momentum with central lifting, ρ-weighted velocity penalties) onto
   ClimaAtmos's `Y`/cache. Energy variable is `ρe_tot` (same quantity as the
   experiment's `ρe`).
3. **Config/dispatch**: an `atmos.horizontal_discretization = CG | DG` flag
   gating seams 1–3 + dss no-op + limiter off.
4. **MPI**: DG interface fluxes need ghost exchange (the distributed DG path
   exists in the ClimaCore branch; wire through ClimaAtmos's comms context).
5. **Validation ladder**:
   a. dry baroclinic wave: CA-CG vs CA-DG parity (this experiment's
      configs are the reference solutions);
   b. moist baroclinic wave (0-moment);
   c. aquaplanet with idealized physics (gray radiation, bulk fluxes,
      vert_diff) — also exercises the 30 m/dt≈120 regime, which requires BL
      mixing (measured in this experiment: the unmixed 30 m grid is
      dynamically unstable regardless of the transport scheme);
   d. full aquaplanet (RRTMGP + EDMF + 1M);
   e. Earth topography + SLEVE (0.7, 10.0) with the full stack.

## Known constraints from this experiment (2026-09)

- The vertical fluxes must carry the total contravariant flux (terrain
  cross-term); the pointwise metric needs no reconstruction.
- Lin–VanLeer vertical transport requires vertical advective C < 1; in
  production that is guaranteed by BL physics keeping near-surface w small
  (exactly ClimaAtmos's regime), not by the dynamics.
- ρw = 0 is the correct IC over terrain; never a column-wide tangent w.
- SLEVE at (0.7, 10.0) — the ClimaAtmos default — keeps near-surface cells
  at ≈ linear-warp thickness; aggressive (0.5, 0.5) is unsteppable over
  Earth topography with a 300 m floor.
