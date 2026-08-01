# DG dycore sandbox (`experiments/dg_dycore/`)

Sandboxed port of the ClimaCore DG-FD sphere dynamical cores into the
ClimaAtmos repository — **Phase A** of the staged DG-integration plan.
Nothing under `ClimaAtmos/src` is touched; ClimaAtmos is consumed as a
library. Phase B (a `horizontal_discretization` option on `AtmosModel`)
starts only after this sandbox validates against the ClimaCore references.

Provenance: `ClimaCore.jl` branch `as/bickley-jet`,
`examples/hybrid/sphere/discontinuous_galerkin/`. The DG operators
(Kennedy–Gruber flux differencing, KG-Rusanov/KG-Roe interfaces, DG
liftings, LDG/SIPG Laplacian) are committed in that branch's
`src/Operators/numericalflux.jl` (ClimaCore ≥ 0.14.55).

## What this is

Two dry dynamical cores on the cubed sphere (DG horizontal spectral
elements **without DSS** + FD vertical staggering, HEVI or explicit):

- **Flux-form FDDG** (`src/flux_form.jl`) — state
  `Y.c = (ρ, ρe, ρu1, ρu2, ρu3)` (momentum in global Cartesian
  components), `Y.f = (ρw)`. Kinetic-energy-preserving flux differencing;
  the validated production form (10-day baroclinic wave, conservation
  ~1e-14, runs pure-KEP with the floored-Roe interface).
- **Vector-invariant** (`src/vector_invariant.jl`) — state
  `Y.c = (ρ, ρe, uₕ::Covariant12)`, `Y.f = (w)`. Two face sets
  (`face_set` keyword; derivation in `docs/vi_kep_face_terms.md`):
  - `:kg` (legacy): KG scalar fluxes + Rusanov + plain λ jump penalties —
    NOT KE-compatible with the VI pairing; requires κ₄/filter stabilization
    (defaults: cap/10 + filter_Nc = npoly).
  - `:kep`: the VI-KEP-compatible set — `{ρũ}` mass flux (average of
    nodal contravariant mass fluxes, volume + central interface, NO density
    penalty), matching ρe flux, ρ-weighted velocity penalties (exact
    sign-definite KE sinks). The horizontal advective KE production closes
    to roundoff, on flat AND terrain-warped grids (metric-transparent
    ledger), so κ₄ = 0 / filter_Nc = 0 are the defaults. Verified by
    `test/test_vi_kep_budget.jl`.

  `momentum_adv = :fluctuation` is helem=4-only. Reuses BOTH ClimaAtmos HS
  functions (the uₕ drag applies directly to this state). Over topography
  this core carries the covariant terrain terms (full-metric K, full
  contravariant ᶠu³) — the CG-shared metric machinery.

## Layout

| Path | Content |
|---|---|
| `Project.toml` | own env; `[sources]`: ClimaAtmos → `../..`, ClimaCore → local `as/bickley-jet` checkout (switch to URL+rev before CI) |
| `driver.jl` | YAML entry point (`configs/*.yml`) |
| `src/DGDycore.jl` | module scaffold |
| `src/parameters.jl` | `DGConstants{FT}`: `:parity` (example literals) / `:clima_params` (A2) |
| `src/problems.jl` | `BaroclinicWaveFDDG` kwarg struct (ENV-free) |
| `src/model.jl` | `DGModel`: spaces, Cartesian basis, sponge, κ₄ cap, operators — the de-globalized replacement for the examples' module consts; passed as integrator `p` |
| `src/initial_conditions.jl` | JW06 formulas (`JWParams`) + discrete-hydrostatic ρ correction |
| `src/flux_form.jl` | FDDG tendencies (explicit + HEVI split) |
| `src/jacobians.jl` | analytic column Jacobian (`@name(c.ρ)/@name(c.ρe)/@name(f.ρw)` arrowhead) |
| `src/simulation.jl` | `DGSimulation`, `run!`, step monitor, conservation report |

## Run

```bash
julia --project=experiments/dg_dycore experiments/dg_dycore/driver.jl \
    experiments/dg_dycore/configs/balanced_flow_parity.yml
```

```julia
include("experiments/dg_dycore/src/DGDycore.jl"); using .DGDycore
result = run!(DGSimulation(BaroclinicWaveFDDG(;
    helem = 10, zelem = 30, dt = 60.0, t_end = 3 * 86400.0,
    interface_flux = :roe, κ₄ = 0.0)))   # the pure-KEP configuration
result.sol.u[end].c.ρ
```

Both cores share the ClimaAtmos-standard NetCDF diagnostics
(`pfull, ua, va, wa, ta, ke, rhoa` + DG-consistent `rv`): enable with
`output_dir = ...`, `diag_period = ...`. Post-processing (ClimaAnalysis +
CairoMakie, both write PNGs into the output dir):

- `post/zonal_means.jl <output_dir> [spinup_days]` — time & zonal mean
  u(φ, z), T(φ, z) (Held–Suarez climatology).
- `post/wave_train_maps.jl <output_dir> [z_km] [days]` — 2×2
  baroclinic-wave panels (ta, va, rv, pfull) on lat–lon maps at a fixed
  altitude (default 2.5 km, above the Hughes2023 peaks), one figure per
  snapshot or per requested day. Zonal means would average out the
  mountain-forced wave train; these maps are the H&J signature figures.

## Porting notes (ClimaCore examples → this sandbox)

1. **De-globalization**: the examples configure from ENV at include time
   into module `const`s; here everything lives in `DGModel`. Any `FT`
   used inside broadcast closures must be a **static type parameter**
   (`f(m::DGModel{FT}) where {FT}`) — a runtime `FT = float_type(m)`
   local makes closure return types uninferable (ClimaCore
   `eltype_error`).
2. **Structs in broadcasts** need `Base.broadcastable(x) = tuple(x)`
   (`DGConstants`, `JWParams`).
3. **ClimaTimeSteppers 0.9** (ClimaAtmos-pinned) vs the examples' older
   CTS: use `CTS.ODEFunction` / `CTS.ODEProblem` / `CTS.solve` (not the
   `SciMLBase` constructors) and CTS-native callbacks
   (`CTS.Callbacks.EveryXSimulationSteps`); `SciMLBase.DiscreteCallback`
   is not accepted. Explicit stepping via
   `CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher())` (drops the examples'
   OrdinaryDiffEqSSPRK dependency).
4. **State naming**: `Y.Yc/Y.uₕ/Y.w/Y.ρw` → `Y.c/Y.f` NamedTuples
   (ClimaAtmos convention; enables writer/diagnostics reuse in A3).

## Invariants & limitations

- `Spaces.weighted_dss!` is **never called** (grep-enforced by tests,
  Stage A5). The tendency cutoff filter is **not exposed** for the
  flux-form core (it voids the KEP property — measured destabilization).
- Dry, shallow-atmosphere, single-process CPU or single GPU (the DG face
  loops have no MPI ghost exchange).
- `topography: earth` (ETOPO2022 60arcsec artifact → `SpaceVaryingInput`
  onto the GLL nodes → diffusion smoothing → `Hypsography.LinearAdaption`,
  the ClimaAtmos `grids.jl` recipe; see
  `configs/baroclinic_wave_earth_topo.yml`) or `topography: hughes2023`
  (the Hughes & Jablonowski 2023 analytic double mountain, evaluated
  pointwise, no smoothing; see
  `configs/baroclinic_wave_double_mountain_kep.yml`) warp the extruded
  space; the ICs / discrete-hydrostatic ρ correction / ᶜΦ / HS forcing all
  follow the warped `z` automatically.
  - **Vector-invariant core**: carries the covariant terrain terms
    (full-metric K, full contravariant ᶠu³ = CT3(uₕ) + CT3(w)); remaining
    O(slope) approximations are the horizontal projection of DG face
    normals and the w = 0 surface value (no CA-style surface-velocity
    constraint). The `:kep` KE ledger is exact over terrain regardless
    (`docs/vi_kep_face_terms.md` §6).
  - **FDDG (Cartesian) core**: geometry plumbing only — the horizontal KG
    fluxes run along the warped coordinate surfaces and the metric
    cross-terms are absent; valid for gentle smoothed slopes, pending a
    curvilinear (contravariant-flux) extension.

  The Earth-topography smoothing (`diffuse_surface_elevation!`) runs a CG
  Laplacian with DSS inside — grid generation, not part of the DG
  discretization. NOTE: its per-step κ is scaled by the MINIMUM quadrature
  node area (`min(WJ)`), not the average node spacing of the ClimaAtmos
  recipe, which is measurably unstable at coarse helem (helem = 4 diverges
  after ~12 iterations).
- Stage A1 keeps the examples' per-call temporaries; preallocation into
  `DGModel` is a later perf pass.

## Verification (Stage A1 acceptance)

1-day unperturbed balanced flow, helem=4/zelem=10, HEVI dt=60, κ₄=0
(`configs/balanced_flow_parity.yml`) vs the ClimaCore reference
(`run_problem(BaroclinicWaveFDDG(; perturb=false, dt=60.0, t_end=86400.0,
κ₄=0.0, filter_Nc=0))` in the ClimaCore repo):
mass conservation ≤ 1e-14; balanced-flow drift `max|v|` within 1%;
HEVI split check exactly 0 on centers.

GPU smoke (run on a CUDA machine; this sandbox is device-agnostic):
```bash
CLIMACOMMS_DEVICE=CUDA julia --project=experiments/dg_dycore \
    experiments/dg_dycore/driver.jl experiments/dg_dycore/configs/balanced_flow_parity.yml
```
