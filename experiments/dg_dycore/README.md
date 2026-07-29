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
  `Y.c = (ρ, ρe, uₕ::Covariant12)`, `Y.f = (w)`; requires its κ₄/filter
  stabilization (defaults: cap/10 + filter_Nc = npoly). `momentum_adv =
  :fluctuation` is helem=4-only. Reuses BOTH ClimaAtmos HS functions
  (the uₕ drag applies directly to this state).

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
`output_dir = ...`, `diag_period = ...`; zonal-mean panels via
`post/zonal_means.jl` (ClimaAnalysis).

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
  `configs/baroclinic_wave_earth_topo.yml`) warps the extruded space, and
  the ICs / discrete-hydrostatic ρ correction / ᶜΦ / HS forcing all follow
  the warped `z` automatically. **BUT the FDDG horizontal fluxes are
  evaluated along the warped coordinate surfaces** — the terrain metric
  cross-terms (uₕ transport through sloped ξ³ surfaces, the true-horizontal
  pressure gradient) are not yet included, so this is geometry plumbing
  valid for gentle smoothed slopes only; a curvilinear (contravariant-flux)
  extension of the KG core is the follow-up. The smoothing preprocessing
  (`diffuse_surface_elevation!`) runs a CG Laplacian with DSS inside —
  grid generation, not part of the DG discretization.
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
