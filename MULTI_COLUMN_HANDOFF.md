# Handoff: multi-column simulations in ClimaAtmos

For an agent continuing this work. Read this file, then `MULTI_COLUMN_CHECKLIST.md`
(the deliverable: §0 lists the decisions the user has to make, with paths and options;
§7 is the verification table), then
`DIFF_SPACE.md` (static analysis of every place the single-column and multi-column spaces
differ) and `THINGS_TO_ADDRES.md` (adversarial review of the first commit; checklist §2b
tracks its items). Last updated 2026-09-09, end of the second session, on the `clima`
cluster (Julia 1.12.5 via `module load climacommon`), branch `kp/multi-col` at
`5ff8c5c8e` plus the uncommitted changes listed in §2.

## 0. Status in one screen

- Every column configuration in `config/model_configs` builds and runs on a multi-column
  grid through the real config path (`config: multicolumn`); the file-forced cases
  (GCM-driven, ARM VARANAL, ERA5-driven) need the ClimaUtilities `forcing-intp` prototype
  for the last two. The diurnal SCM case runs after widening the `config == "column"` assertion
  (A8, `model_getters.jl:853`) and is bitwise identical on the identity-metric grid. Still
  failing, on a single column too (pre-existing): the two sparse-autodiff column configs
  and 2M precipitation. See §0b for the
  third-session results that supersede the in-flight notes below.
- Column 1 (at lat = long = 0, `deep_atmosphere: false`, Float64) matches the single
  column to rtol 1e-9 in state, cache, and NetCDF output for: hydrostatic balance; gray,
  clear-sky, and slab-ocean radiative equilibrium; NOGW (3 configs); precipitation 1M;
  EDMF GABLS, simple plume, advection test, DYCOMS RF02, LARCFORM1 (state and NetCDF;
  cache to 4e-7 in `ᶜl_mix`); GCM-driven forcing (exact in Float64); checkpoint/restart;
  pressure-coordinate and fake-pressure-level outputs. Duplicate columns are bitwise
  identical in every case (except uninitialized cache scratch, A9).
- EDMF cases whose column 1 differs after 1 h: Bomex, DYCOMS RF01, TRMM, TRMM 0M,
  Bomex fixed-TKE, Soares (1e-8), Bomex mixing-length cloud (1e-7), and **RICO**. For all
  but RICO a one-ulp perturbation of the single column reproduces the difference (for
  Bomex and DYCOMS RF01 every row of the difference table is identical). **RICO does
  not: the single column reproduces itself to 1e-9 after 1 h under the perturbations
  tried, yet column 1 differs by 7e-5 in the updraft after one 120 s step and by 1e-3
  after 1 h. Treat RICO as a real defect in the multi-column path until proven
  otherwise.** This is the top open item; see §4.1.
- The user's standing caution: do not attribute a difference to floating point until it
  is proven (a perturbed single column must reproduce it). Time spent proving a
  floating-point origin is acceptable; a wrong "it's rounding" verdict is not.

## 0b. Third-session additions (2026-09-09, later)

- Identity-metric attribution experiment (`multi_column_dev/identity_metric.jl`, REPL
  only): redefines `CA.MultiColumnGrid` to build a `MultiPointGrid` whose horizontal
  metric is the identity (`J_h = 1`, `∂x∂ξ_h = I`), so the vertical metric terms equal
  the single column's. Hydrostatic balance, 1 day: **bitwise identical** state and cache,
  confirming `BITWISE_COMPARSION_HANDOFF.md` §1 (the rounding on the real grid comes from
  `J = J_h·J_v` and the 3×3 inverse of `∂x∂ξ`). The RICO run on this grid is the decisive
  test for §4.1: **RICO on the identity grid is bitwise identical to the single column
  after 1 h** (18/18 state entries; cache identical except uninitialized scratch). §4.1 is
  closed: the RICO difference is metric rounding, proven by removing it. The lesson for the
  sensitivity method: perturbing the initial state cannot stand in for per-step operator
  rounding; use the identity-metric grid as the attribution test. Bomex, DYCOMS RF01, and
  TRMM identity runs queued in REPL 5 to put the whole EDMF group on this footing. The user
  does not want bitwise identity as a goal; roundoff from the metric terms is expected.
- Identity-grid results for the EDMF group: Bomex, DYCOMS RF01, TRMM (1 h), and
  ERA5-driven in Float64 (10 min, with the ClimaUtilities prototype) are all bitwise
  identical in state; only uninitialized scratch differs in the cache. Together with RICO
  this attributes every remaining difference on the real grid to metric rounding. Every
  column configuration is now either an exact match (rtol 1e-9) on the real grid or
  bitwise identical on the identity-metric grid; nothing is left unexplained.
- ARM VARANAL in Float64 with the ClimaUtilities prototype: exact match (18 state entries;
  every flagged cache entry is a structural `NaN` from comparing the forcing data
  handlers' regridder coordinates/grids, which differ by construction).
  `comparison_utils.jl` now ignores `:data_handler`.
- Sparse-autodiff column configs: root cause located (checklist §7 row): the 1M terminal
  velocities are written into non-Dual cache fields inside the auto Jacobian. A fix means
  moving `ᶜwₗ`/`ᶜwᵢ`/`ᶜwᵣ`/`ᶜwₛ` into `implicit_precomputed_quantities` (Dual copies, like
  `ᶜρ_dq_tot_dt`); not attempted, it is a single-column autodiff issue.
- ClimaUtilities regridder change is now **opt-in** (user's decision): `InterpolationsRegridder(...;
  horizontally_uniform = true)`; without it, Z-only data on a LatLongZ target raises an
  informative error. Atmos opts in at `column_timevaryinginputs`
  (`src/column_datasets/ColumnDatasets.jl`). ARM VARANAL, ERA5-driven, and the diurnal case
  rerun with it (`multi_column_dev/forced_case.jl`, REPLs 6-8) reproduce their results.
  Remaining decision: the ClimaUtilities PR (checklist §0).
- A8 done: `ReanalysisMonthlyAveragedDiurnal` accepts both column kinds; the diurnal SCM
  case (Float64, 10 min, prototype) shows EDMF-level differences on the real grid and is
  bitwise identical on the identity grid (REPL 8, `queue8c.jl`).
- Kinematic driver: matches exactly after the `src/types.jl` fix (REPL 1 had not picked
  the fix up through Revise; rerun in REPL 7).
- Bomex tracerA: exact match. `rico_2M`: fails on a single column (2M disabled upstream).
- `bomex_sparse_autodiff` fails on the **single** column (`Float64(::ForwardDiff.Dual)`
  inside a ClimaCore broadcast loop during the auto Jacobian); `show_failure` truncates
  the stack before the Atmos frame. Not CI-covered. `multi_column_dev/queue7.jl` reruns it
  single-column in Float32 and Float64 to tell a pre-existing failure from one caused by
  the Float64 overlay. GABLS sparse-autodiff in flight (REPL 3).
- ERA5-driven: the raw artifact is present; single runs; multi hits the regridder gap
  without the ClimaUtilities prototype. Float64 rerun with the prototype queued in REPL 8
  after ARM VARANAL Float64 (`queue8b.jl`).
- `Pkg.test` diagnostics group: 72 passed, 2 errored, both in `test/cosp/subcol_test.jl`
  (`UndefVarError: ᶜq_lcl not defined in Main` inside `foreach_cosp_subcolumn`'s
  MultiBroadcastFusion macro); no changed file is involved, so this looks pre-existing or
  environmental, but it was not checked on `main`. The unit diagnostics with the
  multicolumn fixture passed.
- GPU (fourth session, one A100 via `srun --gpus=1` in the `julia-mc` session,
  `CLIMACOMMS_DEVICE=CUDA`): the ladder runs (`multi_column_dev/gpu_ladder.jl`), but every
  case except the kinematic driver fails while building the *single-column* simulation
  with an `InvalidIRError` in ClimaCore's eager stencil kernel; the sphere fails the same
  way. Minimal reproducer and the environment comparison are in the checklist §9 "GPU".
  The same probe is being run under CI's module (`climacommon/2025_03_18`, Julia 1.11.4,
  `Manifest-v1.11.toml`) to decide whether this is the Julia 1.12 / CUDA.jl 6 environment.
  Probe scripts: `multi_column_dev/gpu_probe2.jl`..`gpu_probe6.jl`, `gpu_err.jl`,
  `gpu_warntype*.jl`. **Verdict: Julia 1.12.5 / CUDA.jl 6.2.2 environment**; the same
  code passes under Julia 1.11.4 / CUDA.jl 5.11.3 (CI's `climacommon/2025_03_18`).
- GPU procedure that works (checklist §9 "GPU results"): in the `julia-mc` tmux session,
  `srun --pty -t 08:00:00 --gpus=1 --threads=3 /bin/bash -l`; then
  `export CLIMACOMMS_CONTEXT=SINGLETON CLIMACOMMS_DEVICE=CUDA`,
  `module load climacommon/2025_03_18`, `cd` to the repo, and
  `julia -O1 --project=<scratch env> --history-file=no`, where the scratch env is a copy of
  `.buildkite/Project.toml` and `Manifest-v1.11.toml` whose `[sources]` point at the repo,
  `.buildkite/PrecompileCI`, and the three worktrees by absolute path (this session's copy
  is under the session scratch directory, `.../scratchpad/env111`; recreate it elsewhere,
  it is not in the repo). Run `multi_column_dev/gpu_ladder.jl` (all stages) or
  `gpu_ladder2.jl` (set `GPU_STAGES` and `EDMF_CASES`). The pane is piped to
  `multi_column_dev/logs/gpu_ladder.log`. A 4-hour allocation covered the stages up to two
  EDMF cases; EDMF cases take ~15 min each on the GPU (compilation dominates).
- REPL state at the end of the third session: `julia-mc`, `julia-mc2`, `julia-mc4` exited
  (their Julia predated the `src/types.jl` fix); `julia-mc3` now hosts the formatter REPL
  (`--project=.dev/format`); `julia-mc5` has the identity-metric grid installed (restart
  it before running anything on the real grid); `julia-mc6`, `julia-mc7` idle with
  ClimaAtmos loaded (post-fix, registered ClimaUtilities); `julia-mc8` has the
  ClimaUtilities prototype dev'd (the manifest now points at it for every new REPL).
- Memory: REPLs hold 10-18 GB each after a session of cases; `GC.gc()` frees little.
  Restart a REPL (2.5 min load) before giving it a new batch. Do not open new tmux
  sessions; reuse `julia-mc`..`julia-mc8`.

## 0c. Fourth session (2026-09-10, evening): GPU on the PR branches

Supersedes §0b where they disagree. The user's instructions this session: test the three
hardest configs on one GPU (not all of them, "for most of them, they pass"), fix what
fails, use the ClimaCore branch that will become the PR, then cancel the slurm
reservation.

- **Branches now in use.** ClimaCore `~/worktree/ClimaCore.jl/multi-col-extras` (the PR
  branch; replaces `col-intp`, whose additions it contains: `all_nodes`,
  `quadrature_style`, `node_horizontal_length_scale`, `deep` kwarg, `field2arrays`, the
  multi-column unit test). ClimaUtilities `~/worktree/ClimaUtilities.jl/forcing-intp` at
  commit `1570399` ("Add regridding vertical profiles onto 3D spaces"), the user's version
  of the `horizontally_uniform` regridder. Both are dev'd into the Julia 1.11 scratch
  environment (§0b) via `Pkg.develop(path = ...)`; `.buildkite/Manifest.toml` (Julia 1.12)
  still points at `col-intp`, the user maintains it.
- **Unit horizontal metric (ClimaCore commit `a8ae99ac`, the user).** `MultiPointGrid`
  now has `J = 1`, `∂x∂ξ = I` at every point; the radius only enters the global geometry.
  This is exactly the identity-metric grid of `multi_column_dev/identity_metric.jl`, so
  every "metric rounding" difference in checklist §7 disappears by construction: column 1
  at (0, 0) is expected to be bitwise identical to the single column, on CPU and GPU. The
  §7 real-grid columns and the B1 discussion are now historical. Anything that differs
  from here on is a real implementation difference, not rounding.
- **Regridder flag is derived, not hard-coded.** The new ClimaUtilities constructor
  rejects `horizontally_uniform = true` on a Z-only space, so
  `column_timevaryinginputs` (`src/column_datasets/ColumnDatasets.jl`) now passes
  `horizontally_uniform = !(target_space isa Spaces.FiniteDifferenceSpace)`. Found by
  the GPU run: ERA5-driven and diurnal failed at construction on the *single* column.
- **File-forced configs do not start on the GPU (pre-existing, noted, not fixed).**
  `update_surface_conditions!` broadcasts `surface_state_to_conditions` with `atmos`; with
  file forcing the `AtmosModel` holds a `ColumnDataset` and `Nudging{Tuple{Symbol,
  Symbol}}`, not isbits, so CUDA rejects the kernel argument at initialization, single
  column included. Passing `atmos.microphysics_model` (the only field read) fixes it; that
  change was applied for the ERA5 and diurnal GPU runs below and then reverted at the
  user's request ("if it is already broken, it is already broken"). Checklist §8 has the
  note; the GPU rows for those two configs hold only with the change applied.
- **ClimaCore eager stencil kernel and type-valued `Ref` (pre-existing, noted, not
  fixed).** The subsidence forcing stencil fails on the GPU with a dynamic
  `CuDynamicSharedArray` call because ClimaCore's device-side
  `unsafe_eltype(::CuRefType{T})` returns `T` where the host has `Type{T}`; changing it to
  `Type{T}` fixes it (`multi_column_dev/gpu_probe8.jl`). Applied for the ERA5 and diurnal
  GPU runs, then reverted at the user's request, same reasoning as the surface-conditions
  item; the `multi-col-extras` worktree is clean again. Checklist §8 has the full
  mechanism for a future ClimaCore issue.
- **PR readiness:** checklist §10 lists the dependency PRs (ClimaCore #2629 merged /
  #2635 open, ClimaDiagnostics #188 open, ClimaUtilities #251 open), the compat bumps,
  the `.buildkite/Manifest.toml` that must not ship, the missing NEWS entry, and the
  single-column CI breakage that the unconditional `horizontally_uniform` keyword causes
  against released ClimaUtilities 0.1.32.
- **User requirement recorded (checklist §0 item 2, now with the full analysis of ragged
  time axes, ClimaUtilities #241/#248/#251, and the "interpolate beforehand" vs "ragged
  kernel in ClimaInterpolations" trade-off):** multi-column runs must support
  different sites with their own forcing (ClimaLand `ColumnEnsemble` style); the
  same-forcing-everywhere case is the special case. Not implemented; see U2.
- **Scripts:** `multi_column_dev/gpu_hard3.jl` (the three cases; `HARD3_CASES` selects a
  subset), `gpu_err.jl` (now takes `GPU_ERR_OVERRIDES`, e.g. `Dict("t_end" => "1mins")`
  to reach a solve-time failure), `gpu_probe7.jl`, `gpu_probe8.jl`. Log:
  `multi_column_dev/logs/gpu_ladder.log` (pane pipe of `julia-mc`); full error dumps in
  `multi_column_dev/logs/gpu_error_<case>_<single|multi>.txt`.
- **GPU results (Julia 1.11.4, CUDA.jl 5.11.3, A100, Float32 configs, unit metric):**
  LARCFORM1, ERA5-driven, and the diurnal SCM case all match the single column within
  rtol 1e-9 (state, cache, NetCDF); ERA5-driven rerun at `rtol = atol = 0` is bitwise
  identical in state and in all cache entries except the A9 uninitialized `ᶜmp_tendency`
  scratch. Both GPU-only fixes were reverted afterwards, so these two rows hold only with
  them applied (LARCFORM1 needs neither). Details in checklist §9 "GPU results on the PR
  branches".
  The slurm reservation (job 257500) was cancelled afterwards, as asked; the `julia-mc`
  window is back to a plain shell. `julia-mc5` holds a CPU REPL on `.buildkite` (Julia
  1.12, ClimaCore `col-intp`) with the Atmos edits loaded; `julia-mc3` is the formatter.

## 0d. Fifth session (2026-09-14): `deep_atmosphere` on columns, static only

ClimaCore `multi-col-extras` dropped the `deep` kwarg from `CommonGrids.MultiColumnGrid`
(`MultiPointGrid` is unit-metric, shallow-spherical, always). Decision, with the user:
`deep_atmosphere` is not propagated to the multi-column space, exactly as on the single
column, and a deep single column and a deep multi-column must agree. Changes (no runs,
static analysis only, formatter not run):

- `src/simulation/grids.jl`, `src/config/type_getters.jl`: `MultiColumnGrid` no longer
  takes `deep_atmosphere`; `get_grid` no longer forwards it.
- `src/cache/cache.jl` `compute_coriolis`: any `ColumnSpace` (single or multi) uses the
  f-plane `f_plane_coriolis_frequency(params)`, not `2Ω sin(lat)`. Closes DIFF_SPACE P1
  for every latitude, not just the equator.
- `src/parameterized_tendencies/radiation/radiation.jl`: `planet_radius` (which turns on
  RRTMGP's (r/a)² area scaling) is passed only for `DeepSphericalGlobalGeometry`, so the
  shallow multi-column grid gets no scaling, like the Cartesian column. Closes the 2%
  gray-radiation offset (checklist A2) without an overlay. Sphere behaviour unchanged.
- `examples/multi_column/single_vs_multi_column.jl` no longer forces
  `deep_atmosphere: false`; `docs/src/configuration.md` and `test/grids.jl` updated.
- Known, not changed: `radiation_diagnostics.jl` scales the flux diagnostics by
  `geometric_scaling` whenever `radiation_mode.deep_atmosphere` is set, on any grid,
  so on both column kinds the diagnosed fluxes are inflated by up to (1 + z_top/a)² while
  RRTMGP applied no inverse scaling. Identical on both columns; a pre-existing
  single-column inconsistency to raise separately.

## 0e. Sixth session (2026-09-15): forcing per site with the ragged `TimeVaryingInput`

The user's request: ClimaUtilities `~/worktree/ClimaUtilities.jl/ragged-data-intp` (branch
`kp/ragged-data-intp`, 14 commits on v0.1.32) now has a multi-column, per-column-time-axis
`TimeVaryingInput` built from `DataSource`s; use it in Atmos for different forcing data per
site and verify on the CPU REPL. Full account in checklist §11; short version:

- **All column forcing now goes through the ragged input**, single column included:
  `ColumnDatasets.column_timevaryinginputs` builds `TimeVaryingInput(DataSource(path,
  var), space; start_date, method, preprocess_func)` (one file for every column) or
  `TimeVaryingInput([DataSource...], space; ...)` (one file per column); the surface
  series use the same call on the surface space. The `InterpolationsRegridder`, the
  `horizontally_uniform` flag (ClimaUtilities #251) and the format hook `extrapolation_bc`
  are gone from Atmos.
- **`ColumnDatasets.ColumnData = Union{AbstractColumnData, AbstractVector{<:ColumnDataset}}`**
  is what `ExternalDrivenTVForcing` and `ForcingFromFile` hold; a vector is one dataset per
  column in column order (positional, as in ClimaUtilities). The initial condition, being
  pointwise, matches each column to its file by location: `ForcingFromFile(datasets,
  start_date; sites)` with `sites = column_sites(parsed_args, FT)`; columns at one
  location must share their file.
- **Config**: `external_forcing_file` may list one file per column (`ForcingFromFile`);
  `ReanalysisTimeVarying` / `ReanalysisMonthlyAveragedDiurnal` on `config: multicolumn`
  take the sites from `column_latitudes` / `column_longitudes` (`era5_datasets`), not
  from `site_latitude` / `site_longitude` (behaviour change for existing multicolumn ERA5
  runs; to confirm with the user).
- **Environment.** `.buildkite/Manifest.toml` (the user's) now develops ClimaCore
  `multi-col-extras`, ClimaUtilities `ragged-data-intp`, ClimaDiagnostics `multi-cols`;
  the CPU REPLs `multi-col:0` (window `buildkite`) and `multi-col:1` (`buildkite-verify-`)
  were restarted on it with `--history-file=no` and Revise; `multi-col:2` is the pinned
  formatter (JuliaFormatter 2.10.1). The old `julia-mc*` session names no longer exist.
- **Verification**: `test/column_datasets_tests.jl` 100/100; input-level checks against the
  old path on the real ERA5 file (`multi_column_dev/ragged_inputs_check.jl`,
  `ragged_precision_check.jl`): surface series bitwise equal, profiles at the Float32
  rounding of the file data (one extra Float32 rounding in
  `ClimaUtilities.Utils.linear_interpolation` on a Float64 column; fix suggested for the
  ClimaUtilities branch in checklist §11); shared file on three columns bitwise equal to
  the single column. Full-run check `multi_column_dev/multisite_case.jl` (three columns
  reading files [A, B, A] against single-column runs on A and B) and the ERA5 two-site
  generation `multi_column_dev/era5_sites_check.jl`: both passed. In the 10-minute
  three-column run every column's final state is bitwise identical to its own
  single-column run (columns 1 and 3 to the run on file A, column 2 to the run on the
  warmer, two-hourly file B); the ERA5 check generated the two sites' files from the raw
  artifact in 74 s and built the setup with a 3-vector of datasets. A 3-hour rerun
  crossing file nodes (1 h and 2 h of A, 2 h of B) gives the same bitwise agreement in
  the state, and 69 of the 70 hourly NetCDF diagnostics agree bitwise per column; the one
  exception, the McICA cloud cover `clt`, is sampled by RRTMGP from the global random
  stream shared by the solver's columns (checklist §11), not a forcing issue.
- **Not done / open**: GCM cfsites and ARM per column, per-column `FileHeatFluxes`,
  `TimeVaryingInsolation` with an explicit site (all one-site-for-all, documented in
  checklist §11); NEWS entry (user); ClimaUtilities branch to become a PR and a release
  before Atmos CI can pass (checklist §10 item 3 updated).

## 1. Goal and ground rules (from the user)

- Goal: run N independent columns in one ClimaAtmos simulation on a
  `ClimaCore.Spaces.MultiColumnFiniteDifferenceSpace`, and show the result is identical
  to the single-column (`config: column`) run up to floating-point rounding, for every
  single-column configuration in `config/model_configs`. The bar: anything a single-column
  run can do, a multi-column run must be able to do.
- Minimal, idiomatic changes; widen dispatch, do not special-case "multicolumn" outside
  grid construction and config plumbing. Follow ClimaLand PR #1826 (`ColumnEnsemble`,
  local copy at `~/worktree/ClimaLand.jl/main`). Ask the user when unsure rather than
  guess. Fixing a bug in the single-column path found along the way is welcome.
- Allowed to edit: ClimaAtmos, ClimaCore `~/worktree/ClimaCore.jl/col-intp`,
  ClimaDiagnostics `~/worktree/ClimaDiagnostics.jl/multi-cols`, and (for forcing
  prototypes) ClimaUtilities `~/worktree/ClimaUtilities.jl/forcing-intp`. All three are
  dev'd in `.buildkite/Manifest.toml` (Julia 1.12.5; the user maintains this
  environment, do not re-instantiate it).
- Running is not enough: compare state, cache, and NetCDF output column by column
  against the single-column run (`examples/multi_column/comparison_utils.jl`), and do
  not rely on short-run differences alone.
- Work in persistent Julia REPLs inside tmux (`module load climacommon` first, then
  `julia --project=.buildkite --history-file=no`; never one-off `julia script.jl`). Up to
  ten tmux sessions may be used; no GPU. Poll runs often (`tmux capture-pane`, or the
  logs in `multi_column_dev/logs/`, which `tmux pipe-pane` fills).
- Do not commit; the user commits. Run the pinned formatter (`.dev/format`,
  JuliaFormatter 2.10.1, same pin in ClimaCore) on changed Julia and Markdown files
  before handing over; a formatter REPL is `julia --project=.dev/format` then
  `using JuliaFormatter; format([...])`.
- Keep this file and the checklist updated as items close.

## 2. State of the tree

### ClimaAtmos (this repo), uncommitted, formatted unless noted

- `src/simulation/grids.jl`: `MultiColumnGrid(FT; points, radius, context, z_*, z_mesh)`
  mirroring `ColumnGrid`; no `deep_atmosphere` (see §0d).
- `src/config/type_getters.jl`: `get_grid` multicolumn branch passes
  `deep_atmosphere = parsed_args["deep_atmosphere"]`; the ad-hoc list checks moved out.
- `src/config/model_getters.jl`: `check_case_consistency` asserts the two list rules
  (equal length, non-empty) for `config == "multicolumn"`; docstring reflowed.
- `src/utils/utilities.jl`: `issphere` via `Spaces.global_geometry` (A6).
- `src/diagnostics/core_diagnostics.jl`: `compute_rv` DSS guarded by `do_dss` (A7; also
  makes `rv` work on a single column, where it used to error).
- `src/types.jl`: `ShipwayHill2012VelocityProfile` compares `FT(t) < t1` (was an
  `ITime`-vs-`Float` MethodError on any single-column kinematic-driver run). **Not yet
  verified by a run**: `multi_column_dev/queue1b.jl` is queued in REPL 1.
- Tests: `test/grids.jl` (MultiColumnGrid testset; passed, 84 assertions),
  `test/config/model_from_config.jl` (`check_case_consistency`; passed),
  `test/diagnostics/unit_diagnostics.jl` (multicolumn fixture; being run via
  `Pkg.test` with `TEST_GROUP=diagnostics` in REPL 4, result unknown),
  `test/restart.jl` ("multicolumn" configuration; not run, MANYTESTS only),
  `test/restart_AtmosSimulation.jl` (MultiColumnGrid in the grid tuple; not run).
- CI: two steps in `.buildkite/full_pipeline.yml` ("Non-spherical" group) running
  `examples/multi_column/single_vs_multi_column.jl` on the hydrostatic and gray
  radiative configs. The hydrostatic one was run end to end in the REPL (12/12 tests
  pass). The gray one runs 654 model days twice (single + 3 columns); flag the cost to
  the user.
- Docs: `docs/src/configuration.md` subsection "Multiple independent columns";
  `MultiColumnGrid` in `docs/src/api.md`, `interfaces.md`, and the `AtmosSimulation`
  docstring. A local docs build (`tmux` session `docs`, `julia --project=docs`) failed at
  `docs/make.jl:35` while building the InterLinks inventories (before `makedocs`; likely
  no network access from the cluster). Not investigated.
- `examples/multi_column/single_vs_multi_column.jl`: header comment updated for the deep
  flag. `examples/multi_column/comparison_utils.jl` was modified by someone else during
  this session (it now prints a `maxerr` column); not reviewed here.
- `.buildkite/Manifest.toml` (tracked, with absolute dev paths for ClimaCore,
  ClimaDiagnostics, and now ClimaUtilities `forcing-intp`): the user maintains it. CI
  cannot use it as is (THINGS_TO_ADDRES §1.1). `.gitignore:28` has a trailing comment
  that disables the `*/Manifest*.toml` rule; left alone, decide before a PR.
- `multi_column_dev/`: the experiment ladder (see §5). `hb_bitwise.jl` and `hb_1day.yml`
  there are not mine (another session, `julia-hb`, is doing a bit-level comparison of
  the hydrostatic case).

### ClimaCore `kp/col-intp`, uncommitted, formatted

`src/Spaces/multicolumn.jl`: `quadrature_style`, `node_horizontal_length_scale`,
`all_nodes` for `MultiPointSpace` (parity with `PointSpace`; C1-C3).
`src/MatrixFields/field2arrays.jl`: `all_columns` also accepts
`MultiColumnFiniteDifferenceField`. `src/CommonGrids/CommonGrids.jl` and
`src/CommonSpaces/CommonSpaces.jl`: `deep::Bool = false` keyword (C4).
`test/Spaces/unit_multicolumn.jl` (new, registered in `test/runtests.jl`; passed, 31
assertions), NEWS entry. The REPL shim `multi_column_dev/shims.jl` is now redundant.

### ClimaUtilities `forcing-intp`, uncommitted, not formatted (prototype)

`ext/InterpolationsRegridderExt.jl` `regrid`: when the target coordinates are
`LatLongZPoint` and the data has one dimension, interpolate at each point's `z` (one
site's profile applied to every column). Test added to `test/regridders.jl`
("Z-only data on a LatLongZ space"), NEWS entry. Not run through the ClimaUtilities test
suite. Dev'd into `.buildkite` from REPL 8; the ARM VARANAL comparison with it was
running (`multi_column_dev/queue8.jl`) when this handoff was written.

### ClimaDiagnostics `kp/multi-cols`

No changes were needed; the writer handles multi-column spaces (v0.3.10).

## 3. What is established, and how

Everything below used three columns at (0, 0), (0, 0), (30, -50); Float64 and
`deep_atmosphere: false` unless noted; the CI path (`config: multicolumn` through
`CA.get_simulation`). Numbers and per-case rows are in the checklist §7.

- t = 0 diagnostics (`t0_diag.jl`, `t0_case.jl`): for Bomex, DYCOMS RF01, and RICO the
  initial state (except a 2e-16 rounding in `uₕ`, whose covariant components carry the
  spherical metric), the cache after `set_precomputed_quantities!`, `implicit_tendency!`,
  and `remaining_tendency!` agree to 1e-13. Whatever differs is created inside the time
  step (implicit solve, limiters, or the second stage).
- Sensitivity test (`sens.jl`): rerun the *single column* with its initial `uₕ` (or,
  with `SENS_ALL = true`, every prognostic field) multiplied by `1 + 1e-15`, and compare
  with the unperturbed single column after 1 h. If the difference table equals the
  multi-column one, the multi-column difference is the model's own rounding sensitivity.
  Results: Bomex and DYCOMS RF01 identical tables (every row equal); TRMM and TRMM 0M same
  fields and magnitudes; Bomex fixed-TKE 14 vs 12 entries; Soares 1 entry both ways;
  Bomex mixing-length cloud 8 entries (whole state) vs 2 (multi). **RICO: 0 entries under
  both perturbations vs 14 for the multi-column run.** Limits of the test: a uniform
  factor on the whole state leaves ratios such as `ρq_tot/ρ` unchanged, so `SENS_ALL` is
  effectively a `ρ` and `uₕ` perturbation; a per-element random one-ulp perturbation was
  not run. A DYCOMS RF01 run perturbed only in `ρ` showed nothing (1 entry at 1e-9),
  so the perturbation must hit the field the model is sensitive to.
- Uninitialized cache (A9): fields allocated with `similar` (`ᶜmp_tendency`,
  `ᶜsgs_moments`, `ᶠradiation_flux`, `ᶜκρq`, `ᶜρa_tendencyʲs`, ...) hold recycled memory
  until first written. In a warm REPL, t = 0 comparisons show "single nonzero, multi
  exactly zero" entries with `err = 1` for them; after a run only never-written ones
  remain (`ᶜmp_tendency.e_tot_hlpr`/`dq_*_dt` for some configs, `ᶜsgs_moments.sigma_S`
  when quadrature is off), and they appear in the bitwise duplicate-column check.
- Latitude-dependent physics (A3): column 3 at lat 30 differs in Coriolis (`ᶜf³`),
  default surface temperature (14 K), insolation, and everything downstream. Expected.
- Deep atmosphere (A2): the multi-column grid now honours `deep_atmosphere` like the
  sphere; a comparison against the Cartesian single column needs `deep_atmosphere:
  false` (the driver and the `shallow*.yml` overlays set it).
- File forcing (U1/U2): GCM-driven runs unchanged and matches exactly in Float64 (Float32:
  1e-6..1e-5 rounding growth); its forcing cache fields match to 1e-9. ARM VARANAL's
  canonical file has only `(z, time)`, which hits the regridder gap fixed by the
  prototype. ClimaUtilities 0.1.32 already supports per-column time series
  (`TimeVaryingInput` with matrix input), the first piece of per-column forcing.
  On this cluster ARM VARANAL also needs `ENV["BUILDKITE"] = "true"` so the converted
  forcing file goes to a temporary directory (the artifact directory is read-only).
- 2M precipitation fails on a single column too (disabled upstream, CloudMicrophysics
  0.37). The diurnal SCM case fails on the `ReanalysisMonthlyAveragedDiurnal` assertion
  `config == "column"` (`model_getters.jl:853`, A8): decide with the user whether to
  widen it to both column kinds once ARM VARANAL (the same forcing machinery) matches.

## 4. Open items, in order

### 4.1 RICO (real difference; highest priority)

Facts: t = 0 all agree to 1e-13 (see above). After one 120 s step column 1 differs from
the single column in `sgsʲs.1.mse` (7.35e-5 relative, max 4.8 J/kg), `sgsʲs.1.q_tot`
(8e-5) and a 1e-15-level `sgsʲs.1.u₃`; the grid mean is untouched. After 3 steps
`sgsʲs.1.ρa` differs by 0.25 and the grid mean by 1e-7..1e-5; after 1 h 14 of 18 state
entries differ (`ρtke` 0.6, `ρe_tot` 2e-3) and the surface fluxes by 1e-4. The single
column perturbed by 1e-15 in `uₕ`, or uniformly in the whole state, reproduces itself to
1e-9 after 1 h (REPL 6 log). Bomex shows the same first-step signature but *is*
reproduced by the `uₕ` perturbation; GABLS (same SCM Coriolis path) matches exactly;
DYCOMS RF02 (bulk fluxes, drizzle) matches exactly.

What is specific to RICO in the code: `src/setups/Rico.jl`: `MoninObukhov(; z0 = 1.5e-4)`
with no prescribed fluxes (bulk fluxes from a prescribed 299.8 K surface, so the
SurfaceFluxes solver runs from the interior state), `coriolis_param = 4.5e-5` with
geostrophic profiles, `large_scale_advection_forcing` and `subsidence_forcing` from
AtmosphericProfilesLibrary, 1M microphysics, `z_elem: 100`, `dt: 120secs`.

Suggested plan (none of it started):

1. Run `sens.jl` for RICO with a per-element random perturbation (multiply `parent(f)` of
   every field by `1 .+ 1e-15 .* randn(size)`), and with a perturbation only in
   `ρe_tot`/`ρq_tot`, to close the "which field is it sensitive to" gap. If any of these
   reproduces the multi-column table, RICO joins the rounding group; if none does, go on.
2. Bisect inside the first step. `t0_case.jl` shows both tendency functions agree at
   t = 0, so compare the pieces of the implicit stage: (a) the Jacobian blocks
   (`ManualSparseJacobian`, `approximate_linear_solve_iters: 2`; look for metric terms,
   `g³³_field`, `J`, or `Δz` derived from the 3-D local geometry, which on the spherical
   grid carries `R²cosd(lat)(π/180)²`); (b) the linear solve result for a fixed
   right-hand side; (c) the state after the first implicit stage and after
   `constrain_state!`/the EDMF filter. Compare each between single and column 1 with
   `field_diffs`. `Fields.column(f, 1, 1, h)` extracts column h of a multi-column field.
3. Check the SurfaceFluxes call on a column extracted from the multi-column grid: its
   space is a `FiniteDifferenceSpace` with `LatLongZPoint` coordinates and the 3-D
   metric (DIFF_SPACE §4.4); anything that reads `Δz`, `J`, or a local-geometry
   component there behaves differently from the Cartesian column.
4. Whatever the cause, add the RICO row to the checklist with the evidence.

### 4.2 Runs that were in flight (check the logs first, do not rerun blindly)

| REPL / session | Running or queued | Where to look |
|---|---|---|
| `julia-mc` (REPL 1) | `queue1.jl`: `tv_era5driven` (expected to fail, no ERA5 raw artifact); then `queue1b.jl`: kinematic driver after the `src/types.jl` fix | `multi_column_dev/logs/repl_20260909.log` |
| `julia-mc3` | `step4.jl` remainder: `bomex_tracerA`, `rico_2M` (expect single-column failure too), `bomex_sparse_autodiff`, `gabls_sparse_autodiff` (exercise `column_index_iterator`) | `logs/repl3_*.log` |
| `julia-mc4` | `Pkg.test("ClimaAtmos")` with `ENV["TEST_GROUP"] = "diagnostics"` (unit diagnostics with the multicolumn fixture; direct `include` failed only because `RRTMGP` is not a direct dep of `.buildkite`) | `logs/repl4_*.log`, marker `PKGTEST DIAGNOSTICS DONE` |
| `julia-mc7` | `sens.jl` for `bomex_tracerA` | `logs/repl7_*.log` |
| `julia-mc8` | `queue8.jl`: ARM VARANAL Float32 then Float64, single vs multi, with the ClimaUtilities prototype | `logs/repl8_*.log` |
| `docs` | failed (see §2) | `logs/docs_*.log` |
| `julia-mc2`, `julia-mc5`, `julia-mc6` | idle | |

### 4.3 Then

- Record the outcomes of 4.2 in the checklist §7 (rows for `bomex_tracerA`, `rico_2M`,
  the two `*_sparse_autodiff`, `tv_era5driven`, `kinematic_driver`, ARM VARANAL).
- Run the pinned formatter on `src/types.jl`, `multi_column_dev/*.jl` added since the
  last pass (`bomex_diag.jl`, `growth_case.jl`, `queue*.jl`, `sens.jl`, `t0_case.jl`),
  and the ClimaUtilities files.
- Decide with the user: A8 (`ReanalysisMonthlyAveragedDiurnal` assertion), the gray CI
  step cost, the tracked manifest, `.gitignore:28`, whether to zero-initialize the
  `similar` cache fields (A9) so the bitwise duplicate-column check in CI cannot trip on
  them, and whether the ClimaUtilities prototype should become a PR.
- Update the checklist §1 summary; hand over.

## 5. How the scripts fit together (`multi_column_dev/`)

- `common.jl`: loads ClimaAtmos; `build_simulation(files, job_id; points, overrides)`
  runs `CA.get_simulation` with `config: multicolumn` and `column_latitudes/longitudes`
  set from `points` (vector of `(lat, long)`), `output_dir` under
  `multi_column_dev/output/`, and `overrides` merged into the parsed arguments;
  `run_case` builds and solves; `show_failure` prints a truncated stack. Includes
  `compare.jl`, which includes `examples/multi_column/comparison_utils.jl` and defines
  `compare_runs` (duplicate-column bitwise check plus single-vs-each-column for `Y` and
  `p`) and `diagnostic_diffs` for NetCDF.
- Overlays: `shallow.yml`/`shallow64.yml` (`deep_atmosphere: false`, no checkpoints,
  Float64 for the latter); `short*.yml`; `short_edmf.yml` (1 h, Float64, shallow);
  `short_forcing.yml` (10 min); `short_kid.yml` (2 min, Float64, shallow);
  `bomex_*step.yml`; `checkpoint.yml`, `pressure_diags.yml`, `fake_plev.yml`.
- Ladder: `step1.jl` hydrostatic; `step2b.jl`/`step2c.jl` radiative equilibrium (default
  and shallow); `step3.jl` NOGW and precipitation (Float64); `step4.jl` EDMF (set
  `EDMF_CASES` first); `step5.jl` checkpoint/restart, output variants, GCM-driven and ARM
  VARANAL; `queue1.jl` kinematic driver, LARCFORM1, diurnal SCM, ERA5-driven;
  `queue1b.jl` kinematic driver rerun; `queue4.jl` CI driver end to end, Bomex 1 h, unit
  tests, ARM VARANAL, GCM-driven Float64; `queue8.jl` ARM VARANAL with the prototype.
- Diagnostics: `t0_diag.jl` (Bomex) and `t0_case.jl` (set `CASE`) compare t = 0 state,
  cache, and tendencies; `bomex_diag.jl` one-step single vs multi vs perturbed single;
  `growth.jl` (Bomex) and `growth_case.jl` (set `CASE`) 1/3/10-step growth; `sens.jl`
  (set `SENS_CASES`, optionally `SENS_ALL = true`) the sensitivity test of §3.
- Markers to grep for: `STEP<n> CASE DONE: <name>`, `STEP<n> DONE`, `QUEUE<n> CASE
  DONE`, `SENS CASE DONE`, `T0 CASE DONE`, `GROWTH CASE DONE`, and `FAILED`.
- Each REPL holds 4-8 GB after a few cases; the machine has 1 TB.

## 6. Reference code

- ClimaLand: `~/worktree/ClimaLand.jl/main`: `src/shared_utilities/Domains.jl`
  (`ColumnEnsemble`), `experiments/integrated/era5/comparison_utils.jl` and
  `column_ensemble_comparison.jl`, `.buildkite/pipeline.yml` "ERA5 Column vs
  ColumnEnsemble".
- ClimaCore `kp/col-intp`: `src/Spaces/multicolumn.jl`, `src/Grids/multipoint.jl`,
  `src/CommonGrids/CommonGrids.jl` (`MultiColumnGrid`),
  `src/Remapping/interpolate_pressure.jl`, `test/Spaces/unit_multicolumn.jl`.
- ClimaDiagnostics `kp/multi-cols`: `src/netcdf_writer_coordinates.jl`,
  `src/netcdf_writer.jl`, `NEWS.md` (v0.3.10).
- ClimaUtilities `forcing-intp`: `ext/InterpolationsRegridderExt.jl` (`regrid`),
  `ext/TimeVaryingInputsExt.jl`; registered 0.1.32 is what the other REPLs loaded.

## 0f. Seventh session (2026-09-16): forcing per site implemented and verified

`MULTI_COL_FORCING.md` Part 3 was implemented as eight commits on `kp/multi-col`
(`ec4a5facb`..`94c354886`, one per step; see Part 4 of that file for the deviations) and
verified with the two-single-plus-one-multi template on the CPU (ERA5 `ForcingFromFile` at
10 minutes and 3 hours, GCM, ARM, ERA5 site lists) and on one A100 (ERA5, GCM): every
column's state is bitwise its own single-column run; the before/after runs of the RRTMGP
guard are in checklist §12. Uncommitted in the working tree, for the GPU runs only: the
surface-conditions kernel argument (§0c). GPU environment: Julia 1.12.5 still fails in
ClimaCore's eager stencil kernel; Julia 1.11.4 with `<scratchpad>/env111c` (built from
`.buildkite/Manifest-v1.11.toml`, worktrees developed) works, but the subsidence stencil of
`external_forcing.jl:229` needs the GPU-only override `multi_column_dev/gpu_subsidence_override.jl`
with the current ClimaCore worktree, and the ARM case is blocked by a microphysics kernel
receiving `atmos` (checklist §12). Dev scripts: `multi_column_dev/{multisite_case,
gcm_multisite_case, arm_multisite_case, era5_lists_case, gpu_multisite, gpu_gcm, gpu_arm}.jl`
and `multisite_compare.jl` (the case script must set `COLUMN_MAP`).
