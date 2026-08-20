---
name: run-and-time
description: |
   Time, benchmark, and profile individual ClimaAtmos GPU components using
   the harnesses in perf/run-and-time/ (timer_harness.jl, quickprof_harness.jl,
   prof_harness.jl). Use when the user wants to measure GPU execution time of
   a specific ClimaAtmos component or kernel, check a change for a performance
   regression, get per-kernel launch statistics, capture Nsight Compute (ncu)
   or Nsight Systems (nsys) profiles, or create a new component timing case in
   perf/run-and-time/models/. Triggers on mentions of perf/run-and-time,
   GPU benchmarking, kernel timing, or nsys/ncu profiling of ClimaAtmos code,
   even if the user doesn't name a harness explicitly.
---

# run-and-time — ClimaAtmos GPU component timing & profiling

## What this is

`perf/run-and-time/` contains GPU timing and profiling harnesses for *isolated*
ClimaAtmos components (tendencies, cache-filling functions, cloud fraction).
Each harness runs a "case" file from `models/` that builds just enough model
state (`Y`/`p`/`t`) to call one component — no integrator, callbacks,
diagnostics, or output writers. The committed documentation is
`perf/run-and-time/README.md`; this skill distills it and the harness sources.

Use it to:
- measure average GPU execution time of a component (e.g. before/after a change),
- get a quick per-kernel launch breakdown in the terminal,
- capture Nsight Compute / Nsight Systems profiles of a component,
- add a new component case.

## Ground rules

- Run on a machine with an NVIDIA GPU and `export CLIMACOMMS_DEVICE=CUDA`.
- Always work **from inside** `perf/run-and-time/`. The harnesses and case
  files use relative includes (`include("atmos_config.jl")`) and `models/...`
  paths are relative to it.
- One-time environment setup:

  ```sh
  cd perf/run-and-time
  julia --project -e 'using Pkg; Pkg.instantiate()'
  ```

  The local `Project.toml` path-sources ClimaAtmos from the repo root
  (`[sources] ClimaAtmos = {path = "../.."}`), so edits under `src/` are
  measured immediately — no `dev` step needed.
- Prefer Julia 1.11.x (repo norm).

## Choose a harness

All harnesses take exactly one argument: the path to a case file that defines
`case_setup()` and `case_run(state)`.

| Goal | Harness | Invocation |
|---|---|---|
| Average GPU time + setup/compile times (regression checks) | `timer_harness.jl` | `julia --project timer_harness.jl models/<case>.jl` |
| Quick per-kernel launch summary in the terminal | `quickprof_harness.jl` | `julia --project quickprof_harness.jl models/<case>.jl` |
| Detailed kernel profile in Nsight Compute | `prof_harness.jl` | `ncu` command below |
| Execution timeline in Nsight Systems | `prof_harness.jl` | `nsys` command below |

### Timing (BenchmarkTools)

```sh
julia --project timer_harness.jl models/hyperdiff_tendency.jl
```

Reports setup time, warmup (compilation) time, and mean GPU time in ms from
`BenchmarkTools.@benchmark` around `CUDA.@sync case_run(s)`. This is the
default tool for "did my change make this component faster or slower?"

### Quick kernel stats (CUDA.jl profiling)

```sh
julia --project quickprof_harness.jl models/hyperdiff_tendency.jl
```

Prints the raw `CUDA.@profile` table plus a per-unique-kernel summary (grid,
block, shmem, registers, launch count).

### Nsight Compute

```sh
ncu -o output.ncu-rep --import-source 1 --profile-from-start=off --set=full \
  julia --project prof_harness.jl models/<case>.jl
```

### Nsight Systems

```sh
nsys profile -o output.nsys-rep \
  --capture-range=cudaProfilerApi --trace=nvtx,cuda,osrt --gpu-metrics-device=cuda-visible --cuda-memory-usage=true \
  julia --project prof_harness.jl models/<case>.jl
```

`prof_harness.jl` wraps the measured call in `CUDA.@profile external=true`,
which is what makes `--profile-from-start=off` / `--capture-range=cudaProfilerApi`
work: only the component call is captured, not Julia's setup and compilation.

nsys/ncu must use the same CUDA toolkit version as Julia's CUDA runtime.
Check/set Julia's with e.g.:

```sh
julia --project -e 'using CUDA; CUDA.set_runtime_version!(v"12.2")'
```

## Existing cases (`models/`)

| Case file | Component exercised |
|---|---|
| `hyperdiff_tendency.jl` | `CA.hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)` |
| `implicit_tendency.jl` | `CA.implicit_tendency!(Yₜ, Y, p, t)` |
| `set_cloud_frac.jl` | `CA.set_cloud_fraction!(Y, p, microphysics_model, cloud_model)` |
| `set_CM_cache.jl` | `CA.set_microphysics_tendency_cache!(Y, p, microphysics_model, turbconv_model)` |
| `sgs.jl` | `CA.set_sgs_moments_and_cloud_fraction!(Y, p)` |

`models/atmos_config.jl` is a shared helper (included by every case) that
provides `get_base_atmos_config()` — the single source of truth for the base
`AtmosConfig` these cases start from.

## Fast iteration

Have `Revise.jl` installed in the base (global) Julia environment. Launch the
harness interactively with `-i`:

```sh
julia --project -i timer_harness.jl models/<case>.jl
```

After the run it drops into a REPL. Edit the component's source, then re-measure
without paying setup+compilation again:

```julia
include("timer_harness.jl")
```

(The harnesses `try using Revise` at startup; with it loaded, edits under `src/`
are picked up between `include` calls.)

## Authoring a new component case

A case file must define exactly two functions (the harnesses error out unless
both are `isdefined` after `include`):

- `case_setup()` — builds and returns the state the component needs,
  conventionally a `NamedTuple` like `(; Y, p, t)` plus any preallocated
  buffers. Runs once, untimed.
- `case_run(state)` — calls the component under test. Called many times on the
  same state, so it must be re-runnable; preallocate anything heavy in
  `case_setup`, not here.

Recipe (use `models/hyperdiff_tendency.jl` for a tendency-style case or
`models/set_cloud_frac.jl` for a `Y`/`p`-only case as the template):

1. Start from the standard header:

   ```julia
   import ClimaComms
   ClimaComms.@import_required_backends
   import ClimaAtmos as CA
   import Random
   Random.seed!(1234)

   include("atmos_config.jl")   # provides get_base_atmos_config()
   ```

2. In `case_setup()`, get the base config and **stub out the expensive pieces
   the component never reads** by mutating `config.parsed_args` *before* any
   params/model/grid are built. This avoids RRTMGP lookup tables, ETOPO
   topography loading, aerosol/trace-gas ingestion, and gravity-wave tables —
   the bulk of setup time. Every committed case does:

   ```julia
   pa = config.parsed_args
   pa["rad"] = nothing                        # no RRTMGP model / lookup tables
   pa["aerosol_radiation"] = false
   pa["prescribed_aerosols"] = String[]       # no aerosol data ingestion
   pa["time_varying_trace_gases"] = String[]  # no trace-gas data ingestion
   pa["insolation"] = "idealized"             # unused once `rad` is off
   pa["non_orographic_gravity_wave"] = false  # no Beres source tables
   pa["orographic_gravity_wave"] = nothing
   pa["topography"] = "NoWarp"                # skip ETOPO load + smoothing
   pa["topo_smoothing"] = false
   ```

   Keep whatever knobs select the dispatch path you want to measure (e.g.
   `hyperdiff`, `turbconv = prognostic_edmfx`, `microphysics_model = 1M`,
   `cloud_model = QuadratureCloud`, `implicit_diffusion`, `rayleigh_sponge`) —
   see the "What is KEPT" comments at the top of the existing cases.

3. Build only `Y`/`p`/`t`, mirroring the `Y`/`p`/`t`-producing subset of
   `AtmosSimulation{FT}(...)` — never call `CA.get_simulation` (it drags in the
   integrator, Jacobian, callbacks, diagnostics, output writers). The sequence
   used by the committed cases: `CA.ClimaAtmosParameters` → `CA.get_setup_type`
   → `CA.get_atmos` → `CA.get_grid` → `CA.convert_time_args` → `CA.get_spaces`
   → `CA.Setups.initial_state` (+ `CA.Setups.overwrite_initial_state!`) →
   `CA.steady_state_velocity_from_config` → `CA.build_cache` (which also runs
   `set_precomputed_quantities!`, populating whatever precomputed quantities
   the kept components produce).

4. Preallocate tendency/scratch buffers in `case_setup` (e.g. `Yₜ = similar(Y)`)
   and return them in the state tuple.

5. `case_run(state)` calls the target function with exactly the state fields it
   needs and returns `nothing`.

6. Register the new file in `ci_test_models.jl`'s `case_files` set so CI smoke
   tests it.

## Verification

From `perf/run-and-time/`:

```sh
julia --project ci_test_models.jl
```

Every registered case must print `OK - it runs` and the script must exit 0 —
this is the CI smoke test for the cases. Then time the new case with
`timer_harness.jl`. Only compare timings taken on the same GPU and the same
Julia/CUDA versions.

## Gotchas

- Kernel names: `prof_harness.jl` and `quickprof_harness.jl` set
  `CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE=true` before loading CUDA, so
  kernels appear in profiler output with human-readable names derived from the
  stack trace (function, file, line). The setting is read in
  `ClimaCoreCUDAExt.__init__` — it has no effect if ClimaCore's CUDA extension
  is already loaded.
- `quickprof_harness.jl`'s `get_kernel_data` relies on CUDA.jl's *internal*
  profile representation (not public API; developed against CUDA.jl 5.11.3).
  It can break on any CUDA.jl version bump — if the summary fails, the raw
  `CUDA.@profile` table printed just before is still usable.
- Version matching: nsys/ncu and Julia's CUDA runtime must be on the same CUDA
  version, or external profiling capture fails.
- Artifacts: profile reports (`*.ncu-rep`, `*.nsys-rep`), `output/`, local
  manifests, and ad-hoc notes are intentionally not committed to git in this
  directory — keep new ones uncommitted too.
