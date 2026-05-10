Offline quadrature experiments
=================================

Purpose
- Run a high-resolution "truth" forward run of ClimaAtmos and save final thermodynamic
  and variance fields required for offline quadrature.
- Perform offline quadrature on the saved final profiles regridded to several vertical
  resolutions and with varying `N_quad` and quadrature methods.
- Produce summary plots comparing condensate profiles across methods and resolutions.

Layout
- `scripts/run_highres_truth.jl` : Run a high-res forward run and save final profile.
- `scripts/offline_quadrature.jl` : Read saved profile, regrid, run quadrature methods.
- `scripts/plot_quadrature_results.jl` : Plot summary PNGs into `outputs/`.
- `outputs/` : Default output directory for nets and PNG results.

Quick start
===========

**COMPLETE REPL WORKFLOW** (must run in order):

```julia
# 1. Navigate to ClimaAtmos.jl repo root
cd("/path/to/ClimaAtmos.jl")  # OR: cd(ENV["HOME"] * "/Research_Schneider/CliMA/ClimaAtmos.jl")

# 2. Activate the variance_adjustments project
using Pkg
Pkg.activate("calibration/experiments/variance_adjustments")

# 3. Include the end-to-end orchestrator
include("calibration/experiments/variance_adjustments/offline_quadrature/scripts/run_end_to_end.jl")

# 4. Run with defaults (sequential, Bomex case, [1,2,3,4,5] quadrature orders)
run_offline_quadrature_end_to_end!()
```

This workflow is self-contained under `offline_quadrature/` and uses the local
registry at `offline_quadrature/registries/forward_sweep_cases.yml` plus the local
SCM TOML at `offline_quadrature/toml/prognostic_edmfx.toml`.

**Output files** (in `offline_quadrature/outputs/<case>/`):
- `highres_profile_<tag>.nc` — High-res truth profiles (NetCDF)
- `quadrature_results_<tag>.jld2` — Offline quadrature results (JLD2)
- `quadrature_summary_<tag>.png` — Summary comparison plot (PNG)

Function signature & parameters
================================

```julia
run_offline_quadrature_end_to_end!(;
    forward_cfg::ForwardSweepConfig = _offline_quadrature_forward_cfg(),  # defaults to local registry
    outdir::String = joinpath(@__DIR__, "..", "outputs"),                # output root directory
    tag::String = Dates.format(now(), "yyyymmdd_HHMMSS"),               # timestamp for results
    truth_quadrature_order::Int = 5,                                      # resolution level for truth run
    quadrature_orders::Vector{Int} = [1, 2, 3, 4, 5],                   # quadrature order sweep
    parallel::Symbol = :sequential,                                       # execution mode: :sequential, :threads, or :distributed
    distributed_workers::Int = 1,                                         # number of worker processes (for :distributed)
    distributed_worker_threads::Int = 1,                                  # threads per worker (for :distributed)
)
```

**Execution modes:**

1. **Sequential** (single-threaded, default):
```julia
run_offline_quadrature_end_to_end!()  # Uses :sequential by default
```

2. **Multi-threaded** (use Julia's built-in thread pool):
```julia
# Start Julia with: julia --project=... -t 4
run_offline_quadrature_end_to_end!(parallel = :threads)  # Will use 4 threads (set at startup)
```

3. **Distributed** (spawn separate worker processes):
```julia
run_offline_quadrature_end_to_end!(
    parallel = :distributed,
    distributed_workers = 4,                # spawn 4 workers
    distributed_worker_threads = 2          # each worker gets 2 threads
)
```

**Custom outputs and quadrature sweep:**

```julia
# Run with custom output directory and quadrature orders
run_offline_quadrature_end_to_end!(
    outdir = "/path/to/custom/output",
    tag = "my_experiment_v1",
    truth_quadrature_order = 6,              # higher resolution truth
    quadrature_orders = [1, 2, 3, 4, 5, 6]  # longer sweep
)

# Run only 2 quadrature orders
run_offline_quadrature_end_to_end!(
    quadrature_orders = [3, 5]
)
```

2. Manual step-by-step (if you want to run each stage separately).


```julia
include("calibration/experiments/variance_adjustments/offline_quadrature/scripts/run_highres_truth.jl")
run_highres_truth_profile_from_case_layers!(
  ["../../../config/model_configs/master_column_varquad_diagnostic_edmfx.yml",
   "../../../config/model_configs/bomex_column_varquad_hires.yml"],
  "offline_quadrature/toml/prognostic_edmfx.toml";
  case_name = "Bomex",
  outpath = "calibration/experiments/variance_adjustments/offline_quadrature/outputs/highres_profile.nc",
  quadrature_order = 5,
  experiment_dir = "calibration/experiments/variance_adjustments",
)
```

3. Run offline quadrature

```julia
include("calibration/experiments/variance_adjustments/offline_quadrature/scripts/offline_quadrature.jl")
run_offline_quadrature_from_netcdf!(
  "calibration/experiments/variance_adjustments/offline_quadrature/outputs/highres_profile.nc";
  outpath = "calibration/experiments/variance_adjustments/offline_quadrature/outputs/quadrature_RESULTS.jld2",
)
```

4. Plot

```julia
include("calibration/experiments/variance_adjustments/offline_quadrature/scripts/plot_quadrature_results.jl")
run_plot_quadrature_results!(
  "calibration/experiments/variance_adjustments/offline_quadrature/outputs/quadrature_RESULTS.jld2";
  outpath = "calibration/experiments/variance_adjustments/offline_quadrature/outputs/summary.png",
)
```

Notes
- `offline_quadrature.jl` requires canonical fields in the input NetCDF: `z`, `T`, `qv`, `p`, `q_var`, `T_var`, `corr_Tq`.
- No fallback or approximate moment reconstruction is used.
