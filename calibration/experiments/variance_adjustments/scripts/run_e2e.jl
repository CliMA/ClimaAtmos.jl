# End-to-end pipeline in one Julia process (one precompile).
#
# What this does (single slice only):
#   1) Optional `Pkg.instantiate`
#   2) EKI calibration into `output_dir` — **requires** existing **`observations.jld2`** at **`observations_path`**
#      (place it there from your workflow before running; EKI cannot start without **y**).
#   3) Unless **`VA_SKIP_REFERENCE=1`**: regenerate **`observations.jld2`** via **`reference_truth_from_eki`**
#      (SCM + EKI member **`parameters.toml`**) — runs **after** calibration so EKI output exists.
#
# **One command (no exports required):**
#   julia --project=. scripts/run_e2e.jl
#   julia --project=. scripts/run_e2e.jl --config experiment_configs/experiment_config_gcm_cfsite23_N3_varfix_off.yml --calib-workers=20
#   ./run_e2e.sh --help
#
# Defaults (only if not already set in the environment): `VARIANCE_CALIB_WORKERS` = min(20, CPU−1),
# `VARIANCE_CALIB_WORKER_THREADS` = 1. Override with flags or `export ...` before the command.
#
# What this does **not** do:
#   - No figures — use `include("analysis/plotting/run_post_analysis.jl"); va_run_post_analysis!()` after a run.
#   - No grid over `quadrature_order` or varfix — those are fixed in `experiment_config.yml` (or
#     `VA_EXPERIMENT_CONFIG`). Forward-only grids: `scripts/sweep_forward_runs.jl`. Several EKI YAMLs:
#     `scripts/run_calibration_sweep.jl` or rerun with different `--config` / `VA_EXPERIMENT_CONFIG`.
# For the multi-YAML README workflow (EKI sweep + figures, optional forward grid), use `scripts/run_full_study.jl`.
#
# REPL (from experiment root, after `Pkg.activate(".")`): `empty!(ARGS); include("scripts/run_e2e.jl")`
# (CLI flags were already parsed at parse time; use env vars or re-launch with `ARGS`.)
#
import Pkg

const _VA_ROOT = dirname(@__DIR__) |> abspath

function _e2e_print_help()
    println("""
Usage:
  julia --project=. scripts/run_e2e.jl [options]

Options:
  --config=FILE | --config FILE   Set which experiment YAML to use (same as VA_EXPERIMENT_CONFIG).
  --calib-workers=N               EKI Distributed worker count (VARIANCE_CALIB_WORKERS).
  --calib-worker-threads=N        Threads per worker SCM (VARIANCE_CALIB_WORKER_THREADS), default 1.
  --calib-backend=worker|julia    VARIANCE_CALIB_BACKEND (default: worker).

  --skip-instantiate              VA_SKIP_INSTANTIATE=1
  --skip-reference                VA_SKIP_REFERENCE=1
  --skip-calibration              VA_SKIP_CALIBRATION=1

  -h, --help

If `VARIANCE_CALIB_WORKERS` is unset, it defaults to min(20, max(1, Sys.CPU_THREADS - 1)).
If `VARIANCE_CALIB_WORKER_THREADS` is unset, it defaults to 1.
""")
    return nothing
end

function _parse_e2e_cli!(argv::Vector{String})
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "-h" || a == "--help"
            _e2e_print_help()
            exit(0)
        elseif a == "--skip-instantiate"
            ENV["VA_SKIP_INSTANTIATE"] = "1"
        elseif a == "--skip-reference"
            ENV["VA_SKIP_REFERENCE"] = "1"
        elseif a == "--skip-calibration"
            ENV["VA_SKIP_CALIBRATION"] = "1"
        elseif a == "--config"
            i += 1
            i > length(argv) && error("--config requires a path")
            ENV["VA_EXPERIMENT_CONFIG"] = argv[i]
        elseif startswith(a, "--config=")
            ENV["VA_EXPERIMENT_CONFIG"] = split(a, '=', limit = 2)[2]
        elseif startswith(a, "--calib-workers=")
            ENV["VARIANCE_CALIB_WORKERS"] = split(a, '=', limit = 2)[2]
        elseif a == "--calib-workers"
            i += 1
            i > length(argv) && error("--calib-workers requires N")
            ENV["VARIANCE_CALIB_WORKERS"] = argv[i]
        elseif startswith(a, "--calib-worker-threads=")
            ENV["VARIANCE_CALIB_WORKER_THREADS"] = split(a, '=', limit = 2)[2]
        elseif a == "--calib-worker-threads"
            i += 1
            i > length(argv) && error("--calib-worker-threads requires N")
            ENV["VARIANCE_CALIB_WORKER_THREADS"] = argv[i]
        elseif startswith(a, "--calib-backend=")
            ENV["VARIANCE_CALIB_BACKEND"] = split(a, '=', limit = 2)[2]
        elseif a == "--calib-backend"
            i += 1
            i > length(argv) && error("--calib-backend requires worker|julia")
            ENV["VARIANCE_CALIB_BACKEND"] = argv[i]
        else
            error("Unknown argument: $(repr(a)). Try scripts/run_e2e.jl --help")
        end
        i += 1
    end

    if !haskey(ENV, "VARIANCE_CALIB_WORKERS")
        nw = min(20, max(1, Sys.CPU_THREADS - 1))
        ENV["VARIANCE_CALIB_WORKERS"] = string(nw)
    end
    if !haskey(ENV, "VARIANCE_CALIB_WORKER_THREADS")
        ENV["VARIANCE_CALIB_WORKER_THREADS"] = "1"
    end
    return nothing
end

_parse_e2e_cli!(ARGS)

Pkg.activate(_VA_ROOT)
include(joinpath(_VA_ROOT, "lib", "stdio_flush.jl"))
va_setup_stdio_flushing!()
if get(ENV, "VA_SKIP_INSTANTIATE", "") != "1"
    Pkg.instantiate()
end

include(joinpath(_VA_ROOT, "lib", "eki_calibration.jl"))
if get(ENV, "VA_SKIP_CALIBRATION", "") != "1"
    run_variance_calibration!(va_eki_calibration_options_from_env())
else
    @info "Skipping calibration (VA_SKIP_CALIBRATION=1)"
end
va_flush_stdio()

include(joinpath(_VA_ROOT, "lib", "reference_generation.jl"))
if get(ENV, "VA_SKIP_REFERENCE", "") != "1"
    generate_observations_reference!()
else
    @info "Skipping reference run (VA_SKIP_REFERENCE=1)"
end
va_flush_stdio()

@info "Pipeline finished" root = _VA_ROOT workers = ENV["VARIANCE_CALIB_WORKERS"] worker_threads =
    ENV["VARIANCE_CALIB_WORKER_THREADS"] config = get(ENV, "VA_EXPERIMENT_CONFIG", "config/experiment_config.yml")
va_flush_stdio()
