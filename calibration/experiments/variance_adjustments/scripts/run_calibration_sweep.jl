# Run EKI calibration for each listed experiment YAML **in-process** (no extra `julia` subprocess per slice).
# Uses `Distributed.addprocs` / `rmprocs` inside `run_variance_calibration!` + cleanup after each slice so workers
# do not leak into the forward sweep.
#
#   julia --project=. scripts/run_calibration_sweep.jl
#
# One task for Slurm (0-based index):
#   CALIB_SWEEP_TASK_ID=0 julia --project=. scripts/run_calibration_sweep.jl
#   # or SLURM_ARRAY_TASK_ID
#
# Edit `lib/calibration_sweep_configs.jl` → `va_calibration_sweep_configs()` to choose slices.
#
# Flags:
#   --fail-fast    abort sequential sweep on first failed slice (default: log and continue)
#
import Pkg

const _EXPERIMENT_DIR = dirname(@__DIR__) |> abspath
Pkg.activate(_EXPERIMENT_DIR)

include(joinpath(_EXPERIMENT_DIR, "lib", "stdio_flush.jl"))
va_setup_stdio_flushing!()
if !isdefined(Main, :va_calibration_sweep_configs)
    include(joinpath(_EXPERIMENT_DIR, "lib", "calibration_sweep_configs.jl"))
end
if !isdefined(Main, :run_variance_calibration!)
    include(joinpath(_EXPERIMENT_DIR, "lib", "eki_calibration.jl"))
end

using Distributed

function run_calibration_slice_inprocess(config_relp::AbstractString, opts::EkiCalibrationOptions)
    old = get(ENV, "VA_EXPERIMENT_CONFIG", nothing)
    ENV["VA_EXPERIMENT_CONFIG"] = config_relp
    try
        run_variance_calibration!(opts)
    finally
        if old === nothing
            delete!(ENV, "VA_EXPERIMENT_CONFIG")
        else
            ENV["VA_EXPERIMENT_CONFIG"] = old
        end
        if nprocs() > 1
            rmprocs(workers())
        end
    end
    return nothing
end

function run_calibration_sweep!(opts::EkiCalibrationOptions = va_eki_calibration_options_from_env(); fail_fast::Bool = false)
    tid = get(ENV, "CALIB_SWEEP_TASK_ID", get(ENV, "SLURM_ARRAY_TASK_ID", ""))
    configs = va_calibration_sweep_configs()
    if !isempty(strip(tid))
        i = parse(Int, tid) + 1
        (1 <= i <= length(configs)) ||
            error("Task index $(i - 1) out of range; need 0:$(length(configs) - 1)")
        return run_calibration_slice_inprocess(configs[i], opts)
    end
    failed = String[]
    for c in configs
        @info "Calibration sweep cell" c
        try
            run_calibration_slice_inprocess(c, opts)
        catch err
            push!(failed, c)
            @error "Calibration slice failed" config = c exception = (err, catch_backtrace())
            va_flush_stdio()
            fail_fast && rethrow()
        end
    end
    if !isempty(failed)
        @warn "Calibration sweep finished with failures (continued)" n_failed = length(failed) failed
    end
    va_flush_stdio()
    return nothing
end

"""CLI entry when this file is run as `julia …/run_calibration_sweep.jl` (safe to `include` from `run_full_study.jl` without running)."""
function va_calibration_sweep_cli()
    fail_fast = any(==("--fail-fast"), ARGS)
    return run_calibration_sweep!(va_eki_calibration_options_from_env(); fail_fast = fail_fast)
end

if !isempty(Base.PROGRAM_FILE) && isfile(Base.PROGRAM_FILE) &&
   abspath(Base.PROGRAM_FILE) == abspath(@__FILE__)
    va_calibration_sweep_cli()
end
