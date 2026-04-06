# Run reference generation + EKI for each listed experiment YAML (separate subprocess per slice so
# Distributed workers do not leak between calibrations).
#
#   julia --project=. scripts/run_calibration_sweep.jl
#
# One task for Slurm (0-based index):
#   CALIB_SWEEP_TASK_ID=0 julia --project=. scripts/run_calibration_sweep.jl
#   # or SLURM_ARRAY_TASK_ID
#
# Edit `calibration_sweep_configs.jl` → `va_calibration_sweep_configs()` to choose slices.
#
# Flags:
#   --fail-fast    abort sequential sweep on first failed slice (default: log and continue)
#
import Pkg

const _EXPERIMENT_DIR = dirname(@__DIR__) |> abspath
Pkg.activate(_EXPERIMENT_DIR)

include(joinpath(_EXPERIMENT_DIR, "stdio_flush.jl"))
va_setup_stdio_flushing!()
include(joinpath(_EXPERIMENT_DIR, "calibration_sweep_configs.jl"))

function _run_subprocess(script::AbstractString, env::Pair{String, String})
    cmd = addenv(
        `$(Base.julia_cmd()) --project=$(_EXPERIMENT_DIR) $script`,
        env,
    )
    @info "Running" cmd
    va_flush_stdio()
    run(cmd)
    va_flush_stdio()
    return nothing
end

function run_calibration_slice(config_relp::AbstractString)
    env = "VA_EXPERIMENT_CONFIG" => config_relp
    _run_subprocess(joinpath(_EXPERIMENT_DIR, "generate_observations_reference.jl"), env)
    _run_subprocess(joinpath(_EXPERIMENT_DIR, "run_calibration.jl"), env)
    return nothing
end

function main()
    fail_fast = any(==("--fail-fast"), ARGS)
    tid = get(ENV, "CALIB_SWEEP_TASK_ID", get(ENV, "SLURM_ARRAY_TASK_ID", ""))
    configs = va_calibration_sweep_configs()
    if !isempty(strip(tid))
        i = parse(Int, tid) + 1
        (1 <= i <= length(configs)) ||
            error("Task index $(i - 1) out of range; need 0:$(length(configs) - 1)")
        return run_calibration_slice(configs[i])
    end
    failed = String[]
    for c in configs
        @info "Calibration sweep cell" c
        try
            run_calibration_slice(c)
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

main()
