# Full experiment driver: forward sweep → EKI sweep → naive forwards → figures.
#
# **CLI** (from this directory):
#   julia --project=. run_full_study.jl [flags]
#
# **REPL** (kwargs, no environment variables required):
#   using Pkg; Pkg.activate("."); include("run_full_study.jl")
#   run_full_study!()                              # default pipeline
#   run_full_study!(; skip_forward = true)         # only EKI + naive + figures
#   run_full_study!(; forward_baseline_only = true) # 20 forwards (N_quad 1–5), no resolution ladder
#
# CLI flags (same options as kwargs on `FullStudyOptions`):
#   --skip-instantiate
#   --skip-forward
#   --forward-baseline-only     # forward sweep: single YAML tier per case (no resolution ladder)
#   --skip-calib
#   --skip-naive
#   --skip-figures
#   --forward-skip-done         # passed to forward sweep (--skip-done)
#   --forward-fail-fast         # forward sweep: abort on first failed run (--fail-fast)
#   --calib-fail-fast           # EKI sweep: abort on first failed slice (--fail-fast)
#   --naive-fail-fast           # naive forwards: abort on first failure (--fail-fast)
#   --naive-skip-done           # passed to naive script (--skip-done)
#   --help
#
# Child scripts use **CLI flags** (not `VA_*` env) for their options. Optional env **only** for job
# arrays: `SWEEP_TASK_ID`, `SLURM_ARRAY_TASK_ID`, `NAIVE_SWEEP_TASK_ID` when not passing `--task-id=`.
#
import Pkg

const _VA_ROOT = dirname(@__FILE__) |> abspath
Pkg.activate(_VA_ROOT)
include(joinpath(_VA_ROOT, "stdio_flush.jl"))
va_setup_stdio_flushing!()
include(joinpath(_VA_ROOT, "calibration_sweep_configs.jl"))

Base.@kwdef mutable struct FullStudyOptions
    skip_instantiate::Bool = false
    skip_forward::Bool = false
    """If false, forward subprocess uses `--baseline-only` (one tier per case YAML)."""
    forward_resolution_ladder::Bool = true
    skip_calib::Bool = false
    skip_naive::Bool = false
    skip_figures::Bool = false
    forward_skip_done::Bool = false
    """Pass `--fail-fast` to the forward sweep subprocess (stop on first error)."""
    forward_fail_fast::Bool = false
    """Pass `--fail-fast` to `run_calibration_sweep.jl`."""
    calib_fail_fast::Bool = false
    """Pass `--fail-fast` to naive forwards script."""
    naive_fail_fast::Bool = false
    naive_skip_done::Bool = false
end

function forward_baseline_only(opts::FullStudyOptions)
    return !opts.forward_resolution_ladder
end

function _full_study_print_help()
    println("""
Usage: julia --project=. run_full_study.jl [flags]

  --skip-instantiate
  --skip-forward
  --forward-baseline-only
  --skip-calib
  --skip-naive
  --skip-figures
  --forward-skip-done
  --forward-fail-fast
  --calib-fail-fast
  --naive-fail-fast
  --naive-skip-done

REPL: include(\"run_full_study.jl\"); run_full_study!(; skip_forward = true)
""")
    va_flush_stdio()
    return nothing
end

function parse_full_study_cli(argv::Vector{String})::FullStudyOptions
    opts = FullStudyOptions()
    for a in argv
        if a == "--help" || a == "-h"
            _full_study_print_help()
            exit(0)
        elseif a == "--skip-instantiate"
            opts.skip_instantiate = true
        elseif a == "--skip-forward"
            opts.skip_forward = true
        elseif a == "--forward-baseline-only"
            opts.forward_resolution_ladder = false
        elseif a == "--skip-calib"
            opts.skip_calib = true
        elseif a == "--skip-naive"
            opts.skip_naive = true
        elseif a == "--skip-figures"
            opts.skip_figures = true
        elseif a == "--forward-skip-done"
            opts.forward_skip_done = true
        elseif a == "--forward-fail-fast"
            opts.forward_fail_fast = true
        elseif a == "--calib-fail-fast"
            opts.calib_fail_fast = true
        elseif a == "--naive-fail-fast"
            opts.naive_fail_fast = true
        elseif a == "--naive-skip-done"
            opts.naive_skip_done = true
        else
            error("Unknown argument: $(repr(a)). Try --help.")
        end
    end
    return opts
end

function _run_cmd_julia(args::Vector{String})
    exe = copy(Base.julia_cmd().exec)
    append!(exe, args)
    va_flush_stdio()
    r = run(Cmd(exe))
    va_flush_stdio()
    return r
end

function _run_forward_sweep_subprocess(opts::FullStudyOptions)
    script = joinpath(_VA_ROOT, "scripts", "sweep_forward_runs.jl")
    julia_args = String["--project=$(_VA_ROOT)", script]
    push!(julia_args, opts.forward_resolution_ladder ? "--resolution-ladder" : "--baseline-only")
    if opts.forward_skip_done
        push!(julia_args, "--skip-done")
    end
    if opts.forward_fail_fast
        push!(julia_args, "--fail-fast")
    end
    @info "Running forward sweep subprocess" julia_args use_resolution_ladder = opts.forward_resolution_ladder
    return _run_cmd_julia(julia_args)
end

function run_full_study!(opts::FullStudyOptions)
    if !opts.skip_instantiate
        Pkg.instantiate()
    end
    if !opts.skip_forward
        @info "Forward grid (registry × N_quad × varfix × resolution ladder)" forward_skip_done =
            opts.forward_skip_done forward_resolution_ladder = opts.forward_resolution_ladder
        _run_forward_sweep_subprocess(opts)
    else
        @info "Skipping forward grid (--skip-forward)"
    end

    if !opts.skip_calib
        @info "EKI calibration sweep" configs = va_calibration_sweep_configs()
        script = joinpath(_VA_ROOT, "scripts", "run_calibration_sweep.jl")
        calib_args = String["--project=$(_VA_ROOT)", script]
        if opts.calib_fail_fast
            push!(calib_args, "--fail-fast")
        end
        _run_cmd_julia(calib_args)
    else
        @info "Skipping calibration sweep (--skip-calib)"
    end

    if !opts.skip_naive
        if opts.skip_calib
            @warn "Calibration was skipped; naive forwards need existing varfix-off EKI output (or --skip-naive)"
        end
        naive_script = joinpath(_VA_ROOT, "scripts", "run_naive_varfix_on_forwards.jl")
        julia_args = String["--project=$(_VA_ROOT)", naive_script]
        if opts.naive_skip_done
            push!(julia_args, "--skip-done")
        end
        if opts.naive_fail_fast
            push!(julia_args, "--fail-fast")
        end
        @info "Naive varfix-on forwards"
        _run_cmd_julia(julia_args)
    else
        @info "Skipping naive forwards (--skip-naive)"
    end

    if !opts.skip_figures
        include(joinpath(_VA_ROOT, "analysis/plotting/run_post_analysis.jl"))
        for c in va_calibration_sweep_configs()
            @info "Post-analysis figures" experiment_config = c
            va_run_post_analysis!(; experiment_dir = _VA_ROOT, experiment_config = c)
        end
        if !opts.skip_naive
            for c in va_naive_varfix_off_source_configs()
                expc = va_load_experiment_config(_VA_ROOT, c)
                naive_active = va_naive_forward_output_active(_VA_ROOT, c)
                if !isdir(naive_active)
                    @warn "Skipping naive post-analysis (missing output_active)" experiment_config =
                        c naive_active
                    continue
                end
                ref = va_reference_output_active(_VA_ROOT, c)
                fig = va_naive_post_analysis_figure_dir(_VA_ROOT, expc)
                @info "Post-analysis figures (naive overlay)" experiment_config = c figure_root = fig
                va_run_post_analysis!(;
                    experiment_dir = _VA_ROOT,
                    experiment_config = c,
                    profile_paths = String[ref, naive_active],
                    figure_root = fig,
                )
            end
        end
    else
        @info "Skipping figures (--skip-figures)"
    end

    @info "Full study pipeline finished" root = _VA_ROOT
    va_flush_stdio()
    return nothing
end

function run_full_study!(; forward_baseline_only::Bool = false, kwargs...)
    o = FullStudyOptions(; kwargs...)
    if forward_baseline_only
        o.forward_resolution_ladder = false
    end
    return run_full_study!(o)
end

if abspath(Base.PROGRAM_FILE) == abspath(@__FILE__)
    run_full_study!(parse_full_study_cli(collect(String, ARGS)))
end
