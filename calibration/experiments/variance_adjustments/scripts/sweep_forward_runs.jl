# Forward grid: registered cases × (optional vertical ladder) × quadrature_order × varfix.
#
#   julia --project=. scripts/sweep_forward_runs.jl [--resolution-ladder | --baseline-only] [options]
#
# **Config:** sweep options live on `ForwardSweepConfig` (`forward_sweep_grid.jl`). Environment variables are read
# **only** in `va_forward_sweep_merge_env!` (called here before argv); the sweep driver uses **`cfg` only**.
#
# **Parallelism (no Slurm required):** `--parallel=threads` uses `Threads.@threads` (start Julia with
# `julia -t N`). `--parallel=distributed` uses `Distributed.pmap` after `addprocs`. Set
# `distributed_worker_threads` (CLI `--distributed-worker-threads`) so workers use `-t 1` while the main process can
# use `julia -t N`. Default sweep mode is sequential.
#
# **Default:** merged EKI member TOML (`forward_eki/` output). See [`forward_sweep_cases.yml`](../forward_sweep_cases.yml)
# for `eki_varfix_off_config` / `eki_varfix_on_config`. Use **`--baseline-scm-forward`** for exploratory runs
# with registry `scm_toml` only (`forward_only/`).
#
# Flags:
#   --eki-calibrated-forward   merged member TOML (default)
#   --baseline-scm-forward     registry SCM TOML only
#   --eki-iteration=N          default: latest on disk
#   --eki-member=M             default: best member (min Mahalanobis to obs); or --eki-member=best|auto
#   --parallel=MODE            sequential | threads | distributed
#   --distributed-workers=N
#   --distributed-worker-threads=N   per-worker `-t` for distributed mode (default 1)
#   (… plus resolution-ladder, registry, skip-done, fail-fast, task-id, ladder-*, print-task-count, --help)
#
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "sweep_forward_core.jl"))

function _sweep_print_help()
    println("""
Usage: julia --project=. scripts/sweep_forward_runs.jl [flags]

  --eki-calibrated-forward   Use merged EKI TOML; output under forward_eki/ (default)
  --baseline-scm-forward     Registry scm_toml only; output under forward_only/
  --eki-iteration=N          Latest iteration if omitted
  --eki-member=M|best|auto   Default: best (min Mahalanobis to obs); or fixed member index
  --parallel=MODE            sequential | threads | distributed
  --distributed-workers=N
  --distributed-worker-threads=N  per-worker Julia threads for distributed addprocs (default 1)
  --resolution-ladder        Multiple vertical tiers (default)
  --baseline-only            Single YAML tier only
  --registry=PATH            Case registry YAML
  --skip-done                Skip if output_active exists
  --fail-fast                Stop entire sweep on first failed run (default: continue)
  --print-task-count         Print N tasks and exit
  --task-id=N                0-based; else SWEEP_TASK_ID / SLURM_ARRAY_TASK_ID
  --ladder-n-tiers=N         Default 4
  --ladder-coarsen-ratio=R   Default 2
  --ladder-z-elem-min=N      Default 4
  --ladder-min-dz-factor=F   Default 2
""")
    va_flush_stdio()
    return nothing
end

function parse_forward_sweep_cli(argv::Vector{String})::ForwardSweepConfig
    cfg = ForwardSweepConfig()
    va_forward_sweep_merge_env!(cfg)
    for a in argv
        if a == "--help" || a == "-h"
            _sweep_print_help()
            exit(0)
        elseif a == "--eki-calibrated-forward"
            cfg.forward_parameters = VA_FORWARD_PARAM_EKI_CALIBRATED
        elseif a == "--baseline-scm-forward"
            cfg.forward_parameters = VA_FORWARD_PARAM_BASELINE_SCM
        elseif startswith(a, "--eki-iteration=")
            cfg.eki_iteration = parse(Int, split(a, '=', limit = 2)[2])
        elseif startswith(a, "--eki-member=")
            s = String(split(a, '=', limit = 2)[2])
            if s == "best" || s == "auto"
                cfg.eki_member = nothing
            else
                cfg.eki_member = parse(Int, s)
            end
        elseif startswith(a, "--parallel=")
            cfg.parallel = va_parse_forward_sweep_parallel_mode(split(a, '=', limit = 2)[2])
        elseif startswith(a, "--distributed-workers=")
            cfg.distributed_workers = parse(Int, split(a, '=', limit = 2)[2])
        elseif startswith(a, "--distributed-worker-threads=")
            cfg.distributed_worker_threads = parse(Int, split(a, '=', limit = 2)[2])
        elseif a == "--resolution-ladder"
            cfg.resolution_ladder = true
        elseif a == "--baseline-only" || a == "--no-resolution-ladder"
            cfg.resolution_ladder = false
        elseif startswith(a, "--registry=")
            cfg.registry_path = String(split(a, '=', limit = 2)[2])
        elseif a == "--skip-done"
            cfg.skip_done = true
        elseif a == "--fail-fast"
            cfg.fail_fast = true
        elseif a == "--print-task-count"
            cfg.print_task_count = true
        elseif startswith(a, "--task-id=")
            cfg.task_id = parse(Int, split(a, '=', limit = 2)[2])
        elseif startswith(a, "--ladder-n-tiers=")
            p = cfg.ladder
            cfg.ladder = VALadderParams(
                parse(Int, split(a, '=', limit = 2)[2]),
                p.coarsen_ratio,
                p.z_elem_min,
                p.min_dz_factor,
            )
        elseif startswith(a, "--ladder-coarsen-ratio=")
            p = cfg.ladder
            cfg.ladder = VALadderParams(
                p.n_tiers,
                parse(Float64, split(a, '=', limit = 2)[2]),
                p.z_elem_min,
                p.min_dz_factor,
            )
        elseif startswith(a, "--ladder-z-elem-min=")
            p = cfg.ladder
            cfg.ladder = VALadderParams(
                p.n_tiers,
                p.coarsen_ratio,
                parse(Int, split(a, '=', limit = 2)[2]),
                p.min_dz_factor,
            )
        elseif startswith(a, "--ladder-min-dz-factor=")
            p = cfg.ladder
            cfg.ladder = VALadderParams(
                p.n_tiers,
                p.coarsen_ratio,
                p.z_elem_min,
                parse(Float64, split(a, '=', limit = 2)[2]),
            )
        else
            error("Unknown argument: $(repr(a)). Try --help.")
        end
    end
    return cfg
end

function main()
    return run_forward_sweep!(parse_forward_sweep_cli(collect(String, ARGS)); merge_env = false)
end

if !isempty(Base.PROGRAM_FILE) && isfile(Base.PROGRAM_FILE) &&
   abspath(Base.PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
