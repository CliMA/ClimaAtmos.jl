# Forward-only grid: registered cases × (optional vertical ladder) × quadrature_order × varfix.
#
#   julia --project=. scripts/sweep_forward_runs.jl [--resolution-ladder | --baseline-only] [options]
#
# Cases: [`forward_sweep_cases.yml`](../forward_sweep_cases.yml), overridden by `--registry=REL.yml`.
#
# Flags (no experiment config via environment — use flags or kwargs from a caller):
#   --resolution-ladder     multiple vertical tiers (default)
#   --baseline-only         single tier from each case YAML (`z_elem`, `dt`, `dz_bottom`)
#   --registry=PATH         registry YAML (relative to experiment dir unless absolute)
#   --skip-done             skip run if `output_active` already exists
#   --fail-fast             abort the whole sweep on the first failed run (default: log and continue)
#   --print-task-count      print task count and exit (use with `sbatch --array=0-$((N-1))`; see `submit_forward_sweep_*.sh`)
#   --task-id=N             run only task N (0-based); if omitted, uses `SLURM_ARRAY_TASK_ID` when set
#   --ladder-n-tiers=N      default 4
#   --ladder-coarsen-ratio=R   default 2
#   --ladder-z-elem-min=N   default 4
#   --ladder-min-dz-factor=F   default 2
#   --help
#
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

const EXPERIMENT_DIR = dirname(@__DIR__) |> abspath
include(joinpath(EXPERIMENT_DIR, "stdio_flush.jl"))
va_setup_stdio_flushing!()
include(joinpath(EXPERIMENT_DIR, "experiment_common.jl"))
include(joinpath(@__DIR__, "resolution_ladder.jl"))

import ClimaAtmos as CA
import YAML

Base.@kwdef mutable struct ForwardSweepConfig
    resolution_ladder::Bool = true
    registry_path::Union{Nothing, String} = nothing
    skip_done::Bool = false
    """Abort entire sequential sweep on first error. Slurm single-task mode always propagates failure."""
    fail_fast::Bool = false
    print_task_count::Bool = false
    task_id::Union{Nothing, Int} = nothing
    ladder::VALadderParams = VALadderParams()
end

function va_forward_sweep_registry_path(cfg::ForwardSweepConfig)
    p = cfg.registry_path
    path = if p === nothing || isempty(strip(p))
        joinpath(EXPERIMENT_DIR, "forward_sweep_cases.yml")
    else
        isabspath(p) ? p : joinpath(EXPERIMENT_DIR, p)
    end
    return path
end

function va_load_forward_sweep_case_rows(cfg::ForwardSweepConfig)
    path = va_forward_sweep_registry_path(cfg)
    isfile(path) || error("Forward sweep registry missing: $path")
    data = YAML.load_file(path)
    cases = get(data, "cases", nothing)
    cases isa AbstractVector || error("Registry $path must contain a `cases:` list")
    rows = []
    for c in cases
        yml = string(c["model_config_path"])
        scm = string(c["scm_toml"])
        d = YAML.load_file(joinpath(EXPERIMENT_DIR, yml))
        slug = va_forward_sweep_case_slug(d)
        push!(rows, (; yaml_rel = yml, scm_toml = scm, slug, case_dict = d))
    end
    return rows
end

"""Orders for the forward grid; must match `gauss_hermite` in ClimaAtmos (`N` ≤ 5)."""
function sweep_quadrature_orders()
    return 1:5
end

function resolution_tiers(case_dict::AbstractDict, cfg::ForwardSweepConfig)
    if cfg.resolution_ladder
        return va_resolution_tiers_from_case_dict(case_dict, cfg.ladder)
    end
    z = Int(case_dict["z_elem"])
    dt = string(case_dict["dt"])
    return VAResolutionTier[VAResolutionTier(z, dt, nothing)]
end

function flatten_sweep_tasks(cfg::ForwardSweepConfig)
    rows = va_load_forward_sweep_case_rows(cfg)
    n_list = sweep_quadrature_orders()
    tasks = Tuple{String, String, String, Int, Bool, VAResolutionTier, Bool, Float64}[]
    for row in rows
        tiers = resolution_tiers(row.case_dict, cfg)
        z_stretch = _yaml_bool_stretch(row.case_dict)
        yaml_dz = _yaml_dz_bottom(row.case_dict)
        for tier in tiers, n in n_list, vf in (false, true)
            res_seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
            tag = vf ? "varfix_on" : "varfix_off"
            sub = joinpath(row.slug, res_seg, "N_$(n)", tag, "forward_only")
            push!(tasks, (row.yaml_rel, row.scm_toml, row.slug, n, vf, tier, z_stretch, yaml_dz))
        end
    end
    return tasks
end

function _yaml_bool_stretch(case_dict)
    return Bool(get(case_dict, "z_stretch", true))
end

function _yaml_dz_bottom(case_dict)
    v = get(case_dict, "dz_bottom", nothing)
    v === nothing && return 500.0
    return Float64(v)
end

function forward_sweep_task_count(cfg::ForwardSweepConfig)
    return length(flatten_sweep_tasks(cfg))
end

function run_one(;
    case_yaml::AbstractString,
    scm_toml::AbstractString,
    cas_slug::AbstractString,
    n_quad::Int,
    varfix::Bool,
    tier::VAResolutionTier,
    out_subdir::AbstractString,
    skip_done::Bool,
)
    cfg = YAML.load_file(joinpath(EXPERIMENT_DIR, case_yaml))
    cfg["z_elem"] = tier.z_elem
    cfg["dt"] = tier.dt_str
    if tier.dz_bottom_written !== nothing
        cfg["dz_bottom"] = tier.dz_bottom_written
    end
    nip = get(cfg, "netcdf_interpolation_num_points", nothing)
    if nip isa AbstractVector && length(nip) >= 3
        cfg["netcdf_interpolation_num_points"] = Any[nip[1], nip[2], tier.z_elem]
    end
    cfg["output_dir"] = joinpath(EXPERIMENT_DIR, "simulation_output", out_subdir)
    mkpath(cfg["output_dir"])
    active = joinpath(cfg["output_dir"], "output_active")
    if skip_done && isdir(active)
        @info "Skipping (existing output_active)" active
        va_flush_stdio()
        return nothing
    end
    cfg["quadrature_order"] = n_quad
    cfg["sgs_quadrature_subcell_geometric_variance"] = varfix
    cfg["toml"] = [va_scm_toml_path(EXPERIMENT_DIR, scm_toml)]
    cfg["output_default_diagnostics"] = get(cfg, "output_default_diagnostics", false)
    z_stretch = _yaml_bool_stretch(cfg)
    yaml_dz = _yaml_dz_bottom(cfg)
    res_slug = va_tier_path_segment(tier, z_stretch, yaml_dz)
    job_id = string(
        "va_sweep_",
        cas_slug,
        "_",
        res_slug,
        "_N",
        n_quad,
        "_",
        varfix ? "vf1" : "vf0",
    )
    atmos_config = CA.AtmosConfig(
        cfg;
        comms_ctx = va_comms_ctx(),
        job_id,
    )
    sim = CA.get_simulation(atmos_config)
    CA.solve_atmos!(sim)
    va_flush_stdio()
    @info "Finished run" out = cfg["output_dir"] job_id
    return sim
end

function run_task_index(task_id::Int, cfg::ForwardSweepConfig)
    tasks = flatten_sweep_tasks(cfg)
    task_id < 0 && error("task_id must be >= 0")
    task_id >= length(tasks) &&
        error("task_id $task_id out of range (max $(length(tasks) - 1))")
    yml, scm, slug, n, vf, tier, z_stretch, yaml_dz = tasks[task_id + 1]
    res_seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
    tag = vf ? "varfix_on" : "varfix_off"
    sub = joinpath(slug, res_seg, "N_$(n)", tag, "forward_only")
    @info "Sweep task" task_id slug n vf tier z_stretch
    return run_one(;
        case_yaml = yml,
        scm_toml = scm,
        cas_slug = slug,
        n_quad = n,
        varfix = vf,
        tier,
        out_subdir = sub,
        skip_done = cfg.skip_done,
    )
end

function _sweep_print_help()
    println("""
Usage: julia --project=. scripts/sweep_forward_runs.jl [flags]

  --resolution-ladder      Multiple vertical tiers (default)
  --baseline-only          Single YAML tier only
  --registry=PATH          Case registry YAML
  --skip-done              Skip if output_active exists
  --fail-fast              Stop entire sweep on first failed run (default: continue)
  --print-task-count       Print N tasks and exit (for sbatch --array=0-$((N-1)))
  --task-id=N              Run task N (0-based); else SWEEP_TASK_ID or SLURM_ARRAY_TASK_ID env
  --ladder-n-tiers=N       Default 4
  --ladder-coarsen-ratio=R Default 2
  --ladder-z-elem-min=N    Default 4
  --ladder-min-dz-factor=F Default 2
""")
    va_flush_stdio()
    return nothing
end

function parse_forward_sweep_cli(argv::Vector{String})::ForwardSweepConfig
    cfg = ForwardSweepConfig()
    for a in argv
        if a == "--help" || a == "-h"
            _sweep_print_help()
            exit(0)
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

"""Run the forward sweep; used by CLI and by `run_full_study!` via subprocess flags."""
function run_forward_sweep!(cfg::ForwardSweepConfig = ForwardSweepConfig())
    if cfg.print_task_count
        n = forward_sweep_task_count(cfg)
        println(n)
        va_flush_stdio()
        return nothing
    end
    tid = cfg.task_id
    if tid === nothing
        s = strip(get(ENV, "SWEEP_TASK_ID", get(ENV, "SLURM_ARRAY_TASK_ID", "")))
        tid = isempty(s) ? nothing : parse(Int, s)
    end
    if tid !== nothing
        return run_task_index(tid, cfg)
    end
    tasks = flatten_sweep_tasks(cfg)
    failed_labels = String[]
    for (yml, scm, slug, n, vf, tier, z_stretch, yaml_dz) in tasks
        res_seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
        tag = vf ? "varfix_on" : "varfix_off"
        sub = joinpath(slug, res_seg, "N_$(n)", tag, "forward_only")
        @info "Running" slug n vf tier
        try
            run_one(;
                case_yaml = yml,
                scm_toml = scm,
                cas_slug = slug,
                n_quad = n,
                varfix = vf,
                tier,
                out_subdir = sub,
                skip_done = cfg.skip_done,
            )
        catch err
            lab = "$(slug) $(res_seg) N=$(n) $(tag)"
            push!(failed_labels, lab)
            @error "Forward run failed" label = lab slug n vf tier exception = (err, catch_backtrace())
            va_flush_stdio()
            cfg.fail_fast && rethrow()
        end
    end
    if !isempty(failed_labels)
        @warn "Forward sweep finished with failed runs (continued)" n_failed = length(failed_labels) failed_labels
    end
    va_flush_stdio()
    return nothing
end

function main()
    return run_forward_sweep!(parse_forward_sweep_cli(collect(String, ARGS)))
end

main()
