# Naive workflow: after varfix-**off** EKI, one ClimaAtmos forward with varfix **on** and the same
# ensemble member parameters (merged TOML from that EKI).
#
#   julia --project=. scripts/run_naive_varfix_on_forwards.jl [flags]
#
# Flags:
#   --fail-fast              Stop entire sweep on first failure (default: log and continue)
#   --skip-done              Skip if output_active exists
#   --eki-iteration=N        EKI iteration (default: latest on disk)
#   --eki-member=N|best|auto Default: best (min Mahalanobis to obs); or fixed index
#   --print-task-count       Print number of naive slices and exit
#   --task-id=N              Run slice N only (0-based); else NAIVE_SWEEP_TASK_ID or SLURM_ARRAY_TASK_ID
#   --help
#
using Pkg: Pkg

const _EXPERIMENT_DIR = dirname(@__DIR__) |> abspath
Pkg.activate(_EXPERIMENT_DIR)

include(joinpath(_EXPERIMENT_DIR, "stdio_flush.jl"))
va_setup_stdio_flushing!()
if !isdefined(Main, :va_experiment_config_path)
    include(joinpath(_EXPERIMENT_DIR, "experiment_common.jl"))
end
include(joinpath(_EXPERIMENT_DIR, "calibration_sweep_configs.jl"))
if !isdefined(Main, :VA_COMPOSITE_SHORT_NAME_CLW_PLUS_CLI)
    include(joinpath(_EXPERIMENT_DIR, "observation_map.jl"))
end

using ClimaCalibrate: ClimaCalibrate as CAL
using ClimaAtmos: ClimaAtmos as CA
using YAML: YAML

Base.@kwdef mutable struct NaiveForwardConfig
    fail_fast::Bool = false
    skip_done::Bool = false
    eki_iteration::Union{Nothing, Int} = nothing
    eki_member::Union{Nothing, Int} = nothing
    print_task_count::Bool = false
    task_id::Union{Nothing, Int} = nothing
end

function _naive_task_configs()
    return collect(String, va_naive_varfix_off_source_configs())
end

function _naive_iteration_member(
    experiment_dir::AbstractString,
    config_relp::AbstractString,
    cfg::NaiveForwardConfig,
)
    iteration = if cfg.eki_iteration === nothing
        va_latest_eki_iteration_number(experiment_dir, config_relp)
    else
        cfg.eki_iteration
    end
    member = va_resolve_eki_member_index(experiment_dir, config_relp, iteration, cfg.eki_member)
    return iteration, member
end

function run_naive_one(
    experiment_dir::AbstractString,
    config_relp::AbstractString,
    cfg::NaiveForwardConfig = NaiveForwardConfig(),
)
    expc = YAML.load_file(joinpath(experiment_dir, config_relp))
    if Bool(get(expc, "sgs_quadrature_subcell_geometric_variance", false))
        @warn "Skipping naive forward: source YAML has varfix on" config_relp
        return nothing
    end
    out_rel = expc["output_dir"]
    eki_root = isabspath(out_rel) ? String(out_rel) : joinpath(experiment_dir, out_rel)
    isdir(eki_root) || error("Missing EKI directory $eki_root — run calibration for $config_relp first")

    iteration, member = _naive_iteration_member(experiment_dir, config_relp, cfg)
    member_path = CAL.path_to_ensemble_member(eki_root, iteration, member)
    merged = joinpath(member_path, VA_COMBINED_MEMBER_ATMOS_PARAMETERS_BASENAME)
    if !isfile(merged)
        p_small = joinpath(member_path, "parameters.toml")
        isfile(p_small) || error("No parameters.toml under $member_path")
        scm_baseline = va_merge_scm_baseline_dict(experiment_dir, expc["scm_toml"])
        mkpath(member_path)
        va_write_combined_member_atmos_parameters_toml(scm_baseline, p_small, merged)
    end

    atmos_cfg = va_load_merged_case_yaml_dict(experiment_dir, expc["model_config_path"])
    cas = string(expc["case_name"])
    n = Int(expc["quadrature_order"])
    out_sub = joinpath(cas, "N_$(n)", "varfix_on", "naive_from_varfix_off", "forward_only")
    atmos_cfg["output_dir"] = joinpath(experiment_dir, "simulation_output", out_sub)
    mkpath(atmos_cfg["output_dir"])
    active = joinpath(atmos_cfg["output_dir"], "output_active")
    if cfg.skip_done && isdir(active)
        @info "Skipping naive forward (existing output_active)" active
        va_flush_stdio()
        return nothing
    end

    nip = get(atmos_cfg, "netcdf_interpolation_num_points", nothing)
    if nip isa AbstractVector && length(nip) >= 3
        z_e = Int(atmos_cfg["z_elem"])
        atmos_cfg["netcdf_interpolation_num_points"] = Any[nip[1], nip[2], z_e]
    end

    atmos_cfg["quadrature_order"] = n
    atmos_cfg["sgs_quadrature_subcell_geometric_variance"] = true
    atmos_cfg["toml"] = [merged]
    atmos_cfg["output_default_diagnostics"] = get(atmos_cfg, "output_default_diagnostics", false)

    job_id = string(
        "va_naive_",
        cas,
        "_N",
        n,
        "_vf1_i",
        lpad(iteration, 3, '0'),
        "_m",
        lpad(member, 3, '0'),
    )
    atmos_config = CA.AtmosConfig(
        atmos_cfg;
        comms_ctx = va_comms_ctx(),
        job_id,
    )
    sim = CA.get_simulation(atmos_config)
    CA.solve_atmos!(sim)
    va_flush_stdio()
    @info "Naive varfix-on forward finished" output_dir = atmos_cfg["output_dir"] job_id
    return sim
end

function _naive_print_help()
    println("""
Usage: julia --project=. scripts/run_naive_varfix_on_forwards.jl [--fail-fast] [--skip-done] [--eki-iteration=N] [--eki-member=M] [--print-task-count] [--task-id=N]
""")
    va_flush_stdio()
    return nothing
end

function parse_naive_forward_cli(argv::Vector{String})::NaiveForwardConfig
    cfg = NaiveForwardConfig()
    for a in argv
        if a == "--help" || a == "-h"
            _naive_print_help()
            exit(0)
        elseif a == "--fail-fast"
            cfg.fail_fast = true
        elseif a == "--skip-done"
            cfg.skip_done = true
        elseif startswith(a, "--eki-iteration=")
            cfg.eki_iteration = parse(Int, split(a, '=', limit = 2)[2])
        elseif startswith(a, "--eki-member=")
            s = String(split(a, '=', limit = 2)[2])
            if s == "best" || s == "auto"
                cfg.eki_member = nothing
            else
                cfg.eki_member = parse(Int, s)
            end
        elseif a == "--print-task-count"
            cfg.print_task_count = true
        elseif startswith(a, "--task-id=")
            cfg.task_id = parse(Int, split(a, '=', limit = 2)[2])
        else
            error("Unknown argument: $(repr(a)). Try --help.")
        end
    end
    return cfg
end

function run_naive_forwards!(cfg::NaiveForwardConfig = NaiveForwardConfig())
    if cfg.print_task_count
        println(length(_naive_task_configs()))
        va_flush_stdio()
        return nothing
    end
    tid = cfg.task_id
    if tid === nothing
        s = strip(get(ENV, "NAIVE_SWEEP_TASK_ID", get(ENV, "SLURM_ARRAY_TASK_ID", "")))
        tid = isempty(s) ? nothing : parse(Int, s)
    end
    configs = _naive_task_configs()
    if tid !== nothing
        i = tid + 1
        (1 <= i <= length(configs)) ||
            error("task_id $tid out of range; need 0:$(length(configs) - 1)")
        return run_naive_one(_EXPERIMENT_DIR, configs[i], cfg)
    end
    failed = String[]
    for c in configs
        @info "Naive varfix-on forward" source_config = c
        try
            run_naive_one(_EXPERIMENT_DIR, c, cfg)
        catch err
            push!(failed, c)
            @error "Naive forward failed" source_config = c exception = (err, catch_backtrace())
            va_flush_stdio()
            cfg.fail_fast && rethrow()
        end
    end
    if !isempty(failed)
        @warn "Naive sweep finished with failures (continued)" n_failed = length(failed) failed
    end
    va_flush_stdio()
    return nothing
end

function main()
    return run_naive_forwards!(parse_naive_forward_cli(collect(String, ARGS)))
end

if !isempty(Base.PROGRAM_FILE) && isfile(Base.PROGRAM_FILE) &&
   abspath(Base.PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
