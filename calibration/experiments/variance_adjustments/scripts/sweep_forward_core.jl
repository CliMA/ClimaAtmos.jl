# Core forward-sweep logic (included from `sweep_forward_runs.jl` and `forward_sweep_worker_init.jl`).
# `EXPERIMENT_DIR` is the variance_adjustments experiment root (`scripts/..`).

const EXPERIMENT_DIR = joinpath(@__DIR__, "..") |> abspath

include(joinpath(EXPERIMENT_DIR, "lib", "stdio_flush.jl"))
va_setup_stdio_flushing!()
if !isdefined(Main, :va_experiment_config_path)
    include(joinpath(EXPERIMENT_DIR, "lib", "experiment_common.jl"))
end
if !isdefined(Main, :ForwardSweepConfig)
    include(joinpath(@__DIR__, "resolution_ladder.jl"))
    include(joinpath(EXPERIMENT_DIR, "lib", "forward_sweep_grid.jl"))
end
if !isdefined(Main, :VA_COMPOSITE_SHORT_NAME_CLW_PLUS_CLI)
    include(joinpath(EXPERIMENT_DIR, "lib", "observation_map.jl"))
end

import ClimaAtmos as CA
import ClimaCalibrate as CAL
import YAML
import Distributed

function _va_merged_eki_toml_path(
    experiment_dir::AbstractString,
    config_relp::AbstractString,
    iteration::Int,
    member::Int,
)::String
    expc = va_load_experiment_config(experiment_dir, config_relp)
    out = expc["output_dir"]
    root = isabspath(out) ? String(out) : joinpath(experiment_dir, out)
    member_dir = CAL.path_to_ensemble_member(root, iteration, member)
    p = joinpath(member_dir, VA_COMBINED_MEMBER_ATMOS_PARAMETERS_BASENAME)
    isfile(p) ||
        error(
            "Missing merged EKI TOML at $p — run calibration for $(repr(config_relp)) first (iter=$iteration, member=$member).",
        )
    return p
end

function _va_resolve_sweep_merged_toml(
    experiment_dir::AbstractString,
    sweep_cfg::ForwardSweepConfig,
    varfix::Bool,
    eki_varfix_off_config::Union{Nothing, String},
    eki_varfix_on_config::Union{Nothing, String},
)::Union{Nothing, String}
    sweep_cfg.forward_parameters == VA_FORWARD_PARAM_BASELINE_SCM && return nothing
    eki_varfix_off_config === nothing && error(
        "EKI-parameter forward sweep: registry case missing `eki_varfix_off_config` (see registries/forward_sweep_cases.yml).",
    )
    config_relp = varfix ? something(eki_varfix_on_config, eki_varfix_off_config) : eki_varfix_off_config
    iter = something(sweep_cfg.eki_iteration, va_latest_eki_iteration_number(experiment_dir, config_relp))
    mem = va_resolve_eki_member_index(experiment_dir, config_relp, iter, sweep_cfg.eki_member)
    return _va_merged_eki_toml_path(experiment_dir, config_relp, iter, mem)
end

function run_one(;
    case_yaml_layers::Vector{String},
    scm_toml::Any,
    cas_slug::AbstractString,
    n_quad::Int,
    varfix::Bool,
    tier::VAResolutionTier,
    out_subdir::AbstractString,
    skip_done::Bool,
    merged_member_toml::Union{Nothing, AbstractString} = nothing,
    varfix_on_distribution::Union{Nothing, AbstractString} = nothing,
)
    cfg = va_load_merged_case_yaml_dict(EXPERIMENT_DIR, case_yaml_layers)
    # Same as `model_interface.jl` / README: forward sweep does not use `experiment_config.yml`, so honor env overrides.
    gcm_force = strip(get(ENV, "VA_GCM_FORCING_FILE", ""))
    if !isempty(gcm_force)
        cfg["external_forcing_file"] = gcm_force
    end
    gle_force = strip(get(ENV, "VA_GOOGLELES_FORCING_FILE", ""))
    if !isempty(gle_force)
        cfg["external_forcing_file"] = gle_force
    end
    gle_grp = strip(get(ENV, "VA_GOOGLELES_CFSITE_GROUP", ""))
    if !isempty(gle_grp)
        cfg["cfsite_number"] = gle_grp
    end
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
    base_dist = string(get(cfg, "sgs_distribution", "lognormal"))
    vo = get(cfg, "sgs_distribution_varfix_on", nothing)
    if varfix
        if varfix_on_distribution !== nothing && !isempty(strip(string(varfix_on_distribution)))
            cfg["sgs_distribution"] = String(strip(string(varfix_on_distribution)))
        else
            cfg["sgs_distribution"] =
                vo !== nothing && !isempty(strip(string(vo))) ? String(strip(string(vo))) :
                va_base_to_vertical_profile_sgs_distribution(base_dist)
        end
    else
        cfg["sgs_distribution"] = base_dist
    end
    # Not a ClimaAtmos key — only for sweep driver (would warn as unused in `AtmosConfig`).
    pop!(cfg, "sgs_distribution_varfix_on", nothing)
    if merged_member_toml === nothing
        merged_path = joinpath(cfg["output_dir"], VA_MERGED_SCM_BASELINE_BASENAME)
        va_write_merged_scm_baseline_file!(EXPERIMENT_DIR, scm_toml, merged_path)
        cfg["toml"] = [merged_path]
    else
        cfg["toml"] = [merged_member_toml]
    end
    cfg["output_default_diagnostics"] = get(cfg, "output_default_diagnostics", false)
    z_stretch = va_forward_sweep_case_z_stretch(cfg)
    yaml_dz = va_forward_sweep_case_dz_bottom(cfg)
    res_slug = va_tier_path_segment(tier, z_stretch, yaml_dz)
    job_suffix = ""
    if varfix && varfix_on_distribution !== nothing && !isempty(strip(string(varfix_on_distribution)))
        job_suffix = string("_", va_sgs_dist_path_slug(String(strip(string(varfix_on_distribution)))))
    end
    job_id = string(
        "va_sweep_",
        cas_slug,
        "_",
        res_slug,
        "_N",
        n_quad,
        "_",
        varfix ? "vf1" : "vf0",
        job_suffix,
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

"""Run one sweep row (single SCM) using a precomputed `tasks[i]` tuple — avoids re-flattening the grid."""
function run_sweep_task_row(
    tasks,
    i::Int,
    cfg::ForwardSweepConfig;
    task_id_for_log::Union{Nothing, Int} = nothing,
)
    layers, scm, slug, n, vf, tier, z_stretch, yaml_dz, eki_off, eki_on, vfon = tasks[i]
    res_seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
    tag = va_forward_sweep_varfix_dir_segment(vf, vf ? vfon : nothing)
    leaf = va_forward_sweep_forward_subdir(cfg)
    sub = joinpath(slug, res_seg, "N_$(n)", tag, leaf)
    merged = _va_resolve_sweep_merged_toml(EXPERIMENT_DIR, cfg, vf, eki_off, eki_on)
    if task_id_for_log === nothing
        @info "Running" slug n vf tier forward_parameters = cfg.forward_parameters
    else
        @info "Sweep task" task_id = task_id_for_log slug n vf tier z_stretch forward_parameters =
            cfg.forward_parameters
    end
    return run_one(;
        case_yaml_layers = layers,
        scm_toml = scm,
        cas_slug = slug,
        n_quad = n,
        varfix = vf,
        tier,
        out_subdir = sub,
        skip_done = cfg.skip_done,
        merged_member_toml = merged,
        varfix_on_distribution = vfon,
    )
end

function run_task_index(task_id::Int, cfg::ForwardSweepConfig)
    tasks = va_flatten_forward_sweep_tasks(EXPERIMENT_DIR, cfg)
    task_id < 0 && error("task_id must be >= 0")
    task_id >= length(tasks) &&
        error("task_id $task_id out of range (max $(length(tasks) - 1))")
    return run_sweep_task_row(tasks, task_id + 1, cfg; task_id_for_log = task_id)
end

function _va_sweep_task_label(tasks, i::Int)::String
    layers, scm, slug, n, vf, tier, z_stretch, yaml_dz, eki_off, eki_on, vfon = tasks[i]
    res_seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
    tag = va_forward_sweep_varfix_dir_segment(vf, vf ? vfon : nothing)
    return "$(slug) $(res_seg) N=$(n) $(tag)"
end

function _va_forward_sweep_ensure_workers!(cfg::ForwardSweepConfig)
    nw = max(1, cfg.distributed_workers)
    # Use `nprocs() == 1` (only the driver). Do not use `nworkers() == 0`: stdlib `nworkers()` is implemented as
    # `nprocs() == 1 ? 1 : nprocs()-1` (see `Distributed/src/cluster.jl`), so with a lone process `nworkers()` is **1**,
    # the guard never fires, `addprocs` is skipped, and `pmap` stays on process 1.
    if Distributed.nprocs() == 1
        Distributed.addprocs(
            nw;
            exeflags = va_worker_julia_exeflags(EXPERIMENT_DIR, cfg.distributed_worker_threads),
        )
    end
    wip = joinpath(EXPERIMENT_DIR, "scripts", "forward_sweep_worker_init.jl")
    Distributed.@everywhere include($wip)
    @info "Distributed forward sweep: cluster ready" nworkers = Distributed.nworkers() nprocs = Distributed.nprocs() procs = Distributed.procs()
    return nothing
end

"""Remove `Distributed` workers so the next stage (e.g. EKI or a fresh sweep) starts from `nprocs() == 1`."""
function _va_forward_sweep_cleanup_workers!()
    Distributed.nprocs() <= 1 && return nothing
    ws = copy(Distributed.workers())
    isempty(ws) && return nothing
    Distributed.rmprocs(ws)
    return nothing
end

"""
Run the forward sweep; used by CLI and by `run_full_study!` in-process.

If `merge_env` is `true` (default), applies `va_forward_sweep_merge_env!` once (defined in `forward_sweep_grid.jl`) so
`ENV` is only read at this boundary. The CLI merges env **before** argv and calls with `merge_env=false` to avoid
double application.
"""
function run_forward_sweep!(cfg::ForwardSweepConfig = ForwardSweepConfig(); merge_env::Bool = true)
    merge_env && va_forward_sweep_merge_env!(cfg)
    if cfg.print_task_count
        n = va_forward_sweep_task_count(EXPERIMENT_DIR, cfg)
        println(n)
        va_flush_stdio()
        return nothing
    end
    tid = cfg.task_id
    if tid !== nothing
        return run_task_index(tid, cfg)
    end
    tasks = va_flatten_forward_sweep_tasks(EXPERIMENT_DIR, cfg)
    failed_labels = String[]

    if cfg.parallel === :sequential
        for i in eachindex(tasks)
            try
                run_sweep_task_row(tasks, i, cfg)
            catch err
                lab = _va_sweep_task_label(tasks, i)
                push!(failed_labels, lab)
                layers, scm, slug, n, vf, tier, z_stretch, yaml_dz, eki_off, eki_on, vfon = tasks[i]
                @error "Forward run failed" label = lab slug n vf tier vfon exception = (err, catch_backtrace())
                va_flush_stdio()
                cfg.fail_fast && rethrow()
            end
        end
    elseif cfg.parallel === :threads
        # Avoid `Threads.threadid()` in logs: tasks can migrate between threads; see Julia multithreading PSA
        # (https://discourse.julialang.org/t/psa-thread-local-state-is-no-longer-recommended-common-misconceptions-about-threadid-and-nthreads/101274).
        if Base.Threads.nthreads() < 2
            @warn "parallel=:threads but only $(Base.Threads.nthreads()) Julia thread(s); start with e.g. `julia -t 8` for real parallelism."
        end
        if cfg.fail_fast
            @warn "parallel=:threads with fail-fast may not stop other tasks immediately; use parallel=:sequential for strict fail-fast."
        end
        lk = ReentrantLock()
        Base.Threads.@threads for i in eachindex(tasks)
            try
                run_sweep_task_row(tasks, i, cfg)
            catch err
                lab = _va_sweep_task_label(tasks, i)
                lock(lk) do
                    push!(failed_labels, lab)
                end
                layers, scm, slug, n, vf, tier, z_stretch, yaml_dz, eki_off, eki_on, vfon = tasks[i]
                @error "Forward run failed" label = lab slug n vf tier vfon exception = (err, catch_backtrace())
                va_flush_stdio()
                cfg.fail_fast && rethrow()
            end
        end
    elseif cfg.parallel === :distributed
        if cfg.fail_fast
            @warn "parallel=:distributed with fail-fast may not stop other tasks immediately; use parallel=:sequential for strict fail-fast."
        end
        _va_forward_sweep_ensure_workers!(cfg)
        try
            results = Distributed.pmap(eachindex(tasks)) do i
                try
                    run_sweep_task_row(tasks, i, cfg)
                    return (false, i, nothing)
                catch err
                    return (true, i, err)
                end
            end
            for (failed, i, err) in results
                failed || continue
                lab = _va_sweep_task_label(tasks, i)
                push!(failed_labels, lab)
                @error "Forward run failed" label = lab exception = (err, catch_backtrace())
                va_flush_stdio()
                cfg.fail_fast && rethrow(err)
            end
        finally
            _va_forward_sweep_cleanup_workers!()
        end
    else
        error("Unknown parallel mode: $(repr(cfg.parallel)); use :sequential, :threads, or :distributed.")
    end

    if !isempty(failed_labels)
        @warn "Forward sweep finished with failed runs (continued)" n_failed = length(failed_labels) failed_labels
    end
    va_flush_stdio()
    return nothing
end
