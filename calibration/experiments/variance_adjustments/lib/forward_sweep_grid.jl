# Shared forward-sweep **task grid** (registry × resolution tiers × N_quad × varfix). No ClimaAtmos runs.
# Included after `experiment_common.jl` and `scripts/resolution_ladder.jl` are loaded.
import YAML

"""Default forward-sweep case registry (relative to the experiment directory)."""
const VA_FORWARD_SWEEP_REGISTRY_DEFAULT_RELPATH = joinpath("registries", "forward_sweep_cases.yml")

"""Parameter source for resolution-ladder forwards: registry SCM only vs merged EKI member TOML."""
const VA_FORWARD_PARAM_BASELINE_SCM = :baseline_scm
const VA_FORWARD_PARAM_EKI_CALIBRATED = :eki_calibrated

Base.@kwdef mutable struct ForwardSweepConfig
    resolution_ladder::Bool = true
    registry_path::Union{Nothing, String} = nothing
    skip_done::Bool = false
    """Abort entire sequential sweep on first error. Slurm single-task mode always propagates failure."""
    fail_fast::Bool = false
    print_task_count::Bool = false
    task_id::Union{Nothing, Int} = nothing
    ladder::VALadderParams = VALadderParams()
    """
    `:eki_calibrated` (default): each run uses merged `_atmos_merged_parameters.toml` from the EKI slice
    named in the registry (`eki_varfix_off_config` / optional `eki_varfix_on_config`). Output under
    `forward_eki/`. `:baseline_scm`: registry `scm_toml` only; output under `forward_only/` (exploratory).
    """
    forward_parameters::Symbol = VA_FORWARD_PARAM_EKI_CALIBRATED
    """If `nothing`, use latest completed EKI iteration on disk."""
    eki_iteration::Union{Nothing, Int} = nothing
    """
    If `nothing`, select the ensemble member that **minimizes Mahalanobis distance** to **`observations.jld2`**
    (same metric as calibration); see `va_eki_best_member_by_obs_loss` in `observation_map.jl`.
    Set an explicit `Int` to force a member index.
    """
    eki_member::Union{Nothing, Int} = nothing
    """
    Sweep execution: `:sequential` (default), `:threads` (`Threads.@threads`; use `julia -t N`), or
    `:distributed` (`Distributed.pmap` after `addprocs`; set `distributed_workers`).
    Populated from env only via [`va_forward_sweep_merge_env!`](@ref) (CLI / `run_full_study` flags override).
    """
    parallel::Symbol = :sequential
    """Worker count when `parallel === :distributed` and `addprocs` is used."""
    distributed_workers::Int = min(32, max(1, Sys.CPU_THREADS))
    """`Distributed.addprocs` worker Julia thread count (`-t` per worker); typically `1` when the driver uses `julia -t N`."""
    distributed_worker_threads::Int = 1
end

"""
    va_parse_forward_sweep_parallel_mode(s::AbstractString) -> Symbol

Parse `sequential` / `threads` / `distributed` (and common aliases). Used by CLI and `va_forward_sweep_merge_env!`.
"""
function va_parse_forward_sweep_parallel_mode(s::AbstractString)::Symbol
    s = lowercase(String(strip(s)))
    if s == "sequential" || s == "serial"
        return :sequential
    elseif s == "threads" || s == "threaded"
        return :threads
    elseif s == "distributed" || s == "dist"
        return :distributed
    else
        error("Unknown forward sweep parallel mode $(repr(s)); use sequential, threads, or distributed.")
    end
end

"""
    va_forward_sweep_merge_env!(cfg::ForwardSweepConfig)

**Single boundary** for `VA_FORWARD_SWEEP_*` and `SWEEP_TASK_ID` / `SLURM_ARRAY_TASK_ID`: populate `cfg` from the
environment. Call from CLI entry points **before** argv overrides; the forward sweep driver (`run_forward_sweep!` in
`scripts/sweep_forward_core.jl`) reads **`cfg` only**, not `ENV` directly.
"""
function va_forward_sweep_merge_env!(cfg::ForwardSweepConfig)
    if haskey(ENV, "VA_FORWARD_SWEEP_PARALLEL")
        cfg.parallel = va_parse_forward_sweep_parallel_mode(ENV["VA_FORWARD_SWEEP_PARALLEL"])
    end
    if haskey(ENV, "VA_FORWARD_SWEEP_DISTRIBUTED_WORKERS")
        cfg.distributed_workers = parse(Int, ENV["VA_FORWARD_SWEEP_DISTRIBUTED_WORKERS"])
    end
    if haskey(ENV, "VA_FORWARD_SWEEP_WORKER_THREADS")
        cfg.distributed_worker_threads = parse(Int, ENV["VA_FORWARD_SWEEP_WORKER_THREADS"])
    end
    s = strip(get(ENV, "SWEEP_TASK_ID", get(ENV, "SLURM_ARRAY_TASK_ID", "")))
    if !isempty(s)
        cfg.task_id = parse(Int, s)
    end
    return cfg
end

"""Convenience: `ForwardSweepConfig()` with `va_forward_sweep_merge_env!` applied (REPL / scripts)."""
function va_forward_sweep_config_from_env()
    cfg = ForwardSweepConfig()
    va_forward_sweep_merge_env!(cfg)
    return cfg
end

function va_forward_sweep_forward_subdir(cfg::ForwardSweepConfig)::String
    return cfg.forward_parameters == VA_FORWARD_PARAM_BASELINE_SCM ? "forward_only" : "forward_eki"
end

function va_forward_sweep_registry_path(experiment_dir::AbstractString, cfg::ForwardSweepConfig)
    p = cfg.registry_path
    path = if p === nothing || isempty(strip(p))
        joinpath(experiment_dir, VA_FORWARD_SWEEP_REGISTRY_DEFAULT_RELPATH)
    else
        isabspath(p) ? p : joinpath(experiment_dir, p)
    end
    return path
end

function va_load_forward_sweep_case_rows(experiment_dir::AbstractString, cfg::ForwardSweepConfig)
    path = va_forward_sweep_registry_path(experiment_dir, cfg)
    isfile(path) || error("Forward sweep registry missing: $path")
    data = YAML.load_file(path)
    cases = get(data, "cases", nothing)
    cases isa AbstractVector || error("Registry $path must contain a `cases:` list")
    rows = []
    for c in cases
        scm = c["scm_toml"]
        layers = va_model_config_path_layers(c["model_config_path"])
        d = va_load_merged_case_yaml_dict(experiment_dir, layers)
        slug = va_forward_sweep_case_slug(d)
        yml = layers[end]
        eki_off = get(c, "eki_varfix_off_config", nothing)
        eki_off =
            eki_off === nothing || (eki_off isa AbstractString && isempty(strip(string(eki_off)))) ? nothing :
            string(strip(string(eki_off)))
        eki_on = get(c, "eki_varfix_on_config", nothing)
        eki_on =
            eki_on === nothing || (eki_on isa AbstractString && isempty(strip(string(eki_on)))) ? nothing :
            string(strip(string(eki_on)))
        push!(
            rows,
            (;
                yaml_rel = yml,
                model_config_layers = layers,
                scm_toml = scm,
                slug,
                case_dict = d,
                eki_varfix_off_config = eki_off,
                eki_varfix_on_config = eki_on,
            ),
        )
    end
    return rows
end

"""Registry row for `(slug, model_config_path)` or `nothing` if not found."""
function va_forward_sweep_registry_row_for(
    experiment_dir::AbstractString,
    slug::AbstractString,
    yaml_rel::AbstractString,
    cfg::ForwardSweepConfig,
)
    for r in va_load_forward_sweep_case_rows(experiment_dir, cfg)
        if r.slug == slug && r.yaml_rel == yaml_rel
            return r
        end
    end
    return nothing
end

"""Quadrature orders for the forward grid; must match ClimaAtmos `gauss_hermite` (`N` ≤ 5)."""
function va_forward_sweep_quadrature_orders()
    return 1:5
end

function va_resolution_tiers_for_forward(case_dict::AbstractDict, cfg::ForwardSweepConfig)
    if cfg.resolution_ladder
        return va_resolution_tiers_from_case_dict(case_dict, cfg.ladder)
    end
    z = Int(case_dict["z_elem"])
    dt = string(case_dict["dt"])
    return VAResolutionTier[VAResolutionTier(z, dt, nothing)]
end

function va_forward_sweep_case_z_stretch(case_dict::AbstractDict)
    return Bool(get(case_dict, "z_stretch", true))
end

function va_forward_sweep_case_dz_bottom(case_dict::AbstractDict)
    v = get(case_dict, "dz_bottom", nothing)
    v === nothing && return 500.0
    return Float64(v)
end

"""
    va_flatten_forward_sweep_tasks(experiment_dir, cfg) -> Vector{Tuple}

Each row: `(model_config_layers, scm_toml, slug, n_quad, varfix_bool, tier, z_stretch, yaml_dz, eki_off, eki_on)`
where **`model_config_layers`** is `Vector{String}` (merge order); **`yaml_rel`** for grouping is `layers[end]`.
with `eki_off` / `eki_on` optional registry-relative experiment YAML names (or `nothing`).
Output layout: `simulation_output/<slug>/<res_seg>/N_<n>/(varfix_on|varfix_off)/<forward_subdir>/output_active/`
where `<forward_subdir>` is `forward_eki` or `forward_only` (see [`va_forward_sweep_forward_subdir`](@ref)).
"""
function va_flatten_forward_sweep_tasks(experiment_dir::AbstractString, cfg::ForwardSweepConfig)
    rows = va_load_forward_sweep_case_rows(experiment_dir, cfg)
    n_list = va_forward_sweep_quadrature_orders()
    tasks = Tuple{
        Vector{String},
        Any,
        String,
        Int,
        Bool,
        VAResolutionTier,
        Bool,
        Float64,
        Union{Nothing, String},
        Union{Nothing, String},
    }[]
    for row in rows
        tiers = va_resolution_tiers_for_forward(row.case_dict, cfg)
        z_stretch = va_forward_sweep_case_z_stretch(row.case_dict)
        yaml_dz = va_forward_sweep_case_dz_bottom(row.case_dict)
        for tier in tiers, n in n_list, vf in (false, true)
            push!(
                tasks,
                (
                    row.model_config_layers,
                    row.scm_toml,
                    row.slug,
                    n,
                    vf,
                    tier,
                    z_stretch,
                    yaml_dz,
                    row.eki_varfix_off_config,
                    row.eki_varfix_on_config,
                ),
            )
        end
    end
    return tasks
end

function va_forward_sweep_task_count(experiment_dir::AbstractString, cfg::ForwardSweepConfig)
    return length(va_flatten_forward_sweep_tasks(experiment_dir, cfg))
end

"""`output_active` path for one forward-sweep cell (may not exist yet). Requires `cfg` for `forward_eki` vs `forward_only`."""
function va_forward_sweep_output_active_path(
    experiment_dir::AbstractString,
    slug::AbstractString,
    res_seg::AbstractString,
    n::Int,
    varfix::Bool,
    cfg::ForwardSweepConfig,
)
    tag = varfix ? "varfix_on" : "varfix_off"
    leaf = va_forward_sweep_forward_subdir(cfg)
    return joinpath(
        experiment_dir,
        "simulation_output",
        slug,
        res_seg,
        "N_$(n)",
        tag,
        leaf,
        "output_active",
    )
end

"""
    va_forward_sweep_finest_completed_active(experiment_dir, slug, yaml_rel, cfg; n_quad, varfix) -> Union{Nothing, Tuple{Int,String}}

Among resolution tiers for this registry case, return `(z_elem, output_active)` for the **finest completed**
tier (largest `z_elem` in the ladder that has `output_active`), for the given `N_quad` and varfix flag.
"""
function va_forward_sweep_finest_completed_active(
    experiment_dir::AbstractString,
    slug::AbstractString,
    yaml_rel::AbstractString,
    cfg::ForwardSweepConfig;
    n_quad::Int = 3,
    varfix::Bool = false,
)::Union{Nothing, Tuple{Int, String}}
    rows = va_load_forward_sweep_case_rows(experiment_dir, cfg)
    row = nothing
    for r in rows
        if r.slug == slug && r.yaml_rel == yaml_rel
            row = r
            break
        end
    end
    row === nothing && return nothing
    tiers = va_resolution_tiers_for_forward(row.case_dict, cfg)
    z_stretch = va_forward_sweep_case_z_stretch(row.case_dict)
    yaml_dz = va_forward_sweep_case_dz_bottom(row.case_dict)
    best_z = -1
    best_dz = Inf
    best_path = nothing
    for tier in tiers
        seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
        ap = va_forward_sweep_output_active_path(experiment_dir, slug, seg, n_quad, varfix, cfg)
        !isdir(ap) && continue
        tdz = va_effective_dz_bottom(tier, yaml_dz)
        if tier.z_elem > best_z
            best_z = tier.z_elem
            best_dz = tdz
            best_path = ap
        elseif tier.z_elem == best_z && tdz < best_dz
            best_dz = tdz
            best_path = ap
        end
    end
    best_path === nothing && return nothing
    return (best_z, best_path)
end

function _va_parse_z_dz_from_res_seg(res_seg::AbstractString, yaml_dz::Float64)
    m = match(r"^z(\d+)_", String(res_seg))
    m === nothing && return nothing
    z = parse(Int, m.captures[1])
    md = match(r"_dzb(\d+)", String(res_seg))
    dz = md === nothing ? yaml_dz : parse(Float64, md.captures[1])
    return (z, dz)
end

function _va_tier_strictly_finer_than_panel(
    tier_z::Int,
    tier_dz::Float64,
    panel_z::Int,
    panel_dz::Float64,
    z_stretch::Bool,
)::Bool
    tier_z > panel_z && return true
    tier_z < panel_z && return false
    z_stretch || return false
    return tier_dz < panel_dz
end

"""
    va_forward_sweep_reference_finer_than_panel(experiment_dir, slug, yaml_rel, cfg, panel_res_seg; n_quad, varfix) -> Union{Nothing, Tuple{Int,String,String}}

Returns **`(z_elem, output_active_path, res_segment)`** where **`res_segment`** is the ladder label for that
tier (e.g. `z60_dt10s_dzb20`) for human-readable plot legends.

Among **completed** forward outputs at `(n_quad, varfix)`, pick the **finest** tier that is **strictly finer**
than the panel identified by **`panel_res_seg`** (e.g. `z30_dt10s_dzb20`). Finer means higher `z_elem`, or
the same `z_elem` with **smaller** `dz_bottom` when `z_stretch` is true. Used for the reference overlay on
profile plots; fixes the case where only `z_elem` was compared and same-`z` stretched tiers never got a ref line.
"""
function va_forward_sweep_reference_finer_than_panel(
    experiment_dir::AbstractString,
    slug::AbstractString,
    yaml_rel::AbstractString,
    cfg::ForwardSweepConfig,
    panel_res_seg::AbstractString;
    n_quad::Int = 3,
    varfix::Bool = false,
)::Union{Nothing, Tuple{Int, String, String}}
    rows = va_load_forward_sweep_case_rows(experiment_dir, cfg)
    row = nothing
    for r in rows
        if r.slug == slug && r.yaml_rel == yaml_rel
            row = r
            break
        end
    end
    row === nothing && return nothing
    z_stretch = va_forward_sweep_case_z_stretch(row.case_dict)
    yaml_dz = va_forward_sweep_case_dz_bottom(row.case_dict)
    parsed = _va_parse_z_dz_from_res_seg(panel_res_seg, yaml_dz)
    parsed === nothing && return nothing
    panel_z, panel_dz = parsed
    tiers = va_resolution_tiers_for_forward(row.case_dict, cfg)
    best_z = -1
    best_dz = Inf
    best_path = nothing
    best_seg = ""
    for tier in tiers
        seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
        ap = va_forward_sweep_output_active_path(experiment_dir, slug, seg, n_quad, varfix, cfg)
        !isdir(ap) && continue
        tz = tier.z_elem
        tdz = va_effective_dz_bottom(tier, yaml_dz)
        if !_va_tier_strictly_finer_than_panel(tz, tdz, panel_z, panel_dz, z_stretch)
            continue
        end
        if tz > best_z || (tz == best_z && tdz < best_dz)
            best_z = tz
            best_dz = tdz
            best_path = ap
            best_seg = seg
        end
    end
    best_path === nothing && return nothing
    return (best_z, best_path, best_seg)
end
