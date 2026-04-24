# Shared forward-sweep **task grid** (registry × resolution tiers × N_quad × varfix). No ClimaAtmos runs.
# Included after `experiment_common.jl` and `scripts/resolution_ladder.jl` are loaded.
import YAML

"""Filesystem-safe slug for an `sgs_distribution` string (used in output paths and job ids)."""
function va_sgs_dist_path_slug(dist::AbstractString)::String
    s = replace(String(dist), r"[^A-Za-z0-9_.-]+" => "_")
    s = strip(s, '_')
    return isempty(s) ? "dist" : s
end

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
    `:eki_calibrated` (default): each run uses merged `_atmos_merged_parameters.toml` from the EKI calibration YAML
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
    """
    Which **varfix** legs to run (same ordering as nested loops). Default **`[false, true]`** = off then on.
    REPL: `ForwardSweepConfig(; varfix_values = [true])` for varfix-on only. CLI: `--varfix=on` / `off` / `both` or
    `--varfix=off,on`. Env: **`VA_FORWARD_SWEEP_VARFIX`** (same tokens as CLI).
    """
    varfix_values::Vector{Bool} = Bool[false, true]
    """
    If set, only registry rows whose merged [`va_forward_sweep_case_slug`](@ref) is in this list are run and plotted.
    REPL: `ForwardSweepConfig(; case_slugs = ["TRMM_LBA", "Bomex"])`. Env: comma-separated **`VA_FORWARD_SWEEP_CASE_SLUGS`**.
    CLI (`sweep_forward_runs.jl`): **`--case-slugs=a,b,c`**. `run_full_study.jl`: **`--forward-case-slugs=...`**.
    """
    case_slugs::Union{Nothing, Vector{String}} = nothing
end

"""
    va_forward_sweep_varfix_values_from_spec(s::AbstractString) -> Vector{Bool}

Build the varfix axis for [`ForwardSweepConfig`](@ref). Tokens (comma-separated) are case-insensitive:

  - **`off`**, **`vf0`**, **`false`**, **`0`** → `false` (base `sgs_distribution`, no vertical-profile SGS)
  - **`on`**, **`vf1`**, **`true`**, **`1`** → `true` (`(gaussian|lognormal)_vertical_profile*`)

Shorthand when the whole string has no comma: **`both`** → `[false, true]`; **`off`** alone → `[false]`; **`on`** alone → `[true]`.
"""
function va_forward_sweep_varfix_values_from_spec(s::AbstractString)::Vector{Bool}
    t = String(strip(s))
    isempty(t) && error("Empty varfix spec $(repr(s))")
    if !occursin(',', t)
        w = lowercase(t)
        w == "both" && return Bool[false, true]
        w in ("off", "vf0", "varfix_off", "false", "0") && return Bool[false]
        w in ("on", "vf1", "varfix_on", "true", "1") && return Bool[true]
        error("Unknown varfix spec $(repr(s)); use both, off, on, or comma-separated off/on/0/1/…")
    end
    out = Bool[]
    for part in split(t, ',', keepempty = false)
        w = lowercase(String(strip(part)))
        isempty(w) && continue
        if w in ("off", "vf0", "varfix_off", "false", "0")
            push!(out, false)
        elseif w in ("on", "vf1", "varfix_on", "true", "1")
            push!(out, true)
        else
            error("Unknown varfix token $(repr(part)) in $(repr(s))")
        end
    end
    isempty(out) && error("No varfix tokens parsed from $(repr(s))")
    return out
end

function va_forward_sweep_assert_nonempty_varfix!(cfg::ForwardSweepConfig)
    isempty(cfg.varfix_values) &&
        error("ForwardSweepConfig.varfix_values must be non-empty (e.g. [false], [true], or [false, true]).")
    return cfg
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
    if haskey(ENV, "VA_FORWARD_SWEEP_VARFIX")
        cfg.varfix_values = va_forward_sweep_varfix_values_from_spec(ENV["VA_FORWARD_SWEEP_VARFIX"])
    end
    if haskey(ENV, "VA_FORWARD_SWEEP_CASE_SLUGS")
        cfg.case_slugs = va_forward_sweep_parse_case_slugs(ENV["VA_FORWARD_SWEEP_CASE_SLUGS"])
    end
    return cfg
end

"""Parse comma-separated slugs from CLI / env (empty tokens dropped)."""
function va_forward_sweep_parse_case_slugs(s::AbstractString)::Vector{String}
    parts = split(s, ','; keepempty = false)
    return String[strip(p) for p in parts if !isempty(strip(p))]
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
    if cfg.case_slugs !== nothing && !isempty(cfg.case_slugs)
        wanted = Set{String}(String.(strip.(cfg.case_slugs)))
        have = Set(r.slug for r in rows)
        miss = setdiff(wanted, have)
        if !isempty(miss)
            @warn "case_slugs: not found in registry (check spelling vs merged YAML slugs)" missing = collect(miss)
        end
        rows = [r for r in rows if r.slug ∈ wanted]
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
    va_forward_sweep_varfix_on_distribution_list(case_dict) -> Union{Nothing,Vector{String}}

If merged case YAML defines **`forward_sweep_varfix_on_distributions`** (non-empty list of strings), the forward
sweep runs **one varfix-on task per entry** (each under `simulation_output/.../varfix_on_<slug>/...`). If the key is
absent, varfix-on uses a single task and the usual `.../varfix_on/...` directory (see [`run_one`](@ref) in
`sweep_forward_core.jl`).
"""
function va_forward_sweep_varfix_on_distribution_list(case_dict::AbstractDict)::Union{Nothing, Vector{String}}
    raw = get(case_dict, "forward_sweep_varfix_on_distributions", nothing)
    raw isa AbstractVector || return nothing
    out = String[]
    for x in raw
        s = String(strip(string(x)))
        isempty(s) || push!(out, s)
    end
    return isempty(out) ? nothing : out
end

"""Directory segment under `N_<n>/` for this varfix leg (`varfix_off`, `varfix_on`, or `varfix_on_<dist_slug>`)."""
function va_forward_sweep_varfix_dir_segment(
    varfix::Bool,
    forced_varfix_on_dist::Union{Nothing, AbstractString},
)::String
    varfix || return "varfix_off"
    if forced_varfix_on_dist === nothing
        return "varfix_on"
    end
    s = String(strip(string(forced_varfix_on_dist)))
    return isempty(s) ? "varfix_on" : string("varfix_on_", va_sgs_dist_path_slug(s))
end

"""
    va_flatten_forward_sweep_tasks(experiment_dir, cfg) -> Vector{Tuple}

Each row: `(model_config_layers, scm_toml, slug, n_quad, varfix_bool, tier, z_stretch, yaml_dz, eki_off, eki_on, forced_varfix_on_dist)`
where **`model_config_layers`** is `Vector{String}` (merge order); **`yaml_rel`** for grouping is `layers[end]`.
with `eki_off` / `eki_on` optional registry-relative experiment YAML names (or `nothing`).
The last field is **`nothing`** unless **`varfix_bool`** is true and the merged case YAML lists
[`va_forward_sweep_varfix_on_distribution_list`](@ref); then it is the **`sgs_distribution`** string for that task.

Output layout: `simulation_output/<slug>/<res_seg>/N_<n>/(varfix_off|varfix_on|varfix_on_<dist_slug>)/<forward_subdir>/output_active/`
where `<forward_subdir>` is `forward_eki` or `forward_only` (see [`va_forward_sweep_forward_subdir`](@ref)).
Varfix legs are those in **`cfg.varfix_values`** (default both off and on).
"""
function va_flatten_forward_sweep_tasks(experiment_dir::AbstractString, cfg::ForwardSweepConfig)
    va_forward_sweep_assert_nonempty_varfix!(cfg)
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
        Union{Nothing, String},
    }[]
    for row in rows
        tiers = va_resolution_tiers_for_forward(row.case_dict, cfg)
        z_stretch = va_forward_sweep_case_z_stretch(row.case_dict)
        yaml_dz = va_forward_sweep_case_dz_bottom(row.case_dict)
        vo_list = va_forward_sweep_varfix_on_distribution_list(row.case_dict)
        for tier in tiers, n in n_list, vf in cfg.varfix_values
            if vf && vo_list !== nothing
                for d in vo_list
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
                            d,
                        ),
                    )
                end
            else
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
                        nothing,
                    ),
                )
            end
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
    cfg::ForwardSweepConfig;
    forced_varfix_on_distribution::Union{Nothing, AbstractString} = nothing,
)
    tag = va_forward_sweep_varfix_dir_segment(varfix, varfix ? forced_varfix_on_distribution : nothing)
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
    forced_varfix_on_distribution::Union{Nothing, AbstractString} = nothing,
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
        ap = va_forward_sweep_output_active_path(
            experiment_dir,
            slug,
            seg,
            n_quad,
            varfix,
            cfg;
            forced_varfix_on_distribution = varfix ? forced_varfix_on_distribution : nothing,
        )
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
    forced_varfix_on_distribution::Union{Nothing, AbstractString} = nothing,
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
        ap = va_forward_sweep_output_active_path(
            experiment_dir,
            slug,
            seg,
            n_quad,
            varfix,
            cfg;
            forced_varfix_on_distribution = varfix ? forced_varfix_on_distribution : nothing,
        )
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
