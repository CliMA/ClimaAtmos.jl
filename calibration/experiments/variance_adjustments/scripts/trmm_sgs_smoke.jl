# TRMM_LBA diagnostic EDMF: run several `sgs_distribution` strings (default N_quad=3), then plot a condensate
# profile sum (default **`clw` + `cli`**).
#
# "Smoke" here means a **light regression / integration check** (many `sgs_distribution` strings end-to-end), not
# hardware power-on.
#
# Paths below are **absolute** (independent of shell `cwd`). Replace `<REPO>` with your ClimaAtmos.jl clone root.
# At runtime, `va_trmm_sgs_smoke_*` defaults use `VA_TRMM_SGS_SMOKE_ROOT` (the `variance_adjustments` directory).
#
# ## Run **and** plot (use this so you do not forget the second step)
#
# REPL — one call after `include(...)`:
#
# ```julia
# va_run_and_plot_trmm_sgs_smoke(; parallel = :distributed, distributed_procs = 4,
#     continue_on_errors = true, skip_done = true)
# ```
#
# CLI — `both` runs integrations then writes the figure (same flags as `run`):
#
# ```text
# julia --project=<VA_ROOT> <VA_ROOT>/scripts/trmm_sgs_smoke.jl both \\
#   --parallel=distributed --distributed-procs=4 --continue-on-errors --skip-done
# ```
#
# With `continue_on_errors=true`, failed distributions are logged under `simulation_output/sgs_smoke/_failures/`
# as one stable `*_N<n>_<dist_slug>.txt` per failing combo (overwrites on reruns; no timestamp suffix).
# Plotting still runs for any case that produced `output_active`.
#
# ## Parallelism
#
# - **Sequential (default):** `parallel=:sequential` — one integration at a time; lowest RAM, easiest to debug.
#
# - **Distributed:** `parallel=:distributed` runs one case per **worker process** via `Distributed.pmap` (separate
#   Julia heaps — safe for Atmos; unlike `Threads.@threads` over multiple solves in one process, which hits global
#   caches and **“Multiple concurrent writes to Dict”**). Start with `julia -p N --project=...` or pass
#   `distributed_procs=N` to target `N` workers with the same `--project` as this experiment directory.
#   You can also use `distributed_procs=:match_jobs` (or CLI `--distributed-procs=auto`) to target
#   one worker per smoke job (`length(distributions)`).
#   Spawning is **idempotent** in a long REPL session: only missing workers are added (target count,
#   not that many new workers on every call).
#
# - **Column threading (`julia -t`):** that parallelizes *inside* one column solve, not across smoke cases.
#
# - **GPUs:** If your build uses a GPU, assume **one active Atmos simulation per device** unless you know
#   otherwise; prefer **`-t 1`** per worker for smoke tests.
#
# ## REPL setup (one session; cwd arbitrary)
#
# `include` activates this file’s parent project (`variance_adjustments`) so CairoMakie and other deps load.
#
# ```julia
# include("/absolute/path/to/.../variance_adjustments/scripts/trmm_sgs_smoke.jl")
# ```
#
# Then either `va_run_and_plot_trmm_sgs_smoke(...)` (recommended) or `va_run_trmm_sgs_smoke()` and later `va_plot_trmm_sgs_smoke()`.
#
# ## CLI
#
# Subcommands: `run` (default), `plot`, or `both` (run then plot). Example:
#
#   julia --project=<VA_ROOT> <VA_ROOT>/scripts/trmm_sgs_smoke.jl both --parallel=distributed --distributed-procs=4
#
# Outputs: `<REPO>/calibration/experiments/variance_adjustments/simulation_output/sgs_smoke/<slug>/.../output_active`
# Figure (default `profile_sum_short`): `<REPO>/.../analysis/figures/sgs_smoke/<slug>/<res_seg>/N_<n>/profile_clw_plus_cli.png`
#          (`<res_seg>` / `N_<n>` match the simulation tree so different quadrature or tier do not overwrite.)
#          Basename follows `profile_sum_short`; see `va_plot_trmm_sgs_smoke` keyword docs in this file.
#
# Stale `…/N_<n>/<dist>/` trees (old `dist` names no longer in `VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS`) trigger plot
# warnings. Remove them with `va_trmm_sgs_smoke_remove_stale_dist_dirs!(; dry_run=true)` then `dry_run=false`.

const VA_TRMM_SGS_SMOKE_SCRIPT_PATH = abspath(@__FILE__)
import Pkg
Pkg.activate(joinpath(@__DIR__, "..") |> abspath)
import Distributed

include(joinpath(@__DIR__, "trmm_sgs_smoke_core.jl"))

function _va_trmm_sgs_smoke_load_distributed!(experiment_dir::AbstractString)
    core_path = abspath(joinpath(@__DIR__, "trmm_sgs_smoke_core.jl"))
    expdir = String(abspath(experiment_dir))
    cpath = String(core_path)
    # Do not use `Distributed.@everywhere` here: it breaks parsing of the rest of this file after `include(core)`.
    code = quote
        import Pkg
        Pkg.activate($expdir)
        include($cpath)
    end
    for w in Distributed.workers()
        Distributed.remotecall_eval(Main, w, code)
    end
    return nothing
end

function va_run_trmm_sgs_smoke(;
    experiment_dir::AbstractString = VA_TRMM_SGS_SMOKE_ROOT,
    case_layers::Vector{String} = VA_TRMM_SGS_SMOKE_CASE_LAYERS,
    scm_toml = VA_TRMM_SGS_SMOKE_SCM_TOML,
    distributions::Vector{String} = copy(VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS),
    n_quad::Int = 3,
    skip_done::Bool = false,
    continue_on_errors::Bool = false,
    external_forcing_file::Union{Nothing, AbstractString} = nothing,
    cfsite_number::Union{Nothing, AbstractString} = nothing,
    parallel::Symbol = :sequential,
    distributed_procs::Union{Int, Symbol} = 0,
)
    parallel in (:sequential, :distributed) ||
        error("parallel must be :sequential or :distributed, got $(repr(parallel))")
    g = va_trmm_sgs_smoke_resolve_grid(; experiment_dir, case_layers)
    fail_dir = joinpath(experiment_dir, "simulation_output", "sgs_smoke", "_failures")
    @info "TRMM SGS smoke" g.slug g.res_seg g.tier n_quad distributions parallel failure_logs = fail_dir

    if parallel === :sequential
        n_fail = 0
        for d in distributions
            try
                va_run_trmm_sgs_smoke_one(;
                    experiment_dir,
                    case_layers,
                    scm_toml,
                    n_quad,
                    dist = d,
                    tier = g.tier,
                    z_stretch = g.z_stretch,
                    yaml_dz = g.yaml_dz,
                    slug = g.slug,
                    skip_done,
                    external_forcing_file,
                    cfsite_number,
                )
            catch e
                bt = catch_backtrace()
                path = _va_sgs_smoke_write_failure!(
                    experiment_dir, g.slug, n_quad, d, e, bt; res_segment = g.res_seg,
                )
                n_fail += 1
                @error "Run failed (full stacktrace in file)" dist = d log_path = path summary =
                    sprint(showerror, e)
                continue_on_errors ||
                    (@info "Stopped after this failure; pass continue_on_errors=true or CLI --continue-on-errors to run remaining distributions."; rethrow())
            end
        end
        if n_fail > 0
            @warn "sgs_smoke: $n_fail run(s) failed; see *.txt under $fail_dir" n_fail fail_dir
        end
        return nothing
    end

    if distributed_procs isa Symbol &&
       !(distributed_procs in (:match_jobs, :auto, :match))
        error(
            "distributed_procs symbol must be one of :match_jobs / :auto / :match; got $(repr(distributed_procs))",
        )
    end
    target_workers = if distributed_procs isa Symbol
        # Match the job count (`length(distributions)`): convenient when the smoke list grows.
        length(distributions)
    else
        distributed_procs
    end
    if target_workers > 0
        proj = abspath(experiment_dir)
        # `addprocs(n)` always *adds* n new processes — repeated REPL calls would pile up workers.
        # Match a **target** worker count: only spawn the shortfall vs. `nworkers()`.
        need = target_workers - Distributed.nworkers()
        if need > 0
            Distributed.addprocs(need; exeflags = `--project=$(proj)`)
        end
    end
    if isempty(Distributed.workers())
        error(
            "parallel=:distributed requires Julia worker processes. Start with `julia -p N ...`, or pass " *
                "keyword `distributed_procs > 0` (or `:match_jobs`), or `Distributed.addprocs` before calling.",
        )
    end
    _va_trmm_sgs_smoke_load_distributed!(experiment_dir)
    scm_vec = Vector{Any}(collect(Any, scm_toml))
    efi = external_forcing_file === nothing ? nothing : String(external_forcing_file)
    cfs = cfsite_number === nothing ? nothing : String(cfsite_number)
    jobs = [
        (;
            experiment_dir = String(abspath(experiment_dir)),
            case_layers = copy(case_layers),
            scm_toml = scm_vec,
            n_quad,
            dist = String(d),
            tier = g.tier,
            z_stretch = g.z_stretch,
            yaml_dz = g.yaml_dz,
            slug = String(g.slug),
            skip_done,
            external_forcing_file = efi,
            cfsite_number = cfs,
        ) for d in distributions
    ]
    results = Distributed.pmap(_va_run_trmm_sgs_smoke_job, jobs)
    n_fail = 0
    for r in results
        r.ok && continue
        n_fail += 1
        @error "Run failed (full stacktrace in file)" dist = r.dist log_path = r.log_path summary = r.summary
        continue_on_errors ||
            error(
                "sgs_smoke run failed for $(repr(r.dist)); see $(r.log_path). " *
                    "Pass continue_on_errors=true or --continue-on-errors to finish all jobs and then error if any failed.",
            )
    end
    if n_fail > 0
        @warn "sgs_smoke: $n_fail run(s) failed; see *.txt under $fail_dir" n_fail fail_dir
    end
    return nothing
end

"""
    va_run_and_plot_trmm_sgs_smoke(; kwargs...)

Calls `va_run_trmm_sgs_smoke` then `va_plot_trmm_sgs_smoke` with the resolved grid slug. Same keywords as `va_run_trmm_sgs_smoke`,
plus optional **`profile_*`** keywords forwarded to `va_plot_trmm_sgs_smoke`.
With `continue_on_errors=true`, failures are logged under `simulation_output/sgs_smoke/_failures/` and plotting still runs for successful outputs.
"""
function va_run_and_plot_trmm_sgs_smoke(;
    experiment_dir::AbstractString = VA_TRMM_SGS_SMOKE_ROOT,
    case_layers::Vector{String} = VA_TRMM_SGS_SMOKE_CASE_LAYERS,
    scm_toml = VA_TRMM_SGS_SMOKE_SCM_TOML,
    distributions::Vector{String} = copy(VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS),
    n_quad::Int = 3,
    skip_done::Bool = false,
    continue_on_errors::Bool = false,
    external_forcing_file::Union{Nothing, AbstractString} = nothing,
    cfsite_number::Union{Nothing, AbstractString} = nothing,
    parallel::Symbol = :sequential,
    distributed_procs::Union{Int, Symbol} = 0,
    profile_short_a::AbstractString = "clw",
    profile_short_b::AbstractString = "cli",
    profile_sum_short::AbstractString = "clw_plus_cli",
    profile_xlabel::AbstractString = "clw+cli (kg/kg)",
)
    va_run_trmm_sgs_smoke(;
        experiment_dir,
        case_layers,
        scm_toml,
        distributions,
        n_quad,
        skip_done,
        continue_on_errors,
        external_forcing_file,
        cfsite_number,
        parallel,
        distributed_procs,
    )
    va_plot_trmm_sgs_smoke(;
        experiment_dir,
        slug = nothing,
        n_quad,
        profile_short_a,
        profile_short_b,
        profile_sum_short,
        profile_xlabel,
    )
    return nothing
end

"""Append `output_active` paths under `slug_root` matching `N_\$(n_quad)`."""
function _va_collect_sgs_smoke_output_actives_under!(
    out::Vector{Tuple{String, String}},
    slug_root::AbstractString,
    n_quad::Int,
)
    isdir(slug_root) || return out
    # ClimaAtmos `activelink` layout: `output_active` is a symlink to `output_NNNN`.
    # Default `walkdir(..., follow_symlinks=false)` never descends into it, so basename(root) never
    # equals `output_active` and plotting would see zero runs even when integrations succeeded.
    for (root, _, _) in walkdir(slug_root; follow_symlinks = true)
        basename(root) == "output_active" || continue
        forward_only = dirname(root)
        basename(forward_only) == "forward_only" || continue
        n_dir = basename(dirname(dirname(forward_only)))
        startswith(n_dir, "N_") || continue
        n_str = n_dir[3:end]
        parse(Int, n_str) != n_quad && continue
        isdir(root) || continue
        dist_slug = basename(dirname(forward_only))
        push!(out, (abspath(root), dist_slug))
    end
    return out
end

"""
    va_collect_sgs_smoke_output_actives(experiment_dir; slug, n_quad)

Find `forward_only/output_active` dirs for Gauss–Legendre order `n_quad`.

Uses `walkdir(..., follow_symlinks = true)` because ClimaAtmos `activelink` output makes `output_active` a symlink; the default `follow_symlinks = false` skips it and would find no paths.

- `slug::AbstractString`: only under `simulation_output/sgs_smoke/\$slug/`.
- `slug = nothing` (**default**): every case directory under `sgs_smoke` except `_failures`.
"""
function va_collect_sgs_smoke_output_actives(
    experiment_dir::AbstractString;
    slug::Union{Nothing, AbstractString} = nothing,
    n_quad::Int = 3,
)::Vector{Tuple{String, String}}
    sgs_root = joinpath(experiment_dir, "simulation_output", "sgs_smoke")
    !isdir(sgs_root) && return Tuple{String, String}[]
    out = Tuple{String, String}[]
    if slug === nothing
        for name in sort(readdir(sgs_root))
            name == "_failures" && continue
            p = joinpath(sgs_root, name)
            isdir(p) || continue
            _va_collect_sgs_smoke_output_actives_under!(out, p, n_quad)
        end
    else
        _va_collect_sgs_smoke_output_actives_under!(out, joinpath(sgs_root, String(slug)), n_quad)
    end
    sort!(out; by = x -> x[2])
    return out
end

"""
    va_trmm_sgs_smoke_remove_stale_dist_dirs!(; experiment_dir, n_quad, dry_run) -> Vector{String}

Collect paths to `simulation_output/sgs_smoke/<case>/<tier>/N_<n_quad>/<dist>/` where `dist` is **not** in
`VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS` (same rule as plotting). With `dry_run=true` (default), only log what
would be deleted; with `dry_run=false`, call `rm(...; recursive=true)` on each. Skips `_failures/`.
"""
function va_trmm_sgs_smoke_remove_stale_dist_dirs!(;
    experiment_dir::AbstractString = VA_TRMM_SGS_SMOKE_ROOT,
    n_quad::Int = 3,
    dry_run::Bool = true,
)::Vector{String}
    allowed = Set{String}(VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS)
    root = joinpath(experiment_dir, "simulation_output", "sgs_smoke")
    to_remove = String[]
    isdir(root) || return to_remove
    n_seg = "N_$(n_quad)"
    for case in readdir(root)
        case == "_failures" && continue
        case_path = joinpath(root, case)
        isdir(case_path) || continue
        for tier in readdir(case_path)
            tier_p = joinpath(case_path, tier)
            isdir(tier_p) || continue
            n_p = joinpath(tier_p, n_seg)
            isdir(n_p) || continue
            for dist in readdir(n_p)
                dist in allowed && continue
                dpath = joinpath(n_p, dist)
                isdir(dpath) || continue
                push!(to_remove, abspath(dpath))
            end
        end
    end
    sort!(to_remove)
    if dry_run
        if isempty(to_remove)
            @info "dry_run: no stale sgs_smoke dist dirs under N_$(n_quad) (all folder names ⊆ default list)" root
        else
            @info "dry_run: would remove $(length(to_remove)) stale dir(s); rerun with dry_run=false to delete" to_remove
        end
    else
        for p in to_remove
            rm(p; recursive = true)
        end
        @info "Removed $(length(to_remove)) stale sgs_smoke dist dir(s)" to_remove
    end
    return to_remove
end

"""
    _va_trmm_sgs_smoke_pairs_matching_grid(pairs, case_slug, res_seg)

Restrict to paths under `simulation_output/sgs_smoke/<slug>/<res_seg>/...` so stale runs from another
`z*_dt*` tier are not plotted (avoids **duplicate legend labels** for the same `dist` slug).
"""
function _va_trmm_sgs_smoke_pairs_matching_grid(
    pairs::Vector{Tuple{String, String}},
    case_slug::AbstractString,
    res_seg::AbstractString,
)::Vector{Tuple{String, String}}
    s_slug, s_seg = String(case_slug), String(res_seg)
    out = Tuple{String, String}[]
    for (p, lab) in pairs
        sp = splitpath(abspath(p))
        i = findfirst(==(s_slug), sp)
        if i !== nothing && i + 1 <= length(sp) && sp[i + 1] == s_seg
            push!(out, (p, lab))
        end
    end
    return out
end

function _va_trmm_sgs_smoke_ln_slugs_ordered()::Vector{String}
    return String[d for d in VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS if startswith(d, "lognormal")]
end

function _va_trmm_sgs_smoke_g_slugs_ordered()::Vector{String}
    return String[d for d in VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS if startswith(d, "gaussian")]
end

"""1-based index of `slug` in the default smoke LN (or G) list; stable sort key and unique line styling."""
function _va_trmm_sgs_smoke_series_index(slug::AbstractString)::Int
    s = String(slug)
    if startswith(s, "lognormal")
        lst = _va_trmm_sgs_smoke_ln_slugs_ordered()
    elseif startswith(s, "gaussian")
        lst = _va_trmm_sgs_smoke_g_slugs_ordered()
    else
        error(
            "sgs_smoke plot: unexpected distribution folder name $(repr(s)); expected a string starting with " *
                "`lognormal` or `gaussian` (same spelling as YAML `sgs_distribution`).",
        )
    end
    i = findfirst(==(s), lst)
    i === nothing && error(
        "sgs_smoke plot: folder name $(repr(s)) is not in the LN/G ordering derived from " *
            "`VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS`. Output dirs must use `va_sgs_dist_path_slug(sgs_distribution)` " *
            "matching that list exactly (no silent fallback styling).",
    )
    return i
end

"""Order paths for plotting/legend: all **lognormal** (default list order) then all **gaussian**."""
function _va_trmm_sgs_sort_pairs_ln_then_g(pairs::Vector{Tuple{String, String}})
    ln = Tuple{String, String}[]
    g = Tuple{String, String}[]
    other = Tuple{String, String}[]
    for p in pairs
        lab = p[2]
        if startswith(lab, "lognormal")
            push!(ln, p)
        elseif startswith(lab, "gaussian")
            push!(g, p)
        else
            push!(other, p)
        end
    end
    ln_s = sort(ln; by = x -> _va_trmm_sgs_smoke_series_index(x[2]))
    g_s = sort(g; by = x -> _va_trmm_sgs_smoke_series_index(x[2]))
    return [ln_s; g_s; other]
end

"""
    va_trmm_sgs_smoke_short_label(slug) -> String

Legend text for an `sgs_distribution` path folder / slug: **`LN`/`G`** for the two baselines; otherwise the
YAML tail after `lognormal_` / `gaussian_` / `*_vertical_profile_`, with underscores shown as **`·`** so every
default smoke case gets a **unique** label (matches the string you would pass in YAML up to typography).
"""
function _va_sgs_smoke_legend_tail(s::AbstractString)::String
    return replace(String(s), '_' => '·')
end

function va_trmm_sgs_smoke_short_label(slug::AbstractString)::String
    s = String(slug)
    if s == "lognormal"
        return "LN"
    elseif s == "gaussian"
        return "G"
    elseif startswith(s, "lognormal_vertical_profile_")
        return "LN·" * _va_sgs_smoke_legend_tail(chopprefix(s, "lognormal_vertical_profile_"))
    elseif startswith(s, "gaussian_vertical_profile_")
        return "G·" * _va_sgs_smoke_legend_tail(chopprefix(s, "gaussian_vertical_profile_"))
    elseif startswith(s, "lognormal_")
        return "LN·" * _va_sgs_smoke_legend_tail(chopprefix(s, "lognormal_"))
    elseif startswith(s, "gaussian_")
        return "G·" * _va_sgs_smoke_legend_tail(chopprefix(s, "gaussian_"))
    else
        return _va_sgs_smoke_legend_tail(s)
    end
end

# **Lognormal = cool family** (blue → cyan → teal → indigo): deliberately *not* green so it won’t read as
# “the other distribution”. **Gaussian = warm family** (orange → red → gold → rust): brown sits on the red–
# orange side of the wheel, not mixed into blues.
const _VA_SGS_LN_COLORS = (
    CM.RGBf(0.00, 0.30, 0.65),  # 1 strong blue
    CM.RGBf(0.25, 0.55, 0.92),  # 2 sky
    CM.RGBf(0.00, 0.62, 0.72),  # 3 cyan
    CM.RGBf(0.12, 0.38, 0.62),  # 4 steel / navy
    CM.RGBf(0.45, 0.32, 0.78),  # 5 blue–violet
    CM.RGBf(0.05, 0.48, 0.52),  # 6 teal (still cool, not olive)
)
const _VA_SGS_G_COLORS = (
    CM.RGBf(0.88, 0.38, 0.05),  # 1 orange
    CM.RGBf(0.82, 0.10, 0.14),  # 2 red
    CM.RGBf(0.95, 0.72, 0.12),  # 3 gold
    CM.RGBf(0.96, 0.48, 0.38),  # 4 coral
    CM.RGBf(0.62, 0.28, 0.10),  # 5 rust / brown-red
    CM.RGBf(1.00, 0.55, 0.00),  # 6 vivid orange
)
const _VA_SGS_LN_BASELINE_COLOR = CM.RGBf(0.10, 0.74, 0.98)  # bright cyan-blue
const _VA_SGS_LN_CUBATURE_COLOR = CM.RGBf(0.00, 0.18, 0.52)  # deep navy
const _VA_SGS_G_BASELINE_COLOR = CM.RGBf(1.00, 0.62, 0.18)   # bright orange
const _VA_SGS_G_CUBATURE_COLOR = CM.RGBf(0.60, 0.08, 0.08)   # deep red

# Built-in symbols only (custom `Linestyle` vectors break outside legends in CairoMakie).
const _VA_SGS_SMOKE_LINE_STYLE_CYCLE = (:solid, :dash, :dashdot, :dot, :dashdotdot)

@inline function _va_trmm_style_method_key(slug::AbstractString)::String
    s = String(slug)
    if s == "lognormal" || s == "gaussian"
        return "baseline_2d"
    elseif occursin("_vertical_profile_full_cubature", s)
        return "full_cubature"
    elseif occursin("_vertical_profile_inner_bracketed", s)
        return "inner_bracketed"
    elseif occursin("_vertical_profile_inner_halley", s)
        return "inner_halley"
    elseif occursin("_vertical_profile_lhs_z", s)
        return "lhs_z"
    elseif occursin("_vertical_profile_principal_axis", s)
        return "principal_axis"
    elseif occursin("_vertical_profile_voronoi", s)
        return "voronoi"
    elseif occursin("_vertical_profile_barycentric", s)
        return "barycentric"
    else
        return "other"
    end
end

"""
Deterministic `(color, linestyle, linewidth)` for a path slug.

Uses one slot per entry in `VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS` within the LN / G block (see
[`_va_trmm_sgs_smoke_series_index`](@ref)) so many `*_vertical_profile_*` strings do not share the same
Makie attributes (which would draw indistinguishable overlays even when the NetCDF profiles differ).
"""
function _va_trmm_sgs_style_for_dist_slug(slug::AbstractString)
    s = String(slug)
    vi = _va_trmm_sgs_smoke_series_index(s)
    is_lognormal = startswith(s, "lognormal")
    colors = if is_lognormal
        _VA_SGS_LN_COLORS
    elseif startswith(s, "gaussian")
        _VA_SGS_G_COLORS
    else
        error("sgs_smoke plot: cannot assign palette for $(repr(s))")
    end
    nc = length(colors)
    nls = length(_VA_SGS_SMOKE_LINE_STYLE_CYCLE)
    color = colors[mod1(vi, nc)]
    method_key = _va_trmm_style_method_key(s)
    # Width hierarchy for readability:
    #   full cubature (truth proxy) > baseline 2D > all reduced 3D→2D variants.
    is_full_cubature = method_key == "full_cubature"
    is_baseline_2d = method_key == "baseline_2d"
    ls = if method_key == "full_cubature"
        :solid
    elseif method_key == "baseline_2d"
        :dash
    elseif method_key == "inner_bracketed"
        :dashdot
    elseif method_key == "inner_halley"
        :dot
    elseif method_key == "lhs_z"
        :dashdotdot
    elseif method_key == "principal_axis"
        :dash
    elseif method_key == "voronoi"
        :dashdot
    elseif method_key == "barycentric"
        :dot
    else
        _VA_SGS_SMOKE_LINE_STYLE_CYCLE[mod1(vi + 1, nls)]
    end
    color = if method_key == "full_cubature"
        is_lognormal ? _VA_SGS_LN_CUBATURE_COLOR : _VA_SGS_G_CUBATURE_COLOR
    elseif method_key == "baseline_2d"
        is_lognormal ? _VA_SGS_LN_BASELINE_COLOR : _VA_SGS_G_BASELINE_COLOR
    else
        color
    end
    lw = if is_full_cubature
        3.4f0
    elseif is_baseline_2d
        2.3f0
    else
        1.7f0
    end
    return (color, ls, lw)
end

"""
    va_plot_trmm_sgs_smoke(; kwargs...)

Write the TRMM SGS smoke summed-profile figure from completed `output_active` runs.

**`profile_short_a`**, **`profile_short_b`**, **`profile_sum_short`**, **`profile_xlabel`** are forwarded to
`va_plot_profile_shortname_sum!` in `analysis/plotting/plot_profiles.jl` (defaults: `clw` + `cli` → `clw_plus_cli`).
"""
function va_plot_trmm_sgs_smoke(;
    experiment_dir::AbstractString = VA_TRMM_SGS_SMOKE_ROOT,
    slug::Union{Nothing, AbstractString} = nothing,
    n_quad::Int = 3,
    profile_short_a::AbstractString = "clw",
    profile_short_b::AbstractString = "cli",
    profile_sum_short::AbstractString = "clw_plus_cli",
    profile_xlabel::AbstractString = "clw+cli (kg/kg)",
)
    g = va_trmm_sgs_smoke_resolve_grid(; experiment_dir)
    fig_slug = g.slug
    pairs_all = va_collect_sgs_smoke_output_actives(experiment_dir; slug, n_quad)
    pairs = _va_trmm_sgs_smoke_pairs_matching_grid(pairs_all, g.slug, g.res_seg)
    if isempty(pairs) && !isempty(pairs_all)
        @warn "No output_active under resolved grid tier $(g.res_seg) (case $(g.slug)); found $(length(pairs_all)) path(s) under other tiers — use matching YAML tier or remove stale dirs." n_other =
            length(pairs_all)
    end
    # Only plot distributions in the current smoke list. `walkdir` still finds every `output_active` under the
    # case/tier — including stale dirs left after changing `VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS` or runs
    # from another clone — which would otherwise add stale slug lines back in.
    allowed = Set{String}(VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS)
    skipped = String[]
    pairs_f = Tuple{String, String}[]
    for p in pairs
        lab = p[2]
        if lab in allowed
            push!(pairs_f, p)
        else
            push!(skipped, lab)
        end
    end
    if !isempty(skipped)
        @warn "Skipping sgs_smoke output_active not in VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS (remove those dirs or they linger on disk). Delete with `va_trmm_sgs_smoke_remove_stale_dist_dirs!(; dry_run=false)` after `dry_run=true`." unique_skipped =
            unique(sort(skipped))
    end
    pairs = pairs_f
    if isempty(pairs)
        root = abspath(joinpath(experiment_dir, "simulation_output", "sgs_smoke"))
        tier_allow_hint =
            !isempty(skipped) ?
            "\n  Under this grid tier, found $(length(skipped)) `output_active` path(s) whose dist folder name is not in `VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS` (see @warn above). Delete stale `…/N_$n_quad/<dist>/` trees (e.g. `va_trmm_sgs_smoke_remove_stale_dist_dirs!(; dry_run=false)`) or re-run smoke with the current default list." :
            ""
        hint = if !isdir(root)
            "Directory does not exist: $root\n  → Runs write under the same `experiment_dir` as `VA_TRMM_SGS_SMOKE_ROOT` " *
            "(scripts parent = variance_adjustments). Pass `experiment_dir=\"/path/to/that/folder\"` if outputs live elsewhere."
        else
            sub = filter(x -> x != "_failures", readdir(root))
            fail_root = joinpath(root, "_failures")
            n_fail_logs = isdir(fail_root) ? count(x -> endswith(x, ".txt"), readdir(fail_root)) : 0
            "Looked under: $root\n  Case subdirs found: $(repr(sub))\n  → Need `.../<case>/<z*_dt*>/N_$(n_quad)/<dist>/forward_only/output_active` " *
            "(ClimaAtmos only materializes this after a successful integration). Try another `n_quad=` if you used a different quadrature order.\n" *
            "  Each smoke run that errors (including `solve_atmos!` → `:simulation_crashed`) should leave a log under `_failures/`; " *
            "if you see \"Finished sgs_smoke run\" in the REPL but no `output_active`, re-run with an updated `trmm_sgs_smoke_core.jl` that checks `ret_code`.\n" *
            (n_fail_logs > 0 ? "  Found $n_fail_logs `.txt` file(s) in `_failures/` — no completed `output_active` paths to plot.\n" : "")
        end
        error(
            "No sgs_smoke output_active dirs for n_quad=$n_quad.\n" * hint * tier_allow_hint *
            "\n  Or run `va_run_and_plot_trmm_sgs_smoke()` / CLI `both` after integrations finish.",
        )
    end
    pairs = _va_trmm_sgs_sort_pairs_ln_then_g(pairs)
    paths = String[p[1] for p in pairs]
    labels = String[p[2] for p in pairs]
    model_config_rel = VA_TRMM_SGS_SMOKE_CASE_LAYERS
    if length(unique(paths)) != length(paths)
        error(
            "sgs_smoke plot: duplicate `output_active` path(s) — each series must come from a distinct run directory. " *
                "labels = $(repr(labels))",
        )
    end
    short_labels = String[va_trmm_sgs_smoke_short_label(lab) for lab in labels]
    sty = [_va_trmm_sgs_style_for_dist_slug(lab) for lab in labels]
    colors = CM.RGBf[s[1] for s in sty]
    path_linestyles = Any[s[2] for s in sty]
    path_linewidths = Float32[s[3] for s in sty]
    z_top_m = va_condensate_cloud_top_height_m(
        paths,
        nothing,
        nothing;
        experiment_dir,
        model_config_rel,
        padding_m = 380.0,
        # Absolute floor: suppress float noise; relative term: τ = max(floor, frac * q_peak) scales with case.
        condensate_floor_kg_kg = 1e-12,
        condensate_floor_frac_of_peak = 1e-4,
    )
    analysis_dir = joinpath(experiment_dir, "analysis")
    # Match `simulation_output/sgs_smoke/<slug>/<res_seg>/N_<n>/…` so figures for different `n_quad` or
    # resolved grid tier do not overwrite each other.
    figdir = joinpath(analysis_dir, "figures", "sgs_smoke", fig_slug, g.res_seg, "N_$(n_quad)")
    n_series = length(paths)
    # Legend: own column; series order is LN block then G block (each in `VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS` order).
    # `plot_profiles.jl` calls `resize_to_layout!` before `save`.
    # Aspect: profile plots read best with height ≥ axis width; keep total Figure narrower (~½ previous width).
    # PNG: `px_per_unit` keeps output ~4–5k px tall so zoom stays sharp; PDF is vector (`save_pdf_also`).
    fig_w = 620
    fig_h = 720
    prof_sum_human = "$(profile_short_a) + $(profile_short_b)"
    p = va_plot_profile_shortname_sum!(
        paths;
        experiment_dir,
        model_config_rel,
        outdir = figdir,
        path_labels = short_labels,
        path_colors = colors,
        path_linestyles = path_linestyles,
        path_linewidths = path_linewidths,
        short_a = profile_short_a,
        short_b = profile_short_b,
        sum_short = profile_sum_short,
        xlabel = profile_xlabel,
        profile_title = "TRMM SGS smoke (N=$(n_quad), $(g.res_seg)): $(prof_sum_human), final output time\n" *
            "(cool hues = lognormal · warm hues = gaussian; legend lists LN then G)",
        legend_outside = true,
        ylims_height_max = z_top_m,
        figure_size = (fig_w, fig_h),
        compact_x_axis = true,
        legend_labelsize = 9,
        figure_save_px_per_unit = 7,
        save_pdf_also = true,
    )
    p === nothing &&
        error("Plotting produced no lines (missing $(repr(profile_short_a)) / $(repr(profile_short_b)) in output or case YAML?)")
    @info "Wrote figure" p
    return p
end

function _va_trmm_sgs_smoke_parse_slug_nquad(argv::Vector{String})
    slug = nothing  # discover all case dirs under sgs_smoke
    n_quad = 3
    for a in argv
        if startswith(a, "--slug=")
            v = String(strip(split(a, '=', limit = 2)[2]))
            slug = v == "auto" || isempty(v) ? nothing : v
        elseif startswith(a, "--n-quad=")
            n_quad = parse(Int, split(a, '=', limit = 2)[2])
        end
    end
    return slug, n_quad
end

function va_trmm_sgs_smoke_cli_run(argv::Vector{String})
    skip_done = false
    continue_on_errors = false
    n_quad = 3
    dists = String[]
    print_only = false
    parallel = :sequential
    distributed_procs::Union{Int, Symbol} = 0
    for a in argv
        if a == "-h" || a == "--help"
            println("""
trmm_sgs_smoke.jl — TRMM SGS smoke (sequential or Distributed workers). See file header.

Experiment root (project): $(VA_TRMM_SGS_SMOKE_ROOT)
This script:              $(VA_TRMM_SGS_SMOKE_SCRIPT_PATH)

  julia --project=$(VA_TRMM_SGS_SMOKE_ROOT) $(VA_TRMM_SGS_SMOKE_SCRIPT_PATH) run [options]
  julia -p 4 --project=$(VA_TRMM_SGS_SMOKE_ROOT) $(VA_TRMM_SGS_SMOKE_SCRIPT_PATH) run --parallel=distributed
  julia --project=$(VA_TRMM_SGS_SMOKE_ROOT) $(VA_TRMM_SGS_SMOKE_SCRIPT_PATH) run --parallel=distributed --distributed-procs=4
  julia --project=$(VA_TRMM_SGS_SMOKE_ROOT) $(VA_TRMM_SGS_SMOKE_SCRIPT_PATH) plot [--slug=TRMM_LBA] [--n-quad=...]
    (omit --slug to scan every case under sgs_smoke; use --slug=auto same as omit)
  julia --project=$(VA_TRMM_SGS_SMOKE_ROOT) $(VA_TRMM_SGS_SMOKE_SCRIPT_PATH) both [same options as run]

  `both` = run all cases, then write analysis/figures/sgs_smoke/<slug>/<res_seg>/N_<n>/profile_<sum_short>.png (default clw_plus_cli).
  With `--continue-on-errors`, failed jobs are logged under _failures/ and plotting still runs for successful outputs.

Options (run / both): --skip-done  --continue-on-errors  --n-quad=N  --distributions=a,b,c  --print-only
  --parallel=sequential|distributed   --distributed-procs=N|auto  (target N workers, or match smoke job count; optional if you already used -p)
Failures: full stacktraces under simulation_output/sgs_smoke/_failures/<slug>/<res_seg>/N_<n>/<dist>.txt (stable name; overwrites on rerun; res_seg matches the resolved grid path segment)
""")
            return
        elseif startswith(a, "--parallel=")
            v = String(lowercase(strip(split(a, '=', limit = 2)[2])))
            if v == "threads"
                error(
                    "parallel=threads is not supported: multiple Atmos solves in one process are not thread-safe " *
                        "(global caches → \"Multiple concurrent writes to Dict\"). Use --parallel=distributed with " *
                        "`julia -p N` or --distributed-procs=N.",
                )
            elseif v == "sequential" || v == "distributed"
                parallel = Symbol(v)
            else
                error("Unknown --parallel=$(repr(v)); use sequential or distributed.")
            end
        elseif startswith(a, "--distributed-procs=")
            v = String(strip(split(a, '=', limit = 2)[2]))
            if lowercase(v) in ("auto", "match", "match_jobs", "jobs")
                distributed_procs = :match_jobs
            else
                distributed_procs = parse(Int, v)
            end
        elseif a == "--skip-done"
            skip_done = true
        elseif a == "--continue-on-errors"
            continue_on_errors = true
        elseif startswith(a, "--n-quad=")
            n_quad = parse(Int, split(a, '=', limit = 2)[2])
        elseif startswith(a, "--distributions=")
            raw = split(split(a, '=', limit = 2)[2], ','; keepempty = false)
            dists = String[strip(String(x)) for x in raw if !isempty(strip(String(x)))]
        elseif a == "--print-only"
            print_only = true
        elseif startswith(a, "--slug=") || startswith(a, "--")
            continue
        else
            error("Unknown argument $(repr(a)); run: julia --project=$(VA_TRMM_SGS_SMOKE_ROOT) $(VA_TRMM_SGS_SMOKE_SCRIPT_PATH) run --help")
        end
    end
    isempty(dists) && (dists = copy(VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS))
    g = va_trmm_sgs_smoke_resolve_grid()
    @info "TRMM SGS smoke grid" g.slug g.res_seg g.tier n_quad distributions = dists parallel distributed_procs continue_on_errors
    print_only && return nothing
    return va_run_trmm_sgs_smoke(;
        distributions = dists,
        n_quad,
        skip_done,
        continue_on_errors,
        parallel,
        distributed_procs,
    )
end

function _trmm_sgs_smoke_main_cli()
    argv = String.(ARGS)
    mode = :run
    if !isempty(argv) && !startswith(argv[1], "--")
        mode = Symbol(argv[1])
        argv = argv[2:end]
    end
    mode in (:run, :plot, :both) ||
        error("First argument must be run, plot, or both (or omit for run); got $(repr(mode))")

    if mode == :run
        va_trmm_sgs_smoke_cli_run(argv)
    elseif mode == :plot
        slug, n_quad = _va_trmm_sgs_smoke_parse_slug_nquad(argv)
        va_plot_trmm_sgs_smoke(; slug, n_quad)
    else
        va_trmm_sgs_smoke_cli_run(argv)
        slug, n_quad = _va_trmm_sgs_smoke_parse_slug_nquad(argv)
        va_plot_trmm_sgs_smoke(; slug, n_quad)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    _trmm_sgs_smoke_main_cli()
end
