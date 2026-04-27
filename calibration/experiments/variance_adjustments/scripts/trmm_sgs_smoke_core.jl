# Shared TRMM SGS smoke definitions: loaded on the manager (via `trmm_sgs_smoke.jl`) and on
# `Distributed` workers for `parallel=:distributed`.

if abspath(PROGRAM_FILE) == @__FILE__
    error("Load scripts/trmm_sgs_smoke.jl instead of this core file.")
end

const VA_TRMM_SGS_SMOKE_ROOT = joinpath(@__DIR__, "..") |> abspath

include(joinpath(VA_TRMM_SGS_SMOKE_ROOT, "analysis", "plotting", "plot_profiles.jl"))
include(joinpath(VA_TRMM_SGS_SMOKE_ROOT, "scripts", "resolution_ladder.jl"))
include(joinpath(VA_TRMM_SGS_SMOKE_ROOT, "lib", "forward_sweep_grid.jl"))

import Base.CoreLogging: AbstractLogger, handle_message, shouldlog, min_enabled_level
import Logging: ConsoleLogger, SimpleLogger, with_logger
import CairoMakie as CM
import ClimaAtmos as CA
import NCDatasets as NCD

"""Forward all log messages to two loggers (stderr + per-run file) so `@error` from `solve_atmos!` is not lost when batching."""
struct _VA_TeeLogger <: AbstractLogger
    a::AbstractLogger
    b::AbstractLogger
end
function handle_message(tee::_VA_TeeLogger, level, message, m, g, i, f, l; k...)
    if shouldlog(tee.a, level, m, g, i)
        handle_message(tee.a, level, message, m, g, i, f, l; k...)
    end
    if shouldlog(tee.b, level, m, g, i)
        handle_message(tee.b, level, message, m, g, i, f, l; k...)
    end
    return nothing
end
function shouldlog(tee::_VA_TeeLogger, a...)
    return shouldlog(tee.a, a...) || shouldlog(tee.b, a...)
end
min_enabled_level(tee::_VA_TeeLogger) = min(min_enabled_level(tee.a), min_enabled_level(tee.b))

function _va_read_file_tail(path::AbstractString; max_bytes::Int = 12_000)::String
    isfile(path) || return ""
    s = filesize(path)
    return open(String(path), "r") do io
        if s <= max_bytes
            read(io, String)
        else
            seek(io, s - max_bytes)
            read(io, String)
        end
    end
end

const VA_TRMM_SGS_SMOKE_CASE_LAYERS = String[
    "model_configs/master_column_varquad_diagnostic_edmfx.yml",
    "model_configs/trmm_column_varquad_hires.yml",
]

const VA_TRMM_SGS_SMOKE_SCM_TOML = Any["diagnostic_edmfx_1M.toml", "toml/uncalibrated_stability_overlay.toml"]

# TRMM + LogNormal VPF (esp. `*_vertical_profile_*`) vs Gaussian: the crash in logs is *where* it surfaces
# (often `exner`/`liquid_ice_pottemp` with invalid pressure), not *why*. The suspect upstream bug was VPF
# mixing `d(ln q)/dz` with layer/Rosenblatt geometry that assumes physical `dq/dz` — see
# `subgrid_layer_profile_quadrature.jl` `integrate_over_sgs` (LogNormal branch) and comments there.
# Gaussian does not use that re-sloping path.
#
# Smoke lists **explicit** `sgs_distribution` strings so regressions name the discretization (see this branch’s
# `calibration/experiments/variance_adjustments/README.md`). Omit bare `*_vertical_profile` if you want every
# run to spell out an inner-quadrature variant.
const VA_TRMM_SGS_SMOKE_DEFAULT_DISTRIBUTIONS = String[
    "lognormal",
    "gaussian",
    "lognormal_vertical_profile_inner_bracketed",
    "lognormal_vertical_profile_full_cubature",
    "lognormal_vertical_profile_lhs_z",
    "lognormal_vertical_profile_principal_axis",
    "lognormal_vertical_profile_voronoi",
    "lognormal_vertical_profile_barycentric",
    "lognormal_vertical_profile_inner_halley",
    "lognormal_vertical_profile_inner_chebyshev",
    "gaussian_vertical_profile_inner_bracketed",
    "gaussian_vertical_profile_full_cubature",
    "gaussian_vertical_profile_lhs_z",
    "gaussian_vertical_profile_principal_axis",
    "gaussian_vertical_profile_voronoi",
    "gaussian_vertical_profile_barycentric",
    "gaussian_vertical_profile_inner_halley",
    "gaussian_vertical_profile_inner_chebyshev",
]

"""
Write `showerror` + truncated backtrace for a failed smoke run (REPL truncates; open this file instead).

If `solver_log_path` is provided and the file exists, a tail of that file is appended. That log is written
 during `solve_atmos!` and captures the `@error` block from `ClimaAtmos` (the exception is not returned in
`AtmosSolveResults` when `ret_code == :simulation_crashed`).

Filename is stable per `(case_slug, [res_segment,] quadrature_order, sgs_distribution)` and **overwrites** on
a later failure for the same key so `_failures/` does not accumulate timestamped copies. When
`res_segment` is set (default), the path is nested as
`_failures/<slug>/<res_segment>/N_<n>/<dist>.txt` to mirror the simulation tree.
If omitted, this falls back to a flat legacy filename.
"""
function _va_sgs_smoke_write_failure!(
    experiment_dir::AbstractString,
    slug::AbstractString,
    n_quad::Int,
    dist::AbstractString,
    err,
    bt;
    res_segment::Union{Nothing, AbstractString} = nothing,
    max_backtrace_lines::Int = 48,
    solver_log_path::Union{Nothing, String} = nothing,
    solver_log_tail_max_bytes::Int = 12_000,
)::String
    fail_root = joinpath(experiment_dir, "simulation_output", "sgs_smoke", "_failures")
    mkpath(fail_root)
    safe_dist = va_sgs_dist_path_slug(dist)
    path = if res_segment === nothing
        joinpath(fail_root, "$(slug)_N$(n_quad)_$(safe_dist).txt")
    else
        rs = replace(String(res_segment), r"[^\w\.\-]+" => "_")
        joinpath(fail_root, String(slug), rs, "N_$(n_quad)", "$(safe_dist).txt")
    end
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "sgs_distribution = ", dist)
        println(
            io,
            "case_slug = ", slug, "  res_segment = ", res_segment === nothing ? "(none)" : res_segment,
            "  quadrature_order = ", n_quad,
        )
        println(io, repeat("=", 72))
        println(io)
        showerror(io, err, bt)
        println(io)
        println(io, repeat("-", 72))
        bt_s = sprint() do brio
            Base.show_backtrace(brio, bt)
        end
        lines = split(bt_s, '\n'; keepempty = false)
        if length(lines) > max_backtrace_lines
            println(io, join(lines[1:max_backtrace_lines], '\n'))
            println(
                io,
                "\n... ($(length(lines) - max_backtrace_lines) backtrace lines omitted; see manager/worker stderr for full trace)",
            )
        else
            print(io, bt_s)
        end
        println(io)
        if solver_log_path !== nothing && isfile(solver_log_path)
            println(io, repeat("=", 72))
            println(io, "tail of solver log (from solve_atmos!): ", solver_log_path)
            println(io, repeat("=", 72))
            print(io, _va_read_file_tail(solver_log_path; max_bytes = solver_log_tail_max_bytes))
            println(io)
        elseif solver_log_path !== nothing
            println(io, repeat("=", 72))
            println(io, "solver log path (missing file): ", solver_log_path)
            println(io, repeat("=", 72))
        end
    end
    return path
end

function _va_solver_log_path(;
    experiment_dir::AbstractString,
    slug::AbstractString,
    n_quad::Int,
    dist::AbstractString,
    tier,
    z_stretch::Bool,
    yaml_dz::Float64,
)
    dist_slug = va_sgs_dist_path_slug(dist)
    return joinpath(
        experiment_dir,
        "simulation_output",
        "sgs_smoke",
        slug,
        va_tier_path_segment(tier, z_stretch, yaml_dz),
        "N_$(n_quad)",
        dist_slug,
        "forward_only",
        "sgs_smoke_solve.log",
    )
end

"""
Raise if final `clw`/`cli` profile slice contains non-finite values.

This guards against runs that return `:success` yet write NaN diagnostics, which
would otherwise be silently backfilled by plotting code to an earlier time slice.
"""
function _va_sgs_smoke_assert_finite_final_condensate!(output_dir::AbstractString)
    active = joinpath(output_dir, "output_active")
    clw_path = joinpath(active, "clw_10m_inst.nc")
    cli_path = joinpath(active, "cli_10m_inst.nc")
    if !(isfile(clw_path) && isfile(cli_path))
        throw(
            ErrorException(
                "Missing clw/cli diagnostics in output_active.\n" *
                "  clw: $clw_path\n" *
                "  cli: $cli_path",
            ),
        )
    end
    ds_clw = NCD.Dataset(clw_path, "r")
    ds_cli = NCD.Dataset(cli_path, "r")
    try
        clw = ds_clw["clw"][:, :, :, :]
        cli = ds_cli["cli"][:, :, :, :]
        nt = size(clw, 1)
        nt == 0 && throw(ErrorException("No time slices in clw/cli diagnostics."))
        s = clw[nt, :, :, :] .+ cli[nt, :, :, :]
        n_total = length(s)
        n_finite = count(isfinite, s)
        n_nan = count(isnan, s)
        n_inf = n_total - n_finite - n_nan
        n_finite == n_total && return nothing
        throw(
            ErrorException(
                "Non-finite final condensate profile in output_active (clw+cli at last time).\n" *
                "  total=$n_total finite=$n_finite nan=$n_nan inf=$n_inf",
            ),
        )
    finally
        close(ds_clw)
        close(ds_cli)
    end
end

function _va_apply_case_overrides!(
    cfg::Dict;
    external_forcing_file::Union{Nothing, AbstractString} = nothing,
    cfsite_number::Union{Nothing, AbstractString} = nothing,
)
    if external_forcing_file !== nothing
        cfg["external_forcing_file"] = String(external_forcing_file)
    else
        gcm_force = strip(get(ENV, "VA_GCM_FORCING_FILE", ""))
        if !isempty(gcm_force)
            cfg["external_forcing_file"] = gcm_force
        end
        gle_force = strip(get(ENV, "VA_GOOGLELES_FORCING_FILE", ""))
        if !isempty(gle_force)
            cfg["external_forcing_file"] = gle_force
        end
    end
    if cfsite_number !== nothing
        cfg["cfsite_number"] = String(cfsite_number)
    else
        gle_grp = strip(get(ENV, "VA_GOOGLELES_CFSITE_GROUP", ""))
        if !isempty(gle_grp)
            cfg["cfsite_number"] = gle_grp
        end
    end
    return cfg
end

"""If non-empty, append this under `.../forward_only/` so optional `dt`/`t_end` smokes do not clobber the default tree."""
function _va_sgs_smoke_output_sweep_id(
    dt_str_override::Union{Nothing, AbstractString},
    t_end_override::Union{Nothing, AbstractString},
)::Union{Nothing, String}
    if dt_str_override === nothing && t_end_override === nothing
        return nothing
    end
    parts = String[]
    if dt_str_override !== nothing
        push!(parts, "dt" * va_dt_path_slug(String(dt_str_override)))
    end
    if t_end_override !== nothing
        s = replace(lowercase(String(t_end_override)), r"[^0-9a-z]+" => "")
        !isempty(s) && push!(parts, "tend" * s)
    end
    return isempty(parts) ? nothing : join(parts, "_")
end

function va_trmm_sgs_smoke_resolve_grid(;
    experiment_dir::AbstractString = VA_TRMM_SGS_SMOKE_ROOT,
    case_layers::Vector{String} = VA_TRMM_SGS_SMOKE_CASE_LAYERS,
)
    base = va_load_merged_case_yaml_dict(experiment_dir, case_layers)
    slug = va_forward_sweep_case_slug(base)
    z_stretch = va_forward_sweep_case_z_stretch(base)
    yaml_dz = va_forward_sweep_case_dz_bottom(base)
    tier = only(va_resolution_tiers_for_forward(base, ForwardSweepConfig(; resolution_ladder = false)))
    res_seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
    return (; slug, tier, z_stretch, yaml_dz, res_seg)
end

function va_run_trmm_sgs_smoke_one(;
    experiment_dir::AbstractString = VA_TRMM_SGS_SMOKE_ROOT,
    case_layers::Vector{String} = VA_TRMM_SGS_SMOKE_CASE_LAYERS,
    scm_toml = VA_TRMM_SGS_SMOKE_SCM_TOML,
    n_quad::Int = 3,
    dist::AbstractString,
    tier::VAResolutionTier,
    z_stretch::Bool,
    yaml_dz::Float64,
    slug::AbstractString,
    skip_done::Bool = false,
    external_forcing_file::Union{Nothing, AbstractString} = nothing,
    cfsite_number::Union{Nothing, AbstractString} = nothing,
    dt_str_override::Union{Nothing, AbstractString} = nothing,
    t_end_override::Union{Nothing, AbstractString} = nothing,
)
    cfg = va_load_merged_case_yaml_dict(experiment_dir, case_layers)
    _va_apply_case_overrides!(cfg; external_forcing_file, cfsite_number)
    cfg["z_elem"] = tier.z_elem
    cfg["dt"] = dt_str_override === nothing ? tier.dt_str : String(dt_str_override)
    if t_end_override !== nothing
        cfg["t_end"] = String(t_end_override)
    end
    if tier.dz_bottom_written !== nothing
        cfg["dz_bottom"] = tier.dz_bottom_written
    end
    nip = get(cfg, "netcdf_interpolation_num_points", nothing)
    if nip isa AbstractVector && length(nip) >= 3
        cfg["netcdf_interpolation_num_points"] = Any[nip[1], nip[2], tier.z_elem]
    end
    dist_slug = va_sgs_dist_path_slug(dist)
    sweep = _va_sgs_smoke_output_sweep_id(dt_str_override, t_end_override)
    cfg["output_dir"] = let base = joinpath(
            experiment_dir,
            "simulation_output",
            "sgs_smoke",
            slug,
            va_tier_path_segment(tier, z_stretch, yaml_dz),
            "N_$(n_quad)",
            dist_slug,
            "forward_only",
        )
        sweep === nothing ? base : joinpath(base, sweep)
    end
    mkpath(cfg["output_dir"])
    active = joinpath(cfg["output_dir"], "output_active")
    if skip_done && isdir(active)
        @info "Skipping (existing output_active)" active dist
        return nothing
    end
    cfg["quadrature_order"] = n_quad
    cfg["sgs_distribution"] = String(dist)
    merged_path = joinpath(cfg["output_dir"], VA_MERGED_SCM_BASELINE_BASENAME)
    va_write_merged_scm_baseline_file!(experiment_dir, scm_toml, merged_path)
    cfg["toml"] = [merged_path]
    cfg["output_default_diagnostics"] = get(cfg, "output_default_diagnostics", false)
    job_id = string(
        "va_sgs_smoke_",
        slug,
        "_",
        dist_slug,
        sweep === nothing ? "" : string("_", sweep),
    )
    atmos_config = CA.AtmosConfig(cfg; comms_ctx = va_comms_ctx(), job_id)
    sim = CA.get_simulation(atmos_config)
    solver_log = joinpath(cfg["output_dir"], "sgs_smoke_solve.log")
    isfile(solver_log) && rm(solver_log)
    log_io = open(solver_log, "w")
    local sol_res
    try
        with_logger(
            _VA_TeeLogger(ConsoleLogger(stderr), SimpleLogger(log_io)),
        ) do
            sol_res = CA.solve_atmos!(sim)
        end
    finally
        flush(log_io)
        close(log_io)
    end
    # `solve_atmos!` catches integration failures and returns `:simulation_crashed` instead of throwing;
    # the ClimaAtmos catch block uses `@error` (see `src/solver/solve.jl`), which is duplicated to
    # `stderr` and `solver_log` via `_VA_TeeLogger` above.
    if sol_res.ret_code == :simulation_crashed
        od = String(sim.output_dir)
        exc = ErrorException(
            "solve_atmos! returned :simulation_crashed (time integration failed). " *
            "See the tail of the solver log appended below (same content as the `@error` in src/solver/solve.jl). " *
            "Partial state may exist under output_dir:\n  $od",
        )
        bt = backtrace()
        rseg = String(va_tier_path_segment(tier, z_stretch, yaml_dz))
        fail_path = _va_sgs_smoke_write_failure!(
            experiment_dir, slug, n_quad, dist, exc, bt;
            res_segment = rseg,
            solver_log_path = isfile(solver_log) ? String(solver_log) : nothing,
        )
        error(
            "ClimaAtmos integration failed for sgs_distribution=$(repr(dist)). " *
            "Smoke summary written to:\n  $fail_path",
        )
    end
    _va_sgs_smoke_assert_finite_final_condensate!(cfg["output_dir"])
    isfile(solver_log) && rm(solver_log; force = true)
    @info "Finished sgs_smoke run" dist output_dir = cfg["output_dir"]
    return sim
end

"""One pmap task: run one distribution or record failure on disk (worker-local backtrace)."""
function _va_run_trmm_sgs_smoke_job(job::NamedTuple)
    try
        va_run_trmm_sgs_smoke_one(;
            experiment_dir = job.experiment_dir,
            case_layers = job.case_layers,
            scm_toml = job.scm_toml,
            n_quad = job.n_quad,
            dist = job.dist,
            tier = job.tier,
            z_stretch = job.z_stretch,
            yaml_dz = job.yaml_dz,
            slug = job.slug,
            skip_done = job.skip_done,
            external_forcing_file = job.external_forcing_file,
            cfsite_number = job.cfsite_number,
        )
        return (; ok = true, dist = job.dist, log_path = nothing, summary = nothing)
    catch e
        bt = catch_backtrace()
        sol_log = _va_solver_log_path(;
            experiment_dir = job.experiment_dir,
            slug = job.slug,
            n_quad = job.n_quad,
            dist = job.dist,
            tier = job.tier,
            z_stretch = job.z_stretch,
            yaml_dz = job.yaml_dz,
        )
        rseg = String(va_tier_path_segment(job.tier, job.z_stretch, job.yaml_dz))
        path = _va_sgs_smoke_write_failure!(
            job.experiment_dir,
            job.slug,
            job.n_quad,
            job.dist,
            e,
            bt;
            res_segment = rseg,
            solver_log_path = isfile(sol_log) ? String(sol_log) : nothing,
        )
        return (; ok = false, dist = job.dist, log_path = path, summary = sprint(showerror, e))
    end
end
