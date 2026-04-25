# Shared TRMM SGS smoke definitions: loaded on the manager (via `trmm_sgs_smoke.jl`) and on
# `Distributed` workers for `parallel=:distributed`.

if abspath(PROGRAM_FILE) == @__FILE__
    error("Load scripts/trmm_sgs_smoke.jl instead of this core file.")
end

const VA_TRMM_SGS_SMOKE_ROOT = joinpath(@__DIR__, "..") |> abspath

include(joinpath(VA_TRMM_SGS_SMOKE_ROOT, "analysis", "plotting", "plot_profiles.jl"))
include(joinpath(VA_TRMM_SGS_SMOKE_ROOT, "scripts", "resolution_ladder.jl"))
include(joinpath(VA_TRMM_SGS_SMOKE_ROOT, "lib", "forward_sweep_grid.jl"))

import CairoMakie as CM
import ClimaAtmos as CA
import NCDatasets as NCD

const VA_TRMM_SGS_SMOKE_CASE_LAYERS = String[
    "model_configs/master_column_varquad_diagnostic_edmfx.yml",
    "model_configs/trmm_column_varquad_hires.yml",
]

const VA_TRMM_SGS_SMOKE_SCM_TOML = Any["diagnostic_edmfx_1M.toml", "toml/uncalibrated_stability_overlay.toml"]

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

Filename is stable per `(case_slug, quadrature_order, sgs_distribution)` and **overwrites** on a later failure
for the same triple so `_failures/` does not accumulate timestamped copies.
"""
function _va_sgs_smoke_write_failure!(
    experiment_dir::AbstractString,
    slug::AbstractString,
    n_quad::Int,
    dist::AbstractString,
    err,
    bt;
    max_backtrace_lines::Int = 48,
)::String
    fail_root = joinpath(experiment_dir, "simulation_output", "sgs_smoke", "_failures")
    mkpath(fail_root)
    safe_dist = va_sgs_dist_path_slug(dist)
    path = joinpath(fail_root, "$(slug)_N$(n_quad)_$(safe_dist).txt")
    open(path, "w") do io
        println(io, "sgs_distribution = ", dist)
        println(io, "case_slug = ", slug, "  quadrature_order = ", n_quad)
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
    end
    return path
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
)
    cfg = va_load_merged_case_yaml_dict(experiment_dir, case_layers)
    _va_apply_case_overrides!(cfg; external_forcing_file, cfsite_number)
    cfg["z_elem"] = tier.z_elem
    cfg["dt"] = tier.dt_str
    if tier.dz_bottom_written !== nothing
        cfg["dz_bottom"] = tier.dz_bottom_written
    end
    nip = get(cfg, "netcdf_interpolation_num_points", nothing)
    if nip isa AbstractVector && length(nip) >= 3
        cfg["netcdf_interpolation_num_points"] = Any[nip[1], nip[2], tier.z_elem]
    end
    dist_slug = va_sgs_dist_path_slug(dist)
    cfg["output_dir"] = joinpath(
        experiment_dir,
        "simulation_output",
        "sgs_smoke",
        slug,
        va_tier_path_segment(tier, z_stretch, yaml_dz),
        "N_$(n_quad)",
        dist_slug,
        "forward_only",
    )
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
    job_id = string("va_sgs_smoke_", slug, "_", dist_slug)
    atmos_config = CA.AtmosConfig(cfg; comms_ctx = va_comms_ctx(), job_id)
    sim = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(sim)
    # `solve_atmos!` catches timestep failures and returns `:simulation_crashed` instead of throwing,
    # so without this check the smoke script would log "Finished" and leave no `output_active` with no `_failures/` entry.
    if sol_res.ret_code == :simulation_crashed
        od = String(sim.output_dir)
        exc = ErrorException(
            "solve_atmos! returned :simulation_crashed (time integration failed). " *
            "ClimaAtmos logs the underlying exception and backtrace via `@error` in this process's output (capture stderr to keep it). " *
            "Partial state may exist under output_dir:\n  $od",
        )
        bt = backtrace()
        fail_path = _va_sgs_smoke_write_failure!(experiment_dir, slug, n_quad, dist, exc, bt)
        error(
            "ClimaAtmos integration failed for sgs_distribution=$(repr(dist)). " *
            "Smoke summary written to:\n  $fail_path",
        )
    end
    _va_sgs_smoke_assert_finite_final_condensate!(cfg["output_dir"])
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
        path = _va_sgs_smoke_write_failure!(
            job.experiment_dir,
            job.slug,
            job.n_quad,
            job.dist,
            e,
            bt,
        )
        return (; ok = false, dist = job.dist, log_path = path, summary = sprint(showerror, e))
    end
end
