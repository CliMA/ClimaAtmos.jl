# Vertical profiles from `output_active` (reference / ensemble members).
#
# **REPL** (activate this experiment’s project first):
#   include("analysis/plotting/plot_profiles.jl")
#   va_plot_profiles()  # only `observation_fields` (default: thetaa) — EKI observation stack
#   va_plot_all_case_diagnostic_profiles()  # one PNG per case-YAML diagnostic (`analysis/figures/profiles/profile_<name>.png`)
#   va_run_post_analysis!()  # see `run_post_analysis.jl`
#
# **CLI**: julia --project=. analysis/plotting/plot_profiles.jl [output_active paths...]
#
_va_analysis_is_main() =
    !isempty(Base.PROGRAM_FILE) && abspath(Base.PROGRAM_FILE) == abspath(@__FILE__)

const _VA_PLOTTING_DIR = @__DIR__
const _VA_ANALYSIS_DIR = joinpath(_VA_PLOTTING_DIR, "..") |> abspath
const _VA_EXPERIMENT_DIR = joinpath(_VA_ANALYSIS_DIR, "..") |> abspath
const _VA_FIGURES_DIR = joinpath(_VA_ANALYSIS_DIR, "figures")

if _va_analysis_is_main()
    import Pkg
    Pkg.activate(_VA_EXPERIMENT_DIR)
end

include(joinpath(_VA_EXPERIMENT_DIR, "experiment_common.jl"))
import CairoMakie as M
import ClimaAnalysis
import ClimaAnalysis: SimDir, get, slice, average_xy

"""Legend / title label for a run: parent of `output_active` (e.g. `reference`, `member_003`), not the word `output_active`."""
function va_run_label_for_output_active_path(path::AbstractString)
    p = abspath(path)
    if basename(p) == "output_active"
        lab = basename(dirname(p))
        return isempty(lab) ? p : lab
    end
    return basename(p)
end

# Same logic as `observation_map.jl` `_resolve_simdir_period` (`experiment_common` avoids ClimaAnalysis for ClimaAtmos tests).
function _va_resolve_simdir_period_plot(simdir, short_name, reduction, requested::AbstractString)
    avail = collect(ClimaAnalysis.available_periods(simdir; short_name, reduction))
    requested in avail && return requested
    length(avail) == 1 && return first(avail)
    error(
        "Period $(repr(requested)) not available for $(repr(short_name)) and reduction $(repr(reduction)). " *
        "Available: $(repr(avail)).",
    )
end

# Try reductions / period aliases until a variable can be read from `SimDir`.
function _va_resolve_profile_spec_for_simdir(
    simdir,
    short_name::AbstractString,
    period_yaml::AbstractString,
)
    period_candidates = if period_yaml == "10mins"
        ("10m", "10mins", "10min")
    else
        (period_yaml, "10m", "10mins")
    end
    red_list = try
        collect(ClimaAnalysis.available_reductions(simdir; short_name))
    catch
        return nothing
    end
    isempty(red_list) && return nothing
    for reduction in red_list
        avail = try
            collect(ClimaAnalysis.available_periods(simdir; short_name, reduction))
        catch
            continue
        end
        isempty(avail) && continue
        for per in period_candidates
            per in avail && return (; short_name, reduction, period = per)
        end
        length(avail) == 1 && return (; short_name, reduction, period = first(avail))
    end
    return nothing
end

function va_profile_last_time_spec(simdir, spec)
    v = get(
        simdir;
        short_name = spec.short_name,
        reduction = spec.reduction,
        period = spec.period,
    )
    t_last = ClimaAnalysis.times(v)[end]
    s = slice(v, time = t_last)
    prof = average_xy(s)
    zname = ClimaAnalysis.altitude_name(prof)
    haskey(prof.dims, zname) ||
        error("No vertical dim $(repr(zname)); dims=$(repr(collect(keys(prof.dims)))))")
    z = vec(prof.dims[zname])
    return vec(prof.data), z
end

function _va_profile_last_time_obs_dict(simdir, spec::Dict)
    period = _va_resolve_simdir_period_plot(
        simdir,
        spec["short_name"],
        spec["reduction"],
        spec["period"],
    )
    return va_profile_last_time_spec(
        simdir,
        (; short_name = spec["short_name"], reduction = spec["reduction"], period),
    )
end

"""
    va_plot_profiles([paths]; experiment_dir, outpng) -> outpng

**EKI observation stack only** — variables in `observation_fields` / default **`thetaa`**. This is what goes
into **y** for calibration; it is **not** the full case diagnostic list (see
[`va_plot_all_case_diagnostic_profiles`](@ref) for `profiles/profile_*.png`).

Figure size is tuned for a **tall** z axis (meteorology-style x = quantity, y = height).
"""
function va_plot_profiles(
    paths::AbstractVector{<:AbstractString} = String[];
    experiment_dir::AbstractString = _VA_EXPERIMENT_DIR,
    experiment_config::Union{Nothing, AbstractString} = nothing,
    outpng::AbstractString = joinpath(_VA_FIGURES_DIR, "profiles_eki_observation_stack.png"),
)
    mkpath(dirname(outpng))
    expc = va_load_experiment_config(experiment_dir, experiment_config)
    field_specs = va_field_specs(expc)
    z_elem = va_z_elem(experiment_dir, expc)

    path_list = if isempty(paths)
        String[va_reference_output_active(experiment_dir, experiment_config)]
    else
        collect(String, paths)
    end

    n_fields = length(field_specs)
    n_paths = length(path_list)
    # Wide enough per column; one row per run so height scales with number of paths
    col_w = max(200, min(280, 160 + 45 * n_fields))
    fig_w = 80 + col_w * n_fields
    fig_h = max(420, 100 + 520 * n_paths)
    fig = M.Figure(size = (fig_w, fig_h))
    for (pi, path) in enumerate(path_list)
        !isdir(path) && (@warn "Skip missing path" path; continue)
        simdir = SimDir(path)
        isempty(simdir.vars) && (@warn "No vars" path; continue)
        for (fi, spec) in enumerate(field_specs)
            ax = M.Axis(
                fig[pi, fi];
                xlabel = spec["short_name"],
                ylabel = "z (m)",
                title = fi == 1 ? va_run_label_for_output_active_path(path) : "",
                yticksvisible = fi == 1,
                ylabelvisible = fi == 1,
            )
            try
                x, z = _va_profile_last_time_obs_dict(simdir, spec)
                n = min(length(x), z_elem)
                M.lines!(ax, x[1:n], z[1:n])
            catch e
                @warn "Skip field for path" spec["short_name"] path exception = e
            end
        end
    end
    M.save(outpng, fig)
    @info "Saved (EKI observation stack only)" outpng paths = path_list n_fields
    return outpng
end

"""
    va_plot_all_case_diagnostic_profiles([paths]; experiment_dir, outdir) -> Vector{String}

For each `(short_name, period)` in the case YAML `diagnostics` section, write
`outdir/profile_<short_name>.png` with **one axis**: quantity vs height, **one line per** `output_active`
path (legend = basename of path). Skips variables missing from a run with a warning.

Returns paths of PNGs written.
"""
function va_plot_all_case_diagnostic_profiles(
    paths::AbstractVector{<:AbstractString} = String[];
    experiment_dir::AbstractString = _VA_EXPERIMENT_DIR,
    experiment_config::Union{Nothing, AbstractString} = nothing,
    outdir::AbstractString = joinpath(_VA_FIGURES_DIR, "profiles"),
)
    mkpath(outdir)
    expc = va_load_experiment_config(experiment_dir, experiment_config)
    z_elem = va_z_elem(experiment_dir, expc)
    path_list = if isempty(paths)
        String[va_reference_output_active(experiment_dir, experiment_config)]
    else
        collect(String, paths)
    end
    pairs = va_model_diagnostic_shortname_period_pairs(experiment_dir, experiment_config)
    isempty(pairs) && (@warn "No diagnostics in model YAML"; return String[])

    written = String[]
    for (short_name, period_yaml) in pairs
        safe = replace(short_name, r"[^\w\.\-]+" => "_")
        outpng = joinpath(outdir, "profile_$(safe).png")
        fig = M.Figure(size = (520, 680))
        ax = M.Axis(
            fig[1, 1];
            xlabel = short_name,
            ylabel = "height (m)",
            title = "Last time slice, horizontal mean",
        )
        any_line = false
        n_series = 0
        for path in path_list
            !isdir(path) && continue
            simdir = SimDir(path)
            isempty(simdir.vars) && continue
            spec = _va_resolve_profile_spec_for_simdir(simdir, short_name, period_yaml)
            spec === nothing && continue
            try
                x, z = va_profile_last_time_spec(simdir, spec)
                n = min(length(x), z_elem)
                M.lines!(
                    ax,
                    x[1:n],
                    z[1:n];
                    label = va_run_label_for_output_active_path(path),
                )
                any_line = true
                n_series += 1
            catch e
                @warn "Could not plot" short_name path exception = e
            end
        end
        if any_line
            n_series > 1 && M.axislegend(ax; position = :rb)
            M.save(outpng, fig)
            push!(written, outpng)
        else
            @warn "No SimDir data for diagnostic" short_name period_yaml
        end
    end
    @info "Wrote case diagnostic profile figures" outdir n = length(written)
    return written
end

function _va_plot_profiles_cli()
    paths = length(ARGS) >= 1 ? collect(String, ARGS) : String[]
    va_plot_profiles(paths; experiment_dir = _VA_EXPERIMENT_DIR)
end

if _va_analysis_is_main()
    _va_plot_profiles_cli()
end
