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

include(joinpath(_VA_EXPERIMENT_DIR, "lib", "experiment_common.jl"))
import CairoMakie as M

# Default finer-mesh overlay (same intent as `plot_forward_sweep_body.jl`): black, drawn last.
const _VA_FINER_MESH_DEFAULT_COLOR = M.RGBf(0.0, 0.0, 0.0)

"""After `autolimits!`, optionally set `ylims` from `va_condensate_cloud_top_height_m` (cloud top + padding)."""
function _va_finalize_profile_axis_limits!(ax, ylims_height_max::Union{Nothing, Real})
    M.autolimits!(ax)
    if ylims_height_max === nothing
        return
    end
    # Profiles are drawn on the full native `z` column (zeros / noise aloft), so `autolimits` y-extent is
    # essentially the model top — **not** “where the cloud ends”. Do **not** do `max(cap, autolimits)`:
    # that would undo the cloud-top cap (e.g. TRMM ~4.3 km cap vs ~16 km column).
    M.ylims!(ax, 0.0, Float64(ylims_height_max))
    return
end
import ClimaAnalysis
import ClimaAnalysis: SimDir, get, slice, average_xy
import Statistics: mean

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

"""Horizontal mean at the **last time with finite data** (walks backward if the final instant is all-NaN in the file)."""
function _va_horiz_mean_last_slice(simdir, spec)
    v = get(
        simdir;
        short_name = spec.short_name,
        reduction = spec.reduction,
        period = spec.period,
    )
    tcoord = ClimaAnalysis.times(v)
    isempty(tcoord) && return nothing
    for ti in reverse(tcoord)
        s = slice(v, time = ti)
        prof = if haskey(s.dims, "x") && haskey(s.dims, "y")
            average_xy(s)
        else
            s
        end
        any(isfinite, prof.data) || continue
        if ti != tcoord[end]
            @debug "Profile slice: last time all-NaN; using earlier instant" spec.short_name spec.reduction spec.period time = ti t_last = tcoord[end]
        end
        return prof
    end
    return nothing
end

"""`(data, z)` as 1D vectors if `prof` has an altitude dimension; otherwise `nothing` (e.g. **`ts`** surface temperature)."""
function _va_vertical_profile_vectors(prof)
    ClimaAnalysis.has_altitude(prof) || return nothing
    zname = ClimaAnalysis.altitude_name(prof)
    return (vec(prof.data), vec(prof.dims[zname]))
end

function va_profile_last_time_spec(simdir, spec)
    prof = _va_horiz_mean_last_slice(simdir, spec)
    prof === nothing && error("no time instants in SimDir output for $(spec.short_name)")
    xz = _va_vertical_profile_vectors(prof)
    xz === nothing && error(
        "No altitude dimension after horizontal mean; dims=$(repr(collect(keys(prof.dims)))). " *
        "Surface-only fields (e.g. ts) are not height profiles.",
    )
    return xz
end

"""
    va_scalar_surface_mean_last(simdir, short_name, period_yaml) -> Union{Nothing, Float64}

Horizontal mean at the **last time** for a **surface / column-integrated** diagnostic (e.g. **`lwp`**, **`clivi`**).
Returns `nothing` if the variable is missing or resolves to a vertical profile (use profile plotting instead).
"""
function va_scalar_surface_mean_last(simdir, short_name::AbstractString, period_yaml::AbstractString)
    spec = _va_resolve_profile_spec_for_simdir(simdir, short_name, period_yaml)
    spec === nothing && return nothing
    v = get(simdir; short_name = spec.short_name, reduction = spec.reduction, period = spec.period)
    tcoord = ClimaAnalysis.times(v)
    isempty(tcoord) && return nothing
    s = slice(v, time = tcoord[end])
    ClimaAnalysis.has_altitude(s) && return nothing
    return Float64(mean(s.data))
end

"""`(x, z)` for plotting; trim to `z_cap` levels when `z_cap !== nothing` (coarse panel), else full native vertical grid (e.g. high-res reference)."""
function _va_vertical_profile_sized(simdir, spec, z_cap::Union{Nothing, Int})
    prof = _va_horiz_mean_last_slice(simdir, spec)
    prof === nothing && return nothing
    xz = _va_vertical_profile_vectors(prof)
    xz === nothing && return nothing
    x, z = xz
    n = z_cap === nothing ? length(x) : min(length(x), z_cap)
    return (x[1:n], z[1:n])
end

function _va_yaml_period_for_short_name(
    pairs::Vector{Tuple{String, String}},
    short_name::AbstractString,
)::Union{Nothing, String}
    for (sn, per) in pairs
        sn == short_name && return per
    end
    return nothing
end

"""Pointwise sum of two profile fields on the same vertical grid (e.g. `clw` + `cli`)."""
function _va_profile_pair_sum_xy(
    simdir,
    short_a::AbstractString,
    short_b::AbstractString,
    period_a::AbstractString,
    period_b::AbstractString,
    z_cap::Union{Nothing, Int},
)
    spec_a = _va_resolve_profile_spec_for_simdir(simdir, short_a, period_a)
    spec_b = _va_resolve_profile_spec_for_simdir(simdir, short_b, period_b)
    (spec_a === nothing || spec_b === nothing) && return nothing
    vz_a = _va_vertical_profile_sized(simdir, spec_a, z_cap)
    vz_b = _va_vertical_profile_sized(simdir, spec_b, z_cap)
    (vz_a === nothing || vz_b === nothing) && return nothing
    xa, za = vz_a
    xb, zb = vz_b
    n = min(length(xa), length(xb), length(za), length(zb))
    n == 0 && return nothing
    return (xa[1:n] .+ xb[1:n], za[1:n])
end

"""
    va_condensate_cloud_top_height_m(path_list, calibration_truth_series, forward_finest_series;
        experiment_dir, model_config_rel, q_threshold, padding_m) -> Union{Nothing, Float64}

Upper **height (m)** for profile **y-axis** limits: **`padding_m`** above the highest level where
**`clw + cli > q_threshold`** (kg/kg) on **any** sweep line or reference overlay. Used so all profile PNGs
in a forward-sweep panel share the same vertical extent.

Returns **`nothing`** if `clw`/`cli` are missing from the case YAML, or no level exceeds **`q_threshold`**
(then callers keep Makie autoscale). **`q_threshold`** should be **positive** so numerical wisp-negatives in
`clw`/`cli` are not treated as cloud.
"""
function va_condensate_cloud_top_height_m(
    path_list::AbstractVector{<:AbstractString},
    calibration_truth_series,
    forward_finest_series;
    experiment_dir::AbstractString,
    model_config_rel,
    q_threshold::Float64 = 1e-5,
    padding_m::Float64 = 600.0,
)::Union{Nothing, Float64}
    pairs = va_case_yaml_diagnostic_shortname_period_pairs(experiment_dir, model_config_rel)
    per_a = _va_yaml_period_for_short_name(pairs, "clw")
    per_b = _va_yaml_period_for_short_name(pairs, "cli")
    (per_a === nothing || per_b === nothing) && return nothing

    z_elem = va_z_elem_from_case_yaml(experiment_dir, model_config_rel)
    z_cloud = -Inf
    z_data_max = -Inf

    function accum!(simdir_path, z_cap::Union{Nothing, Int})
        !isdir(simdir_path) && return
        simdir = SimDir(simdir_path)
        isempty(simdir.vars) && return
        vz = _va_profile_pair_sum_xy(simdir, "clw", "cli", per_a, per_b, z_cap)
        vz === nothing && return
        x, z = vz
        for i in eachindex(x, z)
            zi = z[i]
            z_data_max = max(z_data_max, zi)
            if x[i] > q_threshold
                z_cloud = max(z_cloud, zi)
            end
        end
        return
    end

    for p in path_list
        accum!(String(p), z_elem)
    end
    if forward_finest_series !== nothing
        for (ref_path, _) in forward_finest_series
            accum!(String(ref_path), nothing)
        end
    end
    if calibration_truth_series !== nothing
        for (ref_path, _) in calibration_truth_series
            accum!(String(ref_path), nothing)
        end
    end

    # `min(z_hi, z_data_max)` must not use `z_data_max` from **truncated** sweep profiles only: sweep lines
    # are plotted with `z_cap = z_elem` (first `z_elem` levels), so `accum!(..., z_elem)` can report a
    # **lower** column top than the native SimDir grid. That made `z_hi` too small vs overlays drawn with
    # `z_cap = nothing` (full column), so `ylims!(0, z_hi)` could clip the overlay while sweep curves still
    # looked fine within the truncated z range.
    for p in path_list
        accum!(String(p), nothing)
    end

    !isfinite(z_cloud) && return nothing
    z_hi = z_cloud + padding_m
    isfinite(z_data_max) && (z_hi = min(z_hi, z_data_max))
    return z_hi
end

"""
    _va_plot_clw_plus_cli_series_on_axis!(ax, path_list, ...; short_a, short_b, per_a, per_b, z_elem, ...) -> (n_series, any_line)

Draw **`clw`+`cli`** sweep lines, optional observations-y, then finer-mesh overlay on **`ax`** (same order as
[`va_plot_profile_shortname_sum!`](@ref)). Returns **`(n_series, any_line)`** for legend / save decisions.
"""
function _va_plot_clw_plus_cli_series_on_axis!(
    ax,
    path_list::Vector{String},
    short_a::AbstractString,
    short_b::AbstractString,
    per_a::AbstractString,
    per_b::AbstractString,
    z_elem::Int,
    path_labels,
    path_colors,
    path_linestyles,
    path_linewidths,
    calibration_truth_series,
    ff_series,
    forward_finest_color,
    forward_finest_linewidth,
)::Tuple{Int, Bool}
    any_line = false
    n_series = 0
    for i in eachindex(path_list)
        path = path_list[i]
        !isdir(path) && continue
        simdir = SimDir(path)
        isempty(simdir.vars) && continue
        vz = _va_profile_pair_sum_xy(simdir, short_a, short_b, per_a, per_b, z_elem)
        vz === nothing && continue
        x, z = vz
        lab = _va_profile_path_label(path_list, path_labels, path)
        lk = (; label = lab)
        if path_colors !== nothing
            lk = (; lk..., color = path_colors[i])
        end
        if path_linestyles !== nothing
            lk = (; lk..., linestyle = path_linestyles[i])
        end
        if path_linewidths !== nothing
            lk = (; lk..., linewidth = path_linewidths[i])
        end
        M.lines!(ax, x, z; lk...)
        any_line = true
        n_series += 1
    end
    if calibration_truth_series !== nothing
        for (ref_path, ref_label) in calibration_truth_series
            !isdir(ref_path) && continue
            rdir = SimDir(ref_path)
            isempty(rdir.vars) && continue
            rv = _va_profile_pair_sum_xy(rdir, short_a, short_b, per_a, per_b, nothing)
            rv === nothing && continue
            rx, rz = rv
            M.lines!(
                ax,
                rx,
                rz;
                label = ref_label,
                color = :black,
                linewidth = 3.15,
                linestyle = :solid,
            )
            any_line = true
            n_series += 1
        end
    end
    ff_drew = false
    if ff_series !== nothing
        for (ref_path, ref_label) in ff_series
            !isdir(ref_path) && continue
            rdir = SimDir(ref_path)
            isempty(rdir.vars) && continue
            rv = _va_profile_pair_sum_xy(rdir, short_a, short_b, per_a, per_b, nothing)
            if rv === nothing
                @warn "Forward-sweep finer-mesh overlay: could not build clw+cli sum profile" ref_path ref_label short_a short_b
                continue
            end
            rx, rz = rv
            n_fin = count(isfinite, rx)
            n_fz = count(isfinite, rz)
            if n_fin < 2 || n_fz < 2
                @warn "Forward-sweep finer-mesh overlay: insufficient finite profile points" ref_path ref_label short_a short_b n_fin n_fz
                continue
            end
            M.lines!(
                ax,
                rx,
                rz;
                label = ref_label,
                color = forward_finest_color,
                linewidth = forward_finest_linewidth,
                linestyle = :solid,
            )
            any_line = true
            n_series += 1
            ff_drew = true
        end
        if !ff_drew
            @warn "Forward-sweep finer-mesh overlay: no clw+cli line was drawn (check SimDir + diagnostics)" ff_series
        end
    end
    return (n_series, any_line)
end

"""
    va_plot_profile_shortname_sum!(paths; ...) -> Union{Nothing, String}

One figure for the **sum of two 3D profile diagnostics** on the same height levels (e.g. **`clw` + `cli`**
as total cloud condensate mass fraction). ClimaAtmos does not expose a single native **`clw+cli`** field;
this is computed in post.

Writes `outdir/profile_<sum_short>.png`. Returns that path, or **`nothing`** if either short name is missing
from the case YAML or no line could be drawn.

Uses the same reference overlays as [`va_plot_all_case_diagnostic_profiles`](@ref).
"""
function va_plot_profile_shortname_sum!(
    paths::AbstractVector{<:AbstractString};
    experiment_dir::AbstractString,
    model_config_rel,
    outdir::AbstractString,
    path_labels = nothing,
    path_colors = nothing,
    path_linestyles = nothing,
    path_linewidths = nothing,
    short_a::AbstractString = "clw",
    short_b::AbstractString = "cli",
    sum_short::AbstractString = "clw_plus_cli",
    xlabel::AbstractString = "clw+cli (kg/kg)",
    calibration_truth_series = nothing,
    forward_finest_series = nothing,
    reference_series = nothing,
    forward_finest_color = _VA_FINER_MESH_DEFAULT_COLOR,
    forward_finest_linewidth = 2.65,
    profile_title::Union{Nothing, AbstractString} = nothing,
    show_axis_title::Bool = true,
    legend_position = :rb,
    ylims_height_max::Union{Nothing, Real} = nothing,
    legend_outside::Bool = false,
)::Union{Nothing, String}
    pairs = va_case_yaml_diagnostic_shortname_period_pairs(experiment_dir, model_config_rel)
    per_a = _va_yaml_period_for_short_name(pairs, short_a)
    per_b = _va_yaml_period_for_short_name(pairs, short_b)
    (per_a === nothing || per_b === nothing) && return nothing

    path_list = collect(String, paths)
    n_p = length(path_list)
    if path_labels !== nothing && length(path_labels) != n_p
        error("path_labels length $(length(path_labels)) must match paths length $(n_p)")
    end
    if path_colors !== nothing && length(path_colors) != n_p
        error("path_colors length must match paths")
    end
    if path_linestyles !== nothing && length(path_linestyles) != n_p
        error("path_linestyles length must match paths")
    end
    if path_linewidths !== nothing && length(path_linewidths) != n_p
        error("path_linewidths length must match paths")
    end

    z_elem = va_z_elem_from_case_yaml(experiment_dir, model_config_rel)
    ff_series = forward_finest_series
    if ff_series === nothing && reference_series !== nothing
        ff_series = reference_series
    end

    safe = replace(sum_short, r"[^\w\.\-]+" => "_")
    outpng = joinpath(outdir, "profile_$(safe).png")
    mkpath(outdir)

    fig = M.Figure(size = (legend_outside ? 660 : 520, 680))
    sum_default_title = "Last time slice, horizontal mean — $sum_short = $short_a + $short_b"
    ax = M.Axis(
        fig[1, 1];
        xlabel = xlabel,
        ylabel = "height (m)",
        title = show_axis_title ? something(profile_title, sum_default_title) : "",
        titlevisible = show_axis_title,
    )
    n_series, any_line = _va_plot_clw_plus_cli_series_on_axis!(
        ax,
        path_list,
        short_a,
        short_b,
        per_a,
        per_b,
        z_elem,
        path_labels,
        path_colors,
        path_linestyles,
        path_linewidths,
        calibration_truth_series,
        ff_series,
        forward_finest_color,
        forward_finest_linewidth,
    )

    if !any_line
        @debug "va_plot_profile_shortname_sum!: no drawable lines" sum_short short_a short_b outdir
        return nothing
    end
    _va_finalize_profile_axis_limits!(ax, ylims_height_max)
    if legend_outside && n_series > 1
        M.Legend(fig[1, 2], ax; framevisible = true, backgroundcolor = :white)
        M.colgap!(fig.layout, 14)
        M.colsize!(fig.layout, 1, M.Auto(1))
        M.colsize!(fig.layout, 2, M.Auto(0.28))
    elseif n_series > 1
        M.axislegend(ax; position = legend_position, framevisible = true, backgroundcolor = :white)
    end
    M.save(outpng, fig)
    @info "Wrote summed profile figure" outpng sum_short = sum_short
    return outpng
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
            period = _va_resolve_simdir_period_plot(
                simdir,
                spec["short_name"],
                spec["reduction"],
                spec["period"],
            )
            sp = (; short_name = spec["short_name"], reduction = spec["reduction"], period)
            vz = _va_vertical_profile_sized(simdir, sp, z_elem)
            if vz !== nothing
                x, z = vz
                M.lines!(ax, x, z)
            end
        end
    end
    M.save(outpng, fig)
    @info "Saved (EKI observation stack only)" outpng paths = path_list n_fields
    return outpng
end

function _va_profile_path_label(path_list, path_labels, path::AbstractString)
    path_labels === nothing && return va_run_label_for_output_active_path(path)
    i = findfirst(==(path), path_list)
    i === nothing && return va_run_label_for_output_active_path(path)
    return String(path_labels[i])
end

"""
    va_plot_all_case_diagnostic_profiles([paths]; experiment_dir, outdir, path_labels, model_config_rel) -> Vector{String}

For each `(short_name, period)` in the case YAML `diagnostics` section, write
`outdir/profile_<short_name>.png` with **one axis**: quantity vs height, **one line per** `output_active`
path. Legend defaults to [`va_run_label_for_output_active_path`](@ref); pass **`path_labels`** (same
length as **`paths`**) for custom labels (e.g. forward-sweep `N=3 vf_on`).

Optional **`path_colors`**, **`path_linestyles`**, **`path_linewidths`**: same length as **`paths`**
(Makie attributes).

**Reference overlays** (after sweep lines): **`calibration_truth_series`** first (**black**, “Observations y”), then
**`forward_finest_series`** last so the finer mesh sits **on top**. Finer mesh defaults to **black** and a thick
**`forward_finest_linewidth`**; override with **`forward_finest_color`** / **`forward_finest_linewidth`**.
- **`calibration_truth_series`**: `(path, label)` for the **reference SCM column** used to build **`observations.jld2`**
  (same `output_active`); label e.g. **“Observations y (ref column)”**.
- **`forward_finest_series`**: `(output_active_path, label)` for a **finer vertical mesh** forward (e.g. `N=3` **vf_off**);
  full native vertical resolution.

Deprecated alias: **`reference_series`** maps to **`forward_finest_series`** when **`forward_finest_series`**
and **`calibration_truth_series`** are both `nothing` (older scripts).

If **`model_config_rel`** is set, diagnostics and `z_elem` are read from that column YAML instead of
**`experiment_config`** + **`model_config_path`**.

**`show_axis_title`** (default `true`): set `false` for publication-style figures where the file path / axis
label identify the panel; **`legend_position`** (default `:rb`) can be set e.g. to `:lt` when the legend
would cover curves.

**`legend_outside`** (default `false`): when `true` and there are multiple series, place a **`Legend`** in a
second column beside the axis so it does not overlap curves.

Returns paths of PNGs written.
"""
function va_plot_all_case_diagnostic_profiles(
    paths::AbstractVector{<:AbstractString} = String[];
    experiment_dir::AbstractString = _VA_EXPERIMENT_DIR,
    experiment_config::Union{Nothing, AbstractString} = nothing,
    outdir::AbstractString = joinpath(_VA_FIGURES_DIR, "profiles"),
    path_labels::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
    model_config_rel = nothing,
    path_colors = nothing,
    path_linestyles = nothing,
    path_linewidths = nothing,
    calibration_truth_series = nothing,
    forward_finest_series = nothing,
    reference_series = nothing,
    forward_finest_color = _VA_FINER_MESH_DEFAULT_COLOR,
    forward_finest_linewidth = 2.65,
    profile_title::Union{Nothing, AbstractString} = nothing,
    show_axis_title::Bool = true,
    legend_position = :rb,
    ylims_height_max::Union{Nothing, Real} = nothing,
    legend_outside::Bool = false,
)
    mkpath(outdir)
    path_list = if isempty(paths)
        model_config_rel !== nothing &&
            error("Non-empty `paths` required when using `model_config_rel` (no default reference).")
        expc = va_load_experiment_config(experiment_dir, experiment_config)
        String[va_reference_output_active(experiment_dir, experiment_config)]
    else
        collect(String, paths)
    end
    if path_labels !== nothing && length(path_labels) != length(path_list)
        error("path_labels length $(length(path_labels)) must match paths length $(length(path_list))")
    end
    n_p = length(path_list)
    if path_colors !== nothing && length(path_colors) != n_p
        error("path_colors length must match paths")
    end
    if path_linestyles !== nothing && length(path_linestyles) != n_p
        error("path_linestyles length must match paths")
    end
    if path_linewidths !== nothing && length(path_linewidths) != n_p
        error("path_linewidths length must match paths")
    end
    ff_series = forward_finest_series
    if ff_series === nothing && reference_series !== nothing
        ff_series = reference_series
    end
    pairs, z_elem = if model_config_rel === nothing
        expc = va_load_experiment_config(experiment_dir, experiment_config)
        z = va_z_elem(experiment_dir, expc)
        va_model_diagnostic_shortname_period_pairs(experiment_dir, experiment_config), z
    else
        va_case_yaml_diagnostic_shortname_period_pairs(experiment_dir, model_config_rel),
        va_z_elem_from_case_yaml(experiment_dir, model_config_rel)
    end
    isempty(pairs) && (@warn "No diagnostics in model YAML"; return String[])

    written = String[]
    for (short_name, period_yaml) in pairs
        safe = replace(short_name, r"[^\w\.\-]+" => "_")
        outpng = joinpath(outdir, "profile_$(safe).png")
        fig = M.Figure(size = (legend_outside ? 660 : 520, 680))
        ax = M.Axis(
            fig[1, 1];
            xlabel = short_name,
            ylabel = "height (m)",
            title = show_axis_title ? something(
                profile_title,
                "Last time slice, horizontal mean",
            ) : "",
            titlevisible = show_axis_title,
        )
        any_line = false
        n_series = 0
        spec_resolved_anywhere = false
        for i in eachindex(path_list)
            path = path_list[i]
            !isdir(path) && continue
            simdir = SimDir(path)
            isempty(simdir.vars) && continue
            spec = _va_resolve_profile_spec_for_simdir(simdir, short_name, period_yaml)
            spec === nothing && continue
            spec_resolved_anywhere = true
            vz = _va_vertical_profile_sized(simdir, spec, z_elem)
            vz === nothing && continue
            x, z = vz
            lab = _va_profile_path_label(path_list, path_labels, path)
            lk = (; label = lab)
            if path_colors !== nothing
                lk = (; lk..., color = path_colors[i])
            end
            if path_linestyles !== nothing
                lk = (; lk..., linestyle = path_linestyles[i])
            end
            if path_linewidths !== nothing
                lk = (; lk..., linewidth = path_linewidths[i])
            end
            M.lines!(ax, x, z; lk...)
            any_line = true
            n_series += 1
        end
        if calibration_truth_series !== nothing
            for (ref_path, ref_label) in calibration_truth_series
                !isdir(ref_path) && continue
                rdir = SimDir(ref_path)
                isempty(rdir.vars) && continue
                rspec = _va_resolve_profile_spec_for_simdir(rdir, short_name, period_yaml)
                rspec === nothing && continue
                rv = _va_vertical_profile_sized(rdir, rspec, nothing)
                rv === nothing && continue
                rx, rz = rv
                M.lines!(
                    ax,
                    rx,
                    rz;
                    label = ref_label,
                    color = :black,
                    linewidth = 3.15,
                    linestyle = :solid,
                )
                any_line = true
                n_series += 1
            end
        end
        if ff_series !== nothing
            for (ref_path, ref_label) in ff_series
                !isdir(ref_path) && continue
                rdir = SimDir(ref_path)
                isempty(rdir.vars) && continue
                rspec = _va_resolve_profile_spec_for_simdir(rdir, short_name, period_yaml)
                if rspec === nothing
                    @warn "Finer-mesh overlay: could not resolve diagnostic spec" ref_path ref_label short_name period_yaml
                    continue
                end
                rv = _va_vertical_profile_sized(rdir, rspec, nothing)
                if rv === nothing
                    @warn "Finer-mesh overlay: empty or non-altitude profile" ref_path ref_label short_name
                    continue
                end
                rx, rz = rv
                n_fin = count(isfinite, rx)
                n_fz = count(isfinite, rz)
                if n_fin < 2 || n_fz < 2
                    @warn "Finer-mesh overlay: insufficient finite profile points" ref_path ref_label short_name n_fin n_fz
                    continue
                end
                M.lines!(
                    ax,
                    rx,
                    rz;
                    label = ref_label,
                    color = forward_finest_color,
                    linewidth = forward_finest_linewidth,
                    linestyle = :solid,
                )
                any_line = true
                n_series += 1
            end
        end
        if any_line
            _va_finalize_profile_axis_limits!(ax, ylims_height_max)
            if legend_outside && n_series > 1
                M.Legend(fig[1, 2], ax; framevisible = true, backgroundcolor = :white)
                M.colgap!(fig.layout, 14)
                M.colsize!(fig.layout, 1, M.Auto(1))
                M.colsize!(fig.layout, 2, M.Auto(0.28))
            elseif n_series > 1
                M.axislegend(ax; position = legend_position, framevisible = true, backgroundcolor = :white)
            end
            M.save(outpng, fig)
            push!(written, outpng)
        elseif !spec_resolved_anywhere
            @warn "No SimDir data for diagnostic" short_name period_yaml
        else
            @debug "Diagnostic is not a height profile in output (e.g. surface `ts`)" short_name period_yaml
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
