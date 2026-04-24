# Assumes `forward_sweep_grid.jl`, `plot_profiles.jl` (`va_plot_all_case_diagnostic_profiles`,
# `va_scalar_surface_mean_last`), `resolution_ladder.jl`, and `experiment_common.jl` are already loaded.
import CairoMakie as CM
import ClimaAnalysis: SimDir

# Distinct hues per N_quad (cycles if N > length). N=4 vs N=5 use purple vs cyan so they do not read as one brown family.
const _VA_NQUAD_LINE_COLORS = CM.RGBf[
    CM.RGBf(0.121569, 0.466667, 0.705882),  # blue
    CM.RGBf(0.172549, 0.627451, 0.172549),  # green
    CM.RGBf(0.839216, 0.152941, 0.156863),  # red
    CM.RGBf(0.580392, 0.403922, 0.741176),  # purple (N=4)
    CM.RGBf(0.090196, 0.745098, 0.811765),  # cyan (N=5; was brown — too close to purple at a glance)
    CM.RGBf(0.890196, 0.466667, 0.760784),  # pink
    CM.RGBf(0.498039, 0.498039, 0.498039),  # gray
    CM.RGBf(0.737255, 0.741176, 0.133333),  # olive
    CM.RGBf(0.549020, 0.337255, 0.294118),  # brown
    CM.RGBf(0.631373, 0.854902, 0.223529),  # lime
]

"""Distinct colors per quadrature order **N** (cycles if N > length of palette)."""
function _va_forward_sweep_nquad_colors(nmax::Int)
    n = max(nmax, 1)
    return [_VA_NQUAD_LINE_COLORS[mod1(i, length(_VA_NQUAD_LINE_COLORS))] for i in 1:n]
end

"""Finer-mesh overlay: black, drawn last so it sits above sweep lines (and above observations-y when both exist)."""
const _VA_FORWARD_FINER_MESH_COLOR = CM.RGBf(0.0, 0.0, 0.0)

# Match TRMM smoke / profile plots: relative floor scales τ with case amplitude; padding above cloud top.
const _VA_FORWARD_SWEEP_CLOUD_TOP_KW = (
    padding_m = 380.0,
    condensate_floor_kg_kg = 1e-12,
    condensate_floor_frac_of_peak = 1e-4,
)

"""
    va_plot_forward_sweep_comparisons!(experiment_dir, cfg::ForwardSweepConfig; figure_root) -> Vector{String}

For each `(case_slug, res_segment)` in the sweep grid, collect existing `output_active` paths for all
`(N_quad, varfix)` and write **`profile_<diagnostic>.png`** under
`figure_root/<slug>/<res_segment>/profiles/`, one line per cell with labels `N=k vf_on|vf_off`.

`figure_root` defaults to `analysis/figures/forward_sweep_eki_calibrated` or `.../forward_sweep_baseline_scm`
to match [`va_forward_sweep_forward_subdir`](@ref) (`forward_eki` vs `forward_only`).
"""
function va_plot_forward_sweep_comparisons!(
    experiment_dir::AbstractString,
    cfg::ForwardSweepConfig;
    figure_root::Union{Nothing, AbstractString} = nothing,
)
    root = something(
        figure_root,
        joinpath(
            experiment_dir,
            "analysis",
            "figures",
            cfg.forward_parameters == VA_FORWARD_PARAM_BASELINE_SCM ? "forward_sweep_baseline_scm" :
            "forward_sweep_eki_calibrated",
        ),
    )
    tasks = va_flatten_forward_sweep_tasks(experiment_dir, cfg)
    groups = Dict{Tuple{String, String, String}, Vector{Tuple{Int, Bool, String, String, Union{Nothing, String}}}}()
    for (layers, _scm, slug, n, vf, tier, z_stretch, yaml_dz, _eo, _en, vfon) in tasks
        yml = layers[end]
        res_seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
        ap = va_forward_sweep_output_active_path(
            experiment_dir,
            slug,
            res_seg,
            n,
            vf,
            cfg;
            forced_varfix_on_distribution = vfon,
        )
        !isdir(ap) && continue
        k = (slug, yml, res_seg)
        lab = if !vf
            "N=$(n) vf_off"
        elseif vfon === nothing
            "N=$(n) vf_on (default)"
        else
            string("N=$(n) vf_on ", va_sgs_dist_path_slug(vfon))
        end
        push!(get!(groups, k, Tuple{Int, Bool, String, String, Union{Nothing, String}}[]), (n, vf, ap, lab, vfon))
    end
    all_pngs = String[]
    for ((slug, yml, res_seg), cells) in groups
        # N_quad ascending, then varfix off before on, then explicit varfix-on distribution (if any)
        sort!(cells; by = x -> (x[1], x[2] ? 1 : 0, something(x[5], "")))
        paths = String[c[3] for c in cells]
        paths_abs = String[abspath(p) for p in paths]
        labels = String[c[4] for c in cells]
        nmax = maximum(c[1] for c in cells)
        pal = _va_forward_sweep_nquad_colors(nmax)
        path_colors = [pal[c[1]] for c in cells]
        path_linestyles = [c[2] ? :solid : :dash for c in cells]
        path_linewidths = fill(1.85, length(cells))
        forward_finest_series = nothing
        calibration_truth_series = nothing
        row = va_forward_sweep_registry_row_for(experiment_dir, slug, yml, cfg)
        cfg_layers =
            row !== nothing ? row.model_config_layers : String[yml]
        if row !== nothing && row.eki_varfix_off_config !== nothing
            truth_active = va_reference_output_active(experiment_dir, row.eki_varfix_off_config)
            if isdir(truth_active)
                calibration_truth_series = Tuple{String, String}[(
                    truth_active,
                    "Observations y (ref column)",
                )]
            end
        end
        n_ref = 3
        vf_ref = false
        fz = va_forward_sweep_reference_finer_than_panel(
            experiment_dir,
            slug,
            yml,
            cfg,
            res_seg;
            n_quad = n_ref,
            varfix = vf_ref,
        )
        if fz !== nothing
            _z_ref_elem, ref_active, ref_tier_seg = fz
            # Compare canonical paths so symlink / relative vs absolute does not duplicate or drop the overlay.
            if abspath(ref_active) ∉ paths_abs
                forward_finest_series = Tuple{String, String}[(
                    ref_active,
                    "Finer mesh (N=$(n_ref) $(vf_ref ? "vf_on" : "vf_off")): $(ref_tier_seg)",
                )]
            end
        end
        outdir = joinpath(root, slug, res_seg, "profiles")
        mkpath(outdir)
        ylims_h = va_condensate_cloud_top_height_m(
            paths,
            calibration_truth_series,
            forward_finest_series;
            experiment_dir,
            model_config_rel = cfg_layers,
            _VA_FORWARD_SWEEP_CLOUD_TOP_KW...,
        )
        written = va_plot_all_case_diagnostic_profiles(
            paths;
            experiment_dir,
            experiment_config = nothing,
            outdir,
            path_labels = labels,
            model_config_rel = cfg_layers,
            path_colors,
            path_linestyles,
            path_linewidths,
            calibration_truth_series,
            forward_finest_series,
            forward_finest_color = _VA_FORWARD_FINER_MESH_COLOR,
            forward_finest_linewidth = 4.35,
            show_axis_title = false,
            legend_position = :lt,
            ylims_height_max = ylims_h,
            legend_outside = true,
        )
        append!(all_pngs, written)
        sum_png = va_plot_profile_shortname_sum!(
            paths;
            experiment_dir,
            model_config_rel = cfg_layers,
            outdir,
            path_labels = labels,
            path_colors,
            path_linestyles,
            path_linewidths,
            calibration_truth_series,
            forward_finest_series,
            forward_finest_color = _VA_FORWARD_FINER_MESH_COLOR,
            forward_finest_linewidth = 4.35,
            show_axis_title = false,
            legend_position = :lt,
            ylims_height_max = ylims_h,
            legend_outside = true,
        )
        sum_png !== nothing && push!(all_pngs, sum_png)
        @info "Forward sweep comparison profiles" slug res_seg n_series = length(paths) outdir
    end
    if isempty(all_pngs)
        @warn "No forward sweep output_active directories found for this grid (run the forward sweep first, or check registry / ladder flags)."
    else
        @info "Forward sweep comparison figures finished" n_png = length(all_pngs) root = root
    end
    return all_pngs
end

"""
    va_plot_forward_sweep_clw_plus_cli_summary!(experiment_dir, cfg::ForwardSweepConfig; figure_root) -> Vector{String}

One **multi-panel figure** with **one row per registry case** (order from [`va_load_forward_sweep_case_rows`](@ref)) and
**one column per resolution tier** (finest left → coarsest right, same ordering as [`va_resolution_tiers_for_forward`](@ref)).
Each panel plots **`clw` + `cli`** with the same sweep styling as [`va_plot_profile_shortname_sum!`](@ref): palette, varfix
linestyles, optional observations-y, and finer-mesh overlay when available.

Writes **`forward_sweep_clw_plus_cli_summary.png`** under the same `figure_root` convention as
[`va_plot_forward_sweep_comparisons!`](@ref).

**Y limits:** Within each case row, all panels share the same height cap: the **maximum** of per-panel
[`va_condensate_cloud_top_height_m`](@ref) values (so coarse and fine tiers align vertically). If no panel
returns a cap, panels fall back to individual caps / autoscale.

**X limits:** After drawing, each row’s panels share the same **x** range (union of per-panel data limits plus small
padding) so mixing-ratio columns are comparable. **X tick labels** are shown on **every** row; the **x-axis label**
string stays on the **bottom** row only (avoids repeating `clw+cli (kg/kg)` on every row).
"""
function va_plot_forward_sweep_clw_plus_cli_summary!(
    experiment_dir::AbstractString,
    cfg::ForwardSweepConfig;
    figure_root::Union{Nothing, AbstractString} = nothing,
)::Vector{String}
    root = something(
        figure_root,
        joinpath(
            experiment_dir,
            "analysis",
            "figures",
            cfg.forward_parameters == VA_FORWARD_PARAM_BASELINE_SCM ? "forward_sweep_baseline_scm" :
            "forward_sweep_eki_calibrated",
        ),
    )
    tasks = va_flatten_forward_sweep_tasks(experiment_dir, cfg)
    groups = Dict{Tuple{String, String, String}, Vector{Tuple{Int, Bool, String, String, Union{Nothing, String}}}}()
    for (layers, _scm, slug, n, vf, tier, z_stretch, yaml_dz, _eo, _en, vfon) in tasks
        yml = layers[end]
        res_seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
        ap = va_forward_sweep_output_active_path(
            experiment_dir,
            slug,
            res_seg,
            n,
            vf,
            cfg;
            forced_varfix_on_distribution = vfon,
        )
        !isdir(ap) && continue
        k = (slug, yml, res_seg)
        lab = if !vf
            "N=$(n) vf_off"
        elseif vfon === nothing
            "N=$(n) vf_on (default)"
        else
            string("N=$(n) vf_on ", va_sgs_dist_path_slug(vfon))
        end
        push!(get!(groups, k, Tuple{Int, Bool, String, String, Union{Nothing, String}}[]), (n, vf, ap, lab, vfon))
    end

    rows = va_load_forward_sweep_case_rows(experiment_dir, cfg)
    isempty(rows) && return String[]

    tier_lens = [length(va_resolution_tiers_for_forward(r.case_dict, cfg)) for r in rows]
    max_cols = maximum(tier_lens)
    max_cols < 1 && return String[]

    ref_i = argmax(tier_lens)
    ref_row = rows[ref_i]
    zsr = va_forward_sweep_case_z_stretch(ref_row.case_dict)
    ydr = va_forward_sweep_case_dz_bottom(ref_row.case_dict)
    ref_tiers = va_resolution_tiers_for_forward(ref_row.case_dict, cfg)
    col_segs = String[va_tier_path_segment(t, zsr, ydr) for t in ref_tiers]
    while length(col_segs) < max_cols
        push!(col_segs, "")
    end

    n_rows = length(rows)
    w_panel = 300
    h_panel = 360
    fig = CM.Figure(;
        size = (80 + max_cols * w_panel + 200, 48 + (n_rows + 1) * h_panel),
    )
    CM.Label(fig[1, 1], ""; tellwidth = true, tellheight = false)
    for ci in 1:max_cols
        CM.Label(
            fig[1, ci+1],
            col_segs[ci];
            fontsize = 11,
            tellwidth = false,
            tellheight = false,
        )
    end

    legend_ax = nothing
    any_panel = false

    for (ri, row) in enumerate(rows)
        CM.Label(fig[ri+1, 1], row.slug; halign = :right, fontsize = 10, tellwidth = true, tellheight = false)
        tiers = va_resolution_tiers_for_forward(row.case_dict, cfg)
        zs = va_forward_sweep_case_z_stretch(row.case_dict)
        yd = va_forward_sweep_case_dz_bottom(row.case_dict)
        segs = String[va_tier_path_segment(t, zs, yd) for t in tiers]

        pairs = va_case_yaml_diagnostic_shortname_period_pairs(experiment_dir, row.model_config_layers)
        per_a = _va_yaml_period_for_short_name(pairs, "clw")
        per_b = _va_yaml_period_for_short_name(pairs, "cli")
        z_elem = va_z_elem_from_case_yaml(experiment_dir, row.model_config_layers)

        n_ref = 3
        vf_ref = false
        row_ylim_caps = Union{Nothing, Float64}[nothing for _ in 1:max_cols]
        for ci in 1:max_cols
            if ci > length(segs) || per_a === nothing || per_b === nothing
                continue
            end
            res_seg = segs[ci]
            k = (row.slug, row.yaml_rel, res_seg)
            cells = get(groups, k, Tuple{Int, Bool, String, String, Union{Nothing, String}}[])
            isempty(cells) && continue
            sort!(cells; by = x -> (x[1], x[2] ? 1 : 0, something(x[5], "")))
            paths = String[c[3] for c in cells]
            paths_abs = String[abspath(p) for p in paths]
            calibration_truth_series = nothing
            if row.eki_varfix_off_config !== nothing
                truth_active = va_reference_output_active(experiment_dir, row.eki_varfix_off_config)
                if isdir(truth_active)
                    calibration_truth_series = Tuple{String, String}[(
                        truth_active,
                        "Observations y (ref column)",
                    )]
                end
            end
            forward_finest_series = nothing
            fz = va_forward_sweep_reference_finer_than_panel(
                experiment_dir,
                row.slug,
                row.yaml_rel,
                cfg,
                res_seg;
                n_quad = n_ref,
                varfix = vf_ref,
            )
            if fz !== nothing
                _, ref_active, ref_tier_seg = fz
                if abspath(ref_active) ∉ paths_abs
                    forward_finest_series = Tuple{String, String}[(
                        ref_active,
                        "Finer mesh (N=$(n_ref) $(vf_ref ? "vf_on" : "vf_off")): $(ref_tier_seg)",
                    )]
                end
            end
            row_ylim_caps[ci] = va_condensate_cloud_top_height_m(
                paths,
                calibration_truth_series,
                forward_finest_series;
                experiment_dir,
                model_config_rel = row.model_config_layers,
                _VA_FORWARD_SWEEP_CLOUD_TOP_KW...,
            )
        end
        finite_row_caps = Float64[c for c in row_ylim_caps if c !== nothing]
        ylim_row = isempty(finite_row_caps) ? nothing : maximum(finite_row_caps)

        # After drawing, set one shared x-range per row so resolution columns are comparable side-by-side.
        row_axes_for_xlink = CM.Axis[]
        for ci in 1:max_cols
            ax = CM.Axis(
                fig[ri+1, ci+1];
                xlabel = ri == n_rows ? "clw+cli (kg/kg)" : "",
                ylabel = ci == 1 ? "height (m)" : "",
                titlevisible = false,
                # Show x tick **labels** on every row (not only the bottom); x-axis **title** stays on bottom row only.
                xticklabelsvisible = true,
                xticksvisible = true,
                yticklabelsvisible = ci == 1,
                yticksvisible = ci == 1,
            )
            if ci > length(segs) || per_a === nothing || per_b === nothing
                CM.hidedecorations!(ax)
                CM.hidespines!(ax)
                continue
            end
            res_seg = segs[ci]
            k = (row.slug, row.yaml_rel, res_seg)
            cells = get(groups, k, Tuple{Int, Bool, String, String, Union{Nothing, String}}[])
            if isempty(cells)
                CM.hidedecorations!(ax)
                CM.hidespines!(ax)
                continue
            end
            sort!(cells; by = x -> (x[1], x[2] ? 1 : 0, something(x[5], "")))
            paths = String[c[3] for c in cells]
            paths_abs = String[abspath(p) for p in paths]
            labels = String[c[4] for c in cells]
            nmax = maximum(c[1] for c in cells)
            pal = _va_forward_sweep_nquad_colors(nmax)
            path_colors = [pal[c[1]] for c in cells]
            path_linestyles = [c[2] ? :solid : :dash for c in cells]
            path_linewidths = fill(1.85, length(cells))

            calibration_truth_series = nothing
            if row.eki_varfix_off_config !== nothing
                truth_active = va_reference_output_active(experiment_dir, row.eki_varfix_off_config)
                if isdir(truth_active)
                    calibration_truth_series = Tuple{String, String}[(
                        truth_active,
                        "Observations y (ref column)",
                    )]
                end
            end
            forward_finest_series = nothing
            fz = va_forward_sweep_reference_finer_than_panel(
                experiment_dir,
                row.slug,
                row.yaml_rel,
                cfg,
                res_seg;
                n_quad = n_ref,
                varfix = vf_ref,
            )
            if fz !== nothing
                _, ref_active, ref_tier_seg = fz
                if abspath(ref_active) ∉ paths_abs
                    forward_finest_series = Tuple{String, String}[(
                        ref_active,
                        "Finer mesh (N=$(n_ref) $(vf_ref ? "vf_on" : "vf_off")): $(ref_tier_seg)",
                    )]
                end
            end
            ylims_h_panel = row_ylim_caps[ci]
            n_series, any_line = _va_plot_clw_plus_cli_series_on_axis!(
                ax,
                paths,
                "clw",
                "cli",
                per_a,
                per_b,
                z_elem,
                labels,
                path_colors,
                path_linestyles,
                path_linewidths,
                calibration_truth_series,
                forward_finest_series,
                _VA_FORWARD_FINER_MESH_COLOR,
                4.35,
            )
            if !any_line
                CM.hidedecorations!(ax)
                CM.hidespines!(ax)
                continue
            end
            any_panel = true
            ylims_apply = ylim_row !== nothing ? ylim_row : ylims_h_panel
            _va_finalize_profile_axis_limits!(ax, ylims_apply)
            push!(row_axes_for_xlink, ax)
            if legend_ax === nothing && n_series > 1
                legend_ax = ax
            end
        end
        # Shared x limits for every populated panel in this case row (mixing ratio scale aligned).
        if !isempty(row_axes_for_xlink)
            xmin = Inf
            xmax = -Inf
            for ax in row_axes_for_xlink
                fl = ax.finallimits[]
                x0 = Float64(fl.origin[1])
                x1 = Float64(fl.origin[1] + fl.widths[1])
                xmin = min(xmin, x0)
                xmax = max(xmax, x1)
            end
            if isfinite(xmin) && isfinite(xmax) && xmax > xmin
                dx = xmax - xmin
                pad = max(dx * 0.04, 1e-15)
                for ax in row_axes_for_xlink
                    CM.xlims!(ax, xmin - pad, xmax + pad)
                end
            end
        end
    end

    if !any_panel
        @warn "clw+cli summary: no drawable panels (missing forward outputs or clw/cli diagnostics)."
        return String[]
    end
    outpng = joinpath(root, "forward_sweep_clw_plus_cli_summary.png")
    mkpath(root)
    if legend_ax !== nothing
        CM.Legend(
            fig[2:n_rows+1, max_cols+2],
            legend_ax;
            framevisible = true,
            backgroundcolor = :white,
        )
        CM.colgap!(fig.layout, 12)
        CM.colsize!(fig.layout, max_cols + 2, CM.Auto(0.22))
    end
    CM.save(outpng, fig)
    @info "Wrote forward sweep clw+cli summary figure" outpng
    return String[outpng]
end

"""
    va_plot_forward_sweep_scalars_vs_nquad!(experiment_dir, cfg; figure_root, scalar_short_names, period_yaml) -> Vector{String}

For each `(case_slug, res_segment)` in the sweep grid, write **`scalar_<name>_vs_nquad.png`** with **LWP / ice path
(`clivi`) / …** vs **`quadrature_order`** (one line per **varfix off**, **varfix on (default)**, and each explicit
**varfix-on** distribution when `forward_sweep_varfix_on_distributions` is set in merged case YAML), plus a **black**
horizontal line for the **EKI reference** scalar when that run exists. Requires the diagnostics in SimDir output (add
**`lwp`** / **`clivi`** to the case YAML `diagnostics` if missing).
"""
function va_plot_forward_sweep_scalars_vs_nquad!(
    experiment_dir::AbstractString,
    cfg::ForwardSweepConfig;
    figure_root::Union{Nothing, AbstractString} = nothing,
    scalar_short_names = ("lwp", "clivi"),
    period_yaml::AbstractString = "10mins",
)::Vector{String}
    root = something(
        figure_root,
        joinpath(
            experiment_dir,
            "analysis",
            "figures",
            cfg.forward_parameters == VA_FORWARD_PARAM_BASELINE_SCM ? "forward_sweep_baseline_scm" :
            "forward_sweep_eki_calibrated",
        ),
    )
    tasks = va_flatten_forward_sweep_tasks(experiment_dir, cfg)
    panel_keys = Set{Tuple{String, String, String}}()
    for (layers, _scm, slug, n, vf, tier, z_stretch, yaml_dz, _eo, _en, vfon) in tasks
        yml = layers[end]
        res_seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
        ap = va_forward_sweep_output_active_path(
            experiment_dir,
            slug,
            res_seg,
            n,
            vf,
            cfg;
            forced_varfix_on_distribution = vfon,
        )
        !isdir(ap) && continue
        push!(panel_keys, (slug, yml, res_seg))
    end
    out_all = String[]
    wong = collect(CM.Makie.wong_colors())
    for (slug, yml, res_seg) in sort!(collect(panel_keys))
        row = va_forward_sweep_registry_row_for(experiment_dir, slug, yml, cfg)
        truth_active = row !== nothing && row.eki_varfix_off_config !== nothing ?
            va_reference_output_active(experiment_dir, row.eki_varfix_off_config) : nothing
        truth_simdir = truth_active !== nothing && isdir(truth_active) ? SimDir(truth_active) : nothing
        for name in scalar_short_names
            safe = replace(string(name), r"[^\w\.\-]+" => "_")
            outpng = joinpath(root, slug, res_seg, "scalars", "scalar_$(safe)_vs_nquad.png")
            mkpath(dirname(outpng))
            fig = CM.Figure(size = (560, 420))
            ax = CM.Axis(
                fig[1, 1];
                xlabel = "quadrature_order N",
                ylabel = string(name),
                titlevisible = false,
            )
            any_line = false
            line_keys = Tuple{Bool, Union{Nothing, String}}[]
            for (layers2, _scm, slug2, n, vf2, tier, z_stretch, yaml_dz, _eo, _en, vfon2) in tasks
                slug2 != slug && continue
                layers2[end] != yml && continue
                va_tier_path_segment(tier, z_stretch, yaml_dz) != res_seg && continue
                lk = (vf2, vf2 ? vfon2 : nothing)
                ap = va_forward_sweep_output_active_path(
                    experiment_dir,
                    slug,
                    res_seg,
                    n,
                    vf2,
                    cfg;
                    forced_varfix_on_distribution = vfon2,
                )
                !isdir(ap) && continue
                sdir = SimDir(ap)
                isempty(sdir.vars) && continue
                v = va_scalar_surface_mean_last(sdir, string(name), period_yaml)
                v === nothing && continue
                lk ∉ line_keys && push!(line_keys, lk)
            end
            sort!(line_keys; by = x -> (x[1] ? 1 : 0, something(x[2], "")))
            for (ci, lk) in enumerate(line_keys)
                vf, vfon = lk
                linestyle = vf ? :solid : :dash
                lab_prefix = if !vf
                    "vf_off"
                elseif vfon === nothing
                    "vf_on (default)"
                else
                    string("vf_on ", va_sgs_dist_path_slug(vfon))
                end
                ns = Int[]
                vals = Float64[]
                for (layers2, _scm, slug2, n, vf2, tier, z_stretch, yaml_dz, _eo, _en, vfon2) in tasks
                    slug2 != slug && continue
                    layers2[end] != yml && continue
                    va_tier_path_segment(tier, z_stretch, yaml_dz) != res_seg && continue
                    vf2 != vf && continue
                    (vf2 ? vfon2 : nothing) != (vf ? vfon : nothing) && continue
                    ap = va_forward_sweep_output_active_path(
                        experiment_dir,
                        slug,
                        res_seg,
                        n,
                        vf2,
                        cfg;
                        forced_varfix_on_distribution = vfon2,
                    )
                    !isdir(ap) && continue
                    sdir = SimDir(ap)
                    isempty(sdir.vars) && continue
                    v = va_scalar_surface_mean_last(sdir, string(name), period_yaml)
                    v === nothing && continue
                    push!(ns, n)
                    push!(vals, v)
                end
                if length(ns) >= 1
                    p = sortperm(ns)
                    CM.lines!(
                        ax,
                        ns[p],
                        vals[p];
                        color = wong[mod1(ci, length(wong))],
                        linestyle = linestyle,
                        linewidth = 2.0,
                        label = lab_prefix,
                    )
                    any_line = true
                end
            end
            if truth_simdir !== nothing && !isempty(truth_simdir.vars)
                tv = va_scalar_surface_mean_last(truth_simdir, string(name), period_yaml)
                if tv !== nothing
                    CM.hlines!(ax, tv; color = :black, linewidth = 2.85, label = "Observations y (ref column)")
                    any_line = true
                end
            end
            if any_line
                CM.axislegend(
                    ax;
                    position = :lt,
                    framevisible = true,
                    backgroundcolor = :white,
                )
                CM.save(outpng, fig)
                push!(out_all, outpng)
            else
                @debug "Scalar vs N: no data" name slug res_seg
            end
        end
    end
    if isempty(out_all)
        @warn "No scalar vs N_quad figures written (missing outputs or diagnostics `lwp`/`clivi` in SimDir)."
    else
        @info "Forward sweep scalar vs N_quad figures" n_png = length(out_all) root = root
    end
    return out_all
end
