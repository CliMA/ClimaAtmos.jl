# Plot quadrature results saved by offline quadrature workflow.

using JLD2
using CairoMakie

function write_quadrature_summary_figure!(
    case_outdir::String, outpath::String;
    res_segs::Vector{String},
    quadrature_orders::Vector{Int},
    sgs_distributions::Vector{String},
    regrid_method::Symbol,
    color_map=Dict{String,Any}(),
    kwargs...
)
    n_rows = length(res_segs)
    n_cols = length(quadrature_orders)
    
    fig = Figure(size = (400 * n_cols + 200, 400 * n_rows))
    
    # Semantic styling: Group by family and methodology
    # 1. Bivariate (7-arg) -> Black, Dashed
    # 2. Vertically Resolved (18-arg) -> Steelblue, Solid
    # 3. Inner/Rosenblatt Variants -> Orange, Solid
    
    # Semantic styling: Color and Linestyle families
    # Family -> (Linestyle, Palette)
    # lognormal -> Solid, Blues/Purples
    # gaussian -> Dashed, Reds/Oranges
    dist_styles = Dict{String, Tuple{Any, Symbol, Symbol}}()
    for dist in sgs_distributions
        is_gaussian = contains(dist, "gaussian")
        ls = is_gaussian ? :dash : :solid
        mk = is_gaussian ? :rect : :circle
        
        # Base colors by methodology
        col = if contains(dist, "full_cubature")
            is_gaussian ? :firebrick : :steelblue
        elseif contains(dist, "lhs_z")
            is_gaussian ? :orangered : :deepskyblue
        elseif contains(dist, "principal_axis")
            is_gaussian ? :tomato : :royalblue
        elseif contains(dist, "voronoi")
            is_gaussian ? :darkred : :darkcyan
        elseif contains(dist, "barycentric")
            is_gaussian ? :indianred : :cadetblue
        elseif contains(dist, "bracketed")
            is_gaussian ? :orange : :purple
        elseif contains(dist, "halley")
            is_gaussian ? :darkorange : :darkorchid
        elseif contains(dist, "chebyshev")
            is_gaussian ? :goldenrod : :mediumpurple
        elseif dist == "lognormal"
            :black
        elseif dist == "gaussian"
            :grey50
        else
            :grey
        end
        
        # Override linestyle for base cases
        if dist == "lognormal"
            ls = :dash; mk = :circle
        elseif dist == "gaussian"
            ls = :dashdot; mk = :rect
        end
        
        dist_styles[dist] = (col, ls, mk)
    end
    
    truth_path = joinpath(case_outdir, "truth.jld2")
    truth_node = isfile(truth_path) ? load(truth_path, "truth_res") : nothing
    
    # Compute dynamic y-limit based on cloud top (condensate > 1e-9)
    max_z_cloud = 0.0
    if truth_node !== nothing && haskey(truth_node, "cond") && haskey(truth_node, "z")
        idx = findlast(x -> x > 1e-9, truth_node["cond"])
        if idx !== nothing
            max_z_cloud = max(max_z_cloud, truth_node["z"][idx])
        end
    end
    
    for res_seg in res_segs, N in quadrature_orders, dist in sgs_distributions
        dist_path = joinpath(case_outdir, string(regrid_method), res_seg, "N_$N", string(dist, ".jld2"))
        if isfile(dist_path)
            v = load(dist_path, "res")
            if haskey(v, "cond") && haskey(v, "z")
                idx = findlast(x -> x > 1e-9, v["cond"])
                if idx !== nothing
                    max_z_cloud = max(max_z_cloud, v["z"][idx])
                end
            end
        end
    end
    
    y_max = max_z_cloud > 0.0 ? max_z_cloud + 400.0 : nothing

    legend_ax = nothing
    any_plotted = false

    for (r, res_seg) in enumerate(res_segs)
        for (c, N) in enumerate(quadrature_orders)
            ax = Axis(fig[r, c],
                xlabel = (r == n_rows) ? "Condensate (kg/kg)" : "",
                ylabel = (c == 1) ? "Height (m)" : "",
                title = "Grid: $res_seg | N=$N"
            )
            if y_max !== nothing
                ylims!(ax, 0.0, y_max)
            end
            
            # Plot truth
            if truth_node !== nothing && haskey(truth_node, "cond") && haskey(truth_node, "z")
                scatterlines!(ax, truth_node["cond"], truth_node["z"], color=:black, linewidth=5, marker=:circle, markersize=3, label="Truth (High-Res N=5)")
                any_plotted = true
            end
            
            # Plot distributions
            for dist in sgs_distributions
                dist_path = joinpath(case_outdir, string(regrid_method), res_seg, "N_$N", string(dist, ".jld2"))
                if isfile(dist_path)
                    v = load(dist_path, "res")
                    if haskey(v, "cond") && haskey(v, "z")
                        col, ls, mk = dist_styles[dist]
                        scatterlines!(ax, v["cond"], v["z"], color=col, linestyle=ls, linewidth=2.5, marker=mk, markersize=4, label=dist)
                        any_plotted = true
                        if legend_ax === nothing
                            legend_ax = ax
                        end
                    end
                end
            end
            
            if !any_plotted
                hidedecorations!(ax)
                hidespines!(ax)
            end
        end
    end
    
    if legend_ax !== nothing
        Legend(fig[:, n_cols + 1], legend_ax, "SGS Distribution", framevisible=true, backgroundcolor=:white)
    end
    
    mkpath(dirname(outpath))
    save(outpath, fig)
    println("Saved summary plot to ", outpath)
end

function run_plot_registry_cases!(; outputs_root::String = joinpath(@__DIR__, "..", "outputs"), include_cases=nothing, exclude_cases=String[], color_map=Dict{String,Any}(), quadrature_orders::Vector{Int}=[1, 2, 3, 4, 5], sgs_distributions::Vector{String}=String["lognormal"], regrid_methods::Vector{Symbol}=Symbol[:linear, :block_average])
    if !isdefined(Main, :_offline_quadrature_forward_cfg) || !isdefined(Main, :va_load_forward_sweep_case_rows)
        error("Registry helpers not found. Include run_end_to_end.jl (which defines _offline_quadrature_forward_cfg and va_load_forward_sweep_case_rows) before calling this function.")
    end
    forward_cfg = Main._offline_quadrature_forward_cfg()
    experiment_dir = normpath(joinpath(@__DIR__, "..", ".."))
    rows = Main.va_load_forward_sweep_case_rows(experiment_dir, forward_cfg)
    
    slug_to_row = Dict(String(r.slug) => r for r in rows)
    slugs = collect(keys(slug_to_row))
    
    if include_cases !== nothing
        slugs = [s for s in slugs if s in include_cases]
    end
    if !isempty(exclude_cases)
        slugs = [s for s in slugs if !(s in exclude_cases)]
    end
    for slug in slugs
        case_dir = joinpath(outputs_root, slug)
        if !isdir(case_dir)
            @warn "No outputs directory for case" case=slug path=case_dir
            continue
        end
        
        row = slug_to_row[slug]
        z_stretch = Main.va_forward_sweep_case_z_stretch(row.case_dict)
        yaml_dz = Main.va_forward_sweep_case_dz_bottom(row.case_dict)
        tiers = Main.va_resolution_tiers_for_forward(row.case_dict, forward_cfg)
        res_segs = String[Main.va_tier_path_segment(t, z_stretch, yaml_dz) for t in tiers]
        
        for method in regrid_methods
            outpath = joinpath(case_dir, string(method), "quadrature_results_summary.png")
            write_quadrature_summary_figure!(case_dir, outpath; res_segs=res_segs, quadrature_orders=quadrature_orders, sgs_distributions=sgs_distributions, regrid_method=method, color_map=color_map)
        end
    end
    return nothing
end
