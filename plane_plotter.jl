using Revise
using ClimaAnalysis
using CairoMakie
import ClimaAnalysis.Utils: kwargs as ca_kwargs
import ClimaAnalysis: select, view_select, Index, NearestValue, MatchValue
using ClimaInterpolations
using Colors, ColorSchemes
import ClimaInterpolations.Interpolation1D: Linear, interpolate1d!, Flat

### USER ENTERS DIRECTORY
dir_old_pgf = "./output/plane_schar_mountain_float64_test/output_ref"
dir_exner_ref_pgf = "./output/plane_schar_mountain_float64_test/output_0104"

function get_dat_2d(dir_labels, dir_paths, short_name, reduction; time_idx=nothing)
    figsize = (1500, 1200);
    fig = Figure(; size = figsize)
    fontsize = 20
    
    # Store data for difference calculation
    all_data = Dict()
    
    # First, collect all the data
    for (dir_idx, dir_path) in enumerate(dir_paths)
        simdir = ClimaAnalysis.SimDir(dir_path)
        var = get(simdir; short_name = short_name, reduction = reduction);
        
        if isnothing(time_idx)
            vardat = var.data[end, :, :]
        else
            vardat = var.data[time_idx, :, :]
        end
        
        all_data[dir_idx] = (; var, vardat)
    end
    
    # Determine common levels for consistent coloring (symmetric around zero)
    all_arrays = [all_data[i].vardat for i in 1:length(dir_paths)]
    global_min = minimum(minimum.(all_arrays))
    global_max = maximum(maximum.(all_arrays))
    global_extreme = max(abs(global_min), abs(global_max))
    levels = range(-global_extreme, global_extreme, length=20)
    levels = filter(!=(0), levels)
    
    # Plot original data (first two rows)
    plt = nothing
    for (fig_iy, dir_path) in enumerate(dir_paths)
        subtitle = dir_labels[fig_iy]
        var_items = all_data[fig_iy]
        
        println("Row $(fig_iy): ", extrema(var_items.vardat))
        
        fig_items = (; fig, fig_ix = 1, fig_iy, fontsize)
        plt = plot_panel_2d(fig_items, var_items, subtitle, levels, short_name)
    end
    
    # Add colorbar for original data (with proper spacing)
    unit_label = get_unit_label(short_name)
    Colorbar(
        fig[1:2, 2],  # Move to column 3 to avoid overlap
        plt,
        label = "$(short_name) $(unit_label)",
        labelsize = fontsize,
        ticklabelsize = fontsize,
    )
    
    # Plot differences (third row)
    @show "Plotting $(short_name) Differences between methods"
    
    # Calculate difference: method 2 - method 1
    var_items_1 = all_data[1]  # First directory
    var_items_2 = all_data[2]  # Second directory
    
    vardat_diff = var_items_2.vardat - var_items_1.vardat
    diff_extreme = max(abs(minimum(vardat_diff)), abs(maximum(vardat_diff)))
    diff_levels = range(-diff_extreme, diff_extreme, length=20)
    
    println("Difference Row: ", extrema(vardat_diff))
    
    # Create modified var_items for difference plot
    diff_var_items = (; var = var_items_1.var, vardat = vardat_diff)
    fig_items = (; fig, fig_ix = 1, fig_iy = 3, fontsize)  # Third row
    subtitle = "Difference ($(dir_labels[2]) - $(dir_labels[1]))"
    diff_plt = plot_panel_2d(fig_items, diff_var_items, subtitle, diff_levels, short_name, is_difference=true)
    
    # Add colorbar for differences (with proper spacing)
    Colorbar(
        fig[3, 2],  # Move to column 3 to avoid overlap
        diff_plt,
        label = "$(short_name) Difference $(unit_label)",
        labelsize = fontsize,
        ticklabelsize = fontsize,
    )
    
    return fig
end

function get_unit_label(short_name)
    # Map variable names to appropriate units
    unit_map = Dict(
        "ua" => "[m/s]",
        "wa" => "[m/s]",
        "va" => "[m/s]",
        "pfull" => "[Pa]",
        "ta" => "[K]",
        "theta" => "[K]",
        "uaerror" => "[m/s]",
        "waerror" => "[m/s]"
    )
    return get(unit_map, short_name, "[units]")
end

function plot_panel_2d(fig_items, var_items, subtitle, levels, short_name; is_difference=false)
    (;fig, fig_ix, fig_iy, fontsize) = fig_items
    (;var, vardat) = var_items
    
    # Spatial Data
    x = var.dims["x"]
    z_ref = var.dims["z_reference"]
    xvariable = x
    yvariable = z_ref
    
    # Create title
    title_text = subtitle
    
    ax = Axis(
        fig[fig_iy, fig_ix],
        title = title_text,
        xlabel = "x [m]",
        ylabel = "z [m]",
        titlesize = fontsize,
        xlabelsize = 20,
        ylabelsize = 20,
        xticklabelsize = 20,
        yticklabelsize = 20,
        xgridvisible = false,
        ygridvisible = false,
    )
    
    # Use different colorscheme for difference plots
    if is_difference
        plt = CairoMakie.contourf!(
            xvariable,
            yvariable,
            vardat,
            levels = levels,
            colormap = Reverse(:RdBu),  # Red-Blue colormap for differences (reversed)
        )
    else
        plt = CairoMakie.contourf!(
            xvariable,
            yvariable,
            vardat,
            levels = levels,
            colormap = Reverse(:RdBu),  # Use symmetric colormap for all plots
        )
    end
    
    tightlimits!(ax)
    
    return plt
end

# Usage examples
dir_labels = ("Old PGF", "Exner PGF + ref. prof.")
dir_paths = (dir_old_pgf, dir_exner_ref_pgf)

# Plot eastward wind
@show "Plotting Eastward Wind"
fig1 = get_dat_2d(dir_labels, dir_paths, "uaerror", "inst")
fig1[0, 1:2] = Label(fig1, "Error in the eastward wind, instantaneous time = 6 Days", fontsize = 24)
CairoMakie.save("ua_comparison_adapted.pdf", fig1)

# Plot upward air velocity
@show "Plotting Upward Air Velocity"
fig2 = get_dat_2d(dir_labels, dir_paths, "waerror", "inst")
fig2[0, 1:2] = Label(fig2, "Error in the upward air velocity, instantaneous time = 6 Days", fontsize = 24)
CairoMakie.save("wa_comparison_adapted.pdf", fig2)

# If you want to plot at a specific time index instead of the last time step:
# fig3 = get_dat_2d(dir_labels, dir_paths, "ua", "inst"; time_idx=10)

# You can also easily add more variables:
# fig4 = get_dat_2d(dir_labels, dir_paths, "ta", "inst")  # temperature
# fig5 = get_dat_2d(dir_labels, dir_paths, "pfull", "inst")  # pressure