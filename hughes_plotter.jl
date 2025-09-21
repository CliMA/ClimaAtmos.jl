using Revise
using ClimaAnalysis
using CairoMakie
import ClimaAnalysis.Utils: kwargs as ca_kwargs
import ClimaAnalysis: select, view_select, Index, NearestValue, MatchValue
using ClimaInterpolations
using Colors, ColorSchemes
import ClimaInterpolations.Interpolation1D: Linear, interpolate1d!, Flat

### USER ENTERS DIRECTORY
# Note: I switched the namings here, exner is the ref, and ref is the exner!
dir_exner_pgf = "/home/ray/Documents/2025/dycore_paper_review/hughes_jablonowski/exner_pgf/nc_files-exner"
dir_old_pgf = "/home/ray/Documents/2025/dycore_paper_review/hughes_jablonowski/ref_pgf/nc_files-ref"

function get_dat(dir_labels, dir_paths, time_slcs)
    figsize = (1500, 1200);
    fig = Figure(; size = figsize)
    fontsize = 20
    
    # Store data for difference calculation
    all_data = Dict()
    
    # First, collect all the data
    for (dir_idx, dir_name) in enumerate(dir_names)
        all_data[dir_idx] = Dict()
        for (time_idx, time_slc) in enumerate(time_slcs)
            simdir = ClimaAnalysis.SimDir(dir_name)
            dat = get(simdir; short_name = "pfull", reduction = "inst");
            zg = get(simdir; short_name = "zg", reduction = "inst");
            arr = dat.data
            arr = reverse(arr, dims=4)
            arr = arr[time_slc, :, :, :] - arr[1, :, : ,:]
            arr = permutedims(arr, (3, 1, 2))
            arr = arr[43, : , :]
            arr = arr ./ 100.0 # convert to hectopascal
            
            all_data[dir_idx][time_idx] = (; dat, arr, zg)
        end
    end
    
    # Plot original data (first two rows)
    for (fig_iy, dir_name) in enumerate(dir_names)
        levels = nothing
        ticks = nothing
        plt = nothing
		subtitle = dir_labels[fig_iy]
        for (fig_ix, time_slc) in enumerate(time_slcs)
            dat_items = all_data[fig_iy][fig_ix]
            arr = dat_items.arr
            
            if isnothing(levels)
                levels = range(minimum(arr)-1, maximum(arr), 8)
                ticks  = range(minimum(arr)-1, maximum(arr), 8)
            end
            println("Row $(fig_iy), Col $(fig_ix): ", levels)

            fig_items = (; fig, fig_ix, fig_iy, fontsize)
            plt = plot_panel(fig_items, dat_items, time_slc, subtitle, levels)
        end
        Colorbar(
            fig[fig_iy, 3],
            plt,
            label = "Sfc Pressure Perturbation (hPa)",
            labelsize = fontsize,
            ticklabelsize = fontsize,
        )
    end
    
    # Plot differences (third row)
    @show "Plotting Pressure Differences between rows"
    diff_levels = nothing
    diff_plt = nothing
    
    for (fig_ix, time_slc) in enumerate(time_slcs)
        # Calculate difference: row 2 - row 1
        dat_items_1 = all_data[1][fig_ix]  # First directory
        dat_items_2 = all_data[2][fig_ix]  # Second directory
        
        arr_diff = dat_items_2.arr - dat_items_1.arr
        
        if isnothing(diff_levels)
            diff_levels = range(minimum(arr_diff), maximum(arr_diff), 8)
        end
        println("Difference Row, Col $(fig_ix): ", diff_levels)
        
        # Create modified dat_items for difference plot
        diff_dat_items = (; dat = dat_items_1.dat, arr = arr_diff, zg = dat_items_1.zg)
        fig_items = (; fig, fig_ix, fig_iy = 3, fontsize)  # Third row
        diff_plt = plot_panel(fig_items, diff_dat_items, time_slc, diff_levels, is_difference=true)
    end
    
    # Add colorbar for differences
    Colorbar(
        fig[3, 3],
        diff_plt,
        label = "Pressure Difference (old - Exner) (hPa)",
        labelsize = fontsize,
        ticklabelsize = fontsize,
    )
    
    # for (fig_iy, dir_name) in enumerate(dir_names)
    #     levels = nothing
    #     ticks = nothing
    #     plt = nothing
    #     for (fig_ix, time_slc) in enumerate(time_slcs)
    #         simdir = ClimaAnalysis.SimDir(dir_name)
    #         dat = get(simdir; short_name = "pfull", reduction = "inst");
    #         zg = get(simdir; short_name = "zg", reduction = "inst");
    #         arr = dat.data
    #         arr = reverse(arr, dims=4)
    #         arr = arr[time_slc, :, :, :] - arr[1, :, : ,:]
    #         arr = permutedims(arr, (3, 1, 2))
    #         arr = arr[43, : , :]
    #         arr = arr ./ 100.0 # convert to hectopascal

    #         if isnothing(levels)
    #             levels = range(minimum(arr)-1, maximum(arr), 8)
#   #              levels = filter(!=(0), levels)
    #             ticks  = range(minimum(arr)-1, maximum(arr), 8)
    #         end
    #         println(fig_ix, levels)

    #         fig_items = (; fig, fig_ix, fig_iy, fontsize)
    #         dat_items = (; dat, arr, zg)
    #         plt = plot_panel(fig_items, dat_items, time_slc, levels)
    #     end
    #     Colorbar(
    #         fig[fig_iy, 3],
    #         plt,
    #         label = "Sfc Pressure Perturbation (hPa)",
    #         labelsize = fontsize,
    #         ticklabelsize = fontsize,
    #         # ticks = ticks,
    #     )
    # end
    ###### PRESSURE DIFFERENCE (SURFACE)
    @show "Plotting Pressure Difference"
    fig
    CairoMakie.save(
        "paperrevision_mbw_hughes_air_pressure.pdf",
        fig,
    )
end


function offset_longitudes(longitude)
    new_longitude = mod.(longitude .+ 360, 360)
    return circshift(new_longitude, 720)
end

function offset_data(data)
    new_data = circshift(data, (720, 0))
    return new_data
end

function plot_panel(fig_items, dat_items, time_slc, subtitle, levels; is_difference=false)
    (;fig , fig_ix, fig_iy, fontsize) = fig_items
    (;dat, arr, zg) = dat_items
    
    # Spatial Data
    lat = dat.dims["lat"];
    lon = dat.dims["lon"];
    xvariable = lon
    yvariable = lat

    # Create different titles based on whether it's a difference plot
    if is_difference
        title_text = "Difference - Day $(time_slc-1)"
    else
        if fig_iy == 1
            title_text = "$(subtitle) - Day $(time_slc-1)"
        else
            title_text = "$(subtitle) - Day $(time_slc-1)"
        end
    end
    
    ax = Axis(
        fig[fig_iy, fig_ix],
        limits = (0, 245, 0, 90),
        title = title_text,
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)",
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
            offset_longitudes(xvariable),
            yvariable,
            offset_data(arr),
            levels = 10,
            colormap = Reverse(:RdBu),  # Red-Blue colormap for differences
        )
    else
        plt = CairoMakie.contourf!(
            offset_longitudes(xvariable),
            yvariable,
            offset_data(arr),
            levels = 10,
        )
    end
    
    # Add topography contours
    CairoMakie.contour!(
        xvariable,
        yvariable,
        zg.data[1, :, :, 1] .- 15.0,
        alpha = 1,
        levels = 2,
        color = :gray,
    )
    return plt
end

# function plot_panel(fig_items, dat_items, time_slc, levels)
#     (;fig , fig_ix, fig_iy, fontsize) = fig_items
#     (;dat, arr, zg) = dat_items
#     #SpatialData
#     lat = dat.dims["lat"]; # Index 540 for 45 degree north
#     lon = dat.dims["lon"];
#     xvariable = lon
#     yvariable = lat
# 
#     # mycolors = reverse(colormap("RdBu"; logscale = false, mid = 0.5))
#     ax = Axis(
#         fig[fig_ix, fig_iy],
#         limits = (0, 245, 0, 90),
#         title = "Day $(time_slc-1)",
#         xlabel = "Longitude (°E)",
#         ylabel = "Latitude (°N)",
#         titlesize = fontsize,
#         xlabelsize = 20,
#         ylabelsize = 20,
#         xticklabelsize = 20,
#         yticklabelsize = 20,
#         xgridvisible = false,
#         ygridvisible = false,
#     )
#     plt = CairoMakie.contourf!(
#         offset_longitudes(xvariable),
#         yvariable,
#         offset_data(arr),
#         levels = 10,
#         # extendhigh = :auto,
#         # extendlow = :auto
#     )
#     # @Main.infiltrate
#     CairoMakie.contour!(
#         xvariable,
#         yvariable,
#         zg.data[1, :, :, 1] .- 15.0,
#         alpha = 1,
#         levels = 2,
#         color = :gray,
#     )
#     return plt
# end

dir_labels = ("old PGF", "Exner PGF + ref. prof.")
dir_paths = (dir_old_pgf, dir_exner_pgf)
time_slcs  = (5, 7)

get_dat(dir_labels, dir_paths, time_slcs)
