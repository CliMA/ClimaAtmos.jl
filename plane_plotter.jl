using Revise
using ClimaAnalysis
using CairoMakie
import ClimaAnalysis.Utils: kwargs as ca_kwargs
import ClimaAnalysis: select, view_select, Index, NearestValue, MatchValue
using ClimaInterpolations
using Colors, ColorSchemes
import ClimaInterpolations.Interpolation1D: Linear, interpolate1d!, Flat

### USER ENTERS DIRECTORY
#dir = "/Users/akshaysridhar/Research/Data/paper_data/test_hughes/moist/output_0003"
dir_old_pgf = "./output/plane_schar_mountain_float64_test/output_ref"
dir_exner_ref_pgf = "./output/plane_schar_mountain_float64_test/output_0072"

vardir = ClimaAnalysis.SimDir(dir_exner_ref_pgf)
refdir = ClimaAnalysis.SimDir(dir_old_pgf)

fontsize = 20

dirs = (refdir, vardir)
dir_names = ("Old PGF", "Exner+Ref. Prof.")

function gen_panel(fig_ix, dirs, dir_names, short_name, reduction, time_idx=nothing; cmap=nothing)

    plt = nothing
    levels = nothing
    for (fig_iy, dir) in enumerate(dirs)
        var = get(dir; short_name = short_name, reduction = reduction);
        if isnothing(time_idx)
            vardat = var.data[end, :, :]
        else
            vardat = var.data[time_idx, :, :]
        end

        if isnothing(levels)
            levels = range(minimum(vardat), maximum(vardat), length=20)
            levels = filter(!=(0), levels)
        end

        x = var.dims["x"]; # Index 540 for 45 degree north
        z_ref = var.dims["z_reference"];
        xvariable = x
        yvariable = z_ref

        subtitle = string(dir_names[fig_iy]) * " ($short_name)"
        plt = panel(fig_ix, fig_iy, xvariable, yvariable, vardat, subtitle, levels; cmap=cmap)
    end

    Colorbar(
    fig[fig_ix, length(dirs)+1],
    plt,
    label = "$short_name [m/s]",
    labelsize = fontsize,
    ticklabelsize = fontsize,
    )
end

function panel(fig_ix, fig_iy, xvariable, yvariable, var, title, levels; cmap=nothing)
    ax = Axis(
        fig[fig_ix, fig_iy],
        title = title,
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

    if isnothing(cmap)
        cmap = :viridis
    end

    plt = CairoMakie.contourf!(
        xvariable,
        yvariable,
        var,
        levels = levels,
        colormap = cmap,
        )

    tightlimits!(ax)

    return plt
end


@show "Plotting"
figsize = (1500, 1000);
fig = Figure(; size = figsize)
fig[0, 1:3] = Label(fig, "Eastward wind, instantaneous time = 48 Hours", fontsize = 24)

plt1 = gen_panel(1, dirs, dir_names, "ua", "inst")
plt2 = gen_panel(2, dirs, dir_names, "uaerror", "inst"; cmap=:seaborn_colorblind6)

fig
CairoMakie.save(
    "ua_comparison.pdf",
    fig,
)

@show "Plotting"
figsize = (1500, 1000);
fig = Figure(; size = figsize)
fig[0, 1:3] = Label(fig, "Upward air velocity, instantaneous time = 48 Hours", fontsize = 24)

plt1 = gen_panel(1, dirs, dir_names, "wa", "inst")
plt2 = gen_panel(2, dirs, dir_names, "waerror", "inst", cmap=:seaborn_colorblind6)

fig
CairoMakie.save(
    "wa_comparison.pdf",
    fig,
)