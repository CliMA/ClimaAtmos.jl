import CairoMakie
import CairoMakie.Makie
import ClimaAnalysis
import ClimaAnalysis: Visualize as viz
import ClimaAnalysis: SimDir, slice_time, slice
import ClimaAnalysis.Utils: kwargs as ca_kwargs

import ClimaCoreSpectra: power_spectrum_2d

using Poppler_jll: pdfunite
import Base.Filesystem

const days = 86400

# Return the last common directory across several files
function common_dirname(files::Vector{T}) where {T <: AbstractString}
    # Split the path of each file into a vector of strings
    # e.g. "/home/user/file1.txt" -> ["home", "user", "file1.txt"]
    split_files = split.(files, '/')
    # Find the index of the last common directory
    last_common_dir =
        findfirst(
            i -> any(j -> split_files[1][i] != j, split_files[2:end]),
            1:length(split_files[1]),
        ) - 1
    return joinpath(split_files[1][1:last_common_dir]...)
end

function make_plots(sim, _)
    @warn "No plot found for $sim"
end

# The contour plot functions in ClimaAnalysis work by finding the nearest slice available.
# If we want the extremes, we can just ask for the slice closest to a very large number.
const LARGE_NUM = typemax(Int)
const LAST_SNAP = LARGE_NUM
const FIRST_SNAP = -LARGE_NUM
const BOTTOM_LVL = -LARGE_NUM
const TOP_LVL = LARGE_NUM
const H_EARTH = 7000
# Shorthand for logscale on y axis and to move the dimension to the y axis on line plots
# (because they are columns)
Plvl(y) = -H_EARTH * log(y)
Makie.inverse_transform(::typeof(Plvl)) = (y) -> exp(-y / H_EARTH)
Makie.defaultlimits(::typeof(Plvl)) = (0.0000001, 1)
Makie.defined_interval(::typeof(Plvl)) = Makie.OpenInterval(0.0, Inf)
function Makie.get_tickvalues(yticks::Int, yscale::typeof(Plvl), ymin, ymax)
    exp_func = Makie.inverse_transform(yscale)
    exp_z_min, exp_z_max = exp_func(ymin), exp_func(ymax)
    return Plvl.(range(exp_z_min, exp_z_max, yticks))
end

YLOGSCALE = Dict(
    :axis => ca_kwargs(
        dim_on_y = true,
        yscale = Plvl,
        yticks = 7,
        ytickformat = "{:.3e}",
    ),
)

long_name(var) = var.attributes["long_name"]
short_name(var) = var.attributes["short_name"]

"""
    parse_var_attributes(var)

Takes in an OutputVar and parses some of its attributes into a short, informative string.
Used to generate unique titles when the same var is being plotted for several times/locations.
This could be extended to parse more attributes.

For example, the sample attributes:
attributes = Dict(
    "units" => "%",
    "short_name" => "cl",
    "slice_y" => "0.0",
    "long_name" => "Cloud fraction, Instantaneous x = 0.0 m y = 0.0 m",
    "slice_y_units" => "m",
    "slice_x_units" => "m",
    "comments" => "",
    "slice_x" => "0.0",
)
will be parsed into "cl, x = 0.0, y = 0.0"
"""
function parse_var_attributes(var)
    MISSING_STR = "MISSING_ATTRIBUTE"
    attr = var.attributes
    name = replace(short_name(var), "up" => "")

    attributes = ["slice_lat", "slice_lon", "slice_x", "slice_y", "slice_time"]
    info = [
        replace(key, "slice_" => "") * " = " * get(attr, key, MISSING_STR)
        for key in attributes
    ]
    # Filter out missing entries
    info = filter(x -> !occursin(MISSING_STR, x), [name, info...])

    return join(info, ", ")
end

function make_plots_generic(
    output_path,
    vars,
    args...;
    plot_fn = nothing,
    output_name = "summary",
    summary_files = String[],
    MAX_NUM_COLS = 1,
    MAX_NUM_ROWS = 4,
    kwargs...,
)
    # Default plotting function needs access to kwargs
    if isnothing(plot_fn)
        plot_fn =
            (grid_loc, var) -> viz.plot!(grid_loc, var, args...; kwargs...)
    end

    MAX_PLOTS_PER_PAGE = MAX_NUM_ROWS * MAX_NUM_COLS
    vars_left_to_plot = length(vars)

    # Define fig, grid, and grid_pos, used below. (Needed for scope)
    makefig() = CairoMakie.Figure(resolution = (900, 300 * MAX_NUM_ROWS))
    gridlayout() =
        map(1:MAX_PLOTS_PER_PAGE) do i
            row = mod(div(i - 1, MAX_NUM_COLS), MAX_NUM_ROWS) + 1
            col = mod(i - 1, MAX_NUM_COLS) + 1
            return fig[row, col] = CairoMakie.GridLayout()
        end

    fig = makefig()
    grid = gridlayout()
    page = 1
    grid_pos = 1

    for var in vars
        if grid_pos > MAX_PLOTS_PER_PAGE
            fig = makefig()
            grid = gridlayout()
            grid_pos = 1
        end

        plot_fn(grid[grid_pos], var)
        grid_pos += 1

        # Flush current page
        if grid_pos > min(MAX_PLOTS_PER_PAGE, vars_left_to_plot)
            file_path = joinpath(output_path, "$(output_name)_$page.pdf")
            CairoMakie.save(file_path, fig)
            push!(summary_files, file_path)
            vars_left_to_plot -= MAX_PLOTS_PER_PAGE
            page += 1
        end
    end

    output_file = joinpath(output_path, "$(output_name).pdf")

    pdfunite() do unite
        run(Cmd([unite, summary_files..., output_file]))
    end

    # Cleanup
    Filesystem.rm.(summary_files, force = true)
    return output_file
end

"""
    make_spectra_generic

Use ClimaCoreSpectra to compute and plot spectra for the given `vars`.

Extra arguments are passed to `ClimaAnalysis.slice`

"""
function make_spectra_generic(
    output_path,
    vars,
    args...;
    slicing_kwargs = ca_kwargs(),
    output_name = "spectra",
    kwargs...,
)
    sliced_vars = [slice(var; slicing_kwargs...) for var in vars]

    any([length(var.dims) != 2 for var in sliced_vars]) && error("Only 2D spectra are supported")

    # Prepare ClimaAnalysis.OutputVar
    spectra =
        map(sliced_vars) do var
            # power_spectrum_2d seems to work only when the two dimensions have precisely one
            # twice as many points as the other
            dim1, dim2 = var.index2dim[1:2]

            length(var.dims[dim1]) == 2 * length(var.dims[dim2]) ||
                error("Cannot take a this spectrum")

            FT = eltype(var.data)
            mass_weight = ones(FT, 1)
            spectrum_data, wave_numbers, spherical, mesh_info =
                power_spectrum_2d(FT, var.data, mass_weight)

            # From ClimaCoreSpectra/examples
            X = collect(0:1:(mesh_info.num_fourier))
            Y = collect(0:1:(mesh_info.num_spherical))
            Z = spectrum_data[:, :, 1]

            dims = Dict("num_fourier" => X, "num_spherical" => Y)
            dim_attributes = Dict(
                "num_fourier" => Dict("units" => ""),
                "num_spherical" => Dict("units" => ""),
            )

            attributes = Dict(
                "short_name" => "log fft_" * var.attributes["short_name"],
                "long_name" => "Spectrum of " * var.attributes["long_name"],
                "units" => "",
            )
            path = nothing

            return ClimaAnalysis.OutputVar(
                attributes,
                dims,
                dim_attributes,
                log.(Z),
                path,
            )
        end |> collect

    make_plots_generic(output_path, spectra, args...; output_name, kwargs...)
end

ColumnPlots = Union{
    Val{:single_column_hydrostatic_balance_ft64},
    Val{:single_column_radiative_equilibrium_gray},
    Val{:single_column_radiative_equilibrium_clearsky},
    Val{:single_column_radiative_equilibrium_clearsky_prognostic_surface_temp},
    Val{:single_column_radiative_equilibrium_allsky_idealized_clouds},
}

function make_plots(::ColumnPlots, simulation_path)
    simdir = SimDir(simulation_path)
    short_names = ["ta", "wa"]
    vars = [get(simdir, short_name) for short_name in short_names]
    make_plots_generic(
        simulation_path,
        vars,
        time = LAST_SNAP,
        x = 0.0, # Our columns are still 3D objects...
        y = 0.0,
        more_kwargs = YLOGSCALE,
    )
end

function make_plots(::Val{:box_hydrostatic_balance_rhoe}, simulation_path)
    simdir = SimDir(simulation_path)
    short_names, reduction = ["wa", "ua"], "average"
    vars = [get(simdir; short_name, reduction) for short_name in short_names]
    make_plots_generic(
        simulation_path,
        vars,
        y = 0.0,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
end

function make_plots(::Val{:single_column_precipitation_test}, simulation_path)
    simdir = SimDir(simulation_path)
    short_names = ["hus", "clw", "cli", "husra", "hussn", "ta"]
    vars = [
        slice(get(simdir; short_name), x = 0.0, y = 0.0) for
        short_name in short_names
    ]

    # We first prepare the axes with all the nice labels with ClimaAnalysis, then we use
    # CairoMakie to add the additional lines.
    fig = CairoMakie.Figure(resolution = (1200, 600))

    p_loc = [1, 1]

    axes = map(vars) do var
        viz.plot!(
            fig,
            var;
            time = 0.0,
            p_loc,
            more_kwargs = Dict(
                :plot => ca_kwargs(color = :navy),
                :axis => ca_kwargs(dim_on_y = true, title = ""),
            ),
        )

        # Make a grid of plots
        p_loc[2] += 1
        p_loc[2] > 3 && (p_loc[1] += 1; p_loc[2] = 1)
        return CairoMakie.current_axis()
    end

    col = Dict(500 => :blue2, 1000 => :royalblue, 1500 => :skyblue1)

    for (time, color) in col
        for (i, var) in enumerate(vars)
            CairoMakie.lines!(
                axes[i],
                slice(var; time).data,
                var.dims["z"],
                color = color,
            )
        end
    end

    file_path = joinpath(simulation_path, "summary.pdf")
    CairoMakie.save(file_path, fig)
end

function make_plots(::Val{:box_density_current_test}, simulation_path)
    simdir = SimDir(simulation_path)
    vars = [get(simdir, short_name = "thetaa")]
    make_plots_generic(simulation_path, vars, y = 0.0, time = LAST_SNAP)
end

MountainPlots = Union{
    Val{:plane_agnesi_mountain_test_uniform},
    Val{:plane_agnesi_mountain_test_stretched},
    Val{:plane_schar_mountain_test_uniform},
    Val{:plane_schar_mountain_test_stretched},
}

function make_plots(::MountainPlots, simulation_path)
    simdir = SimDir(simulation_path)
    vars = [get(simdir, short_name = "wa", reduction = "average")]
    make_plots_generic(
        simulation_path,
        vars,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
end

function make_plots(::Val{:plane_density_current_test}, simulation_path)
    simdir = SimDir(simulation_path)
    vars = [get(simdir, short_name = "thetaa")]
    make_plots_generic(
        simulation_path,
        vars,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
end

function make_plots(
    ::Val{:sphere_hydrostatic_balance_rhoe_ft64},
    simulation_path,
)
    simdir = SimDir(simulation_path)
    short_names, reduction = ["ua", "wa"], "average"
    vars = [
        get(simdir; short_name, reduction) |> ClimaAnalysis.average_lon for
        short_name in short_names
    ]
    make_plots_generic(
        simulation_path,
        vars,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
end

DryBaroWavePlots = Union{Val{:sphere_baroclinic_wave_rhoe}}

function make_plots(::DryBaroWavePlots, simulation_path)
    simdir = SimDir(simulation_path)
    short_names, reduction = ["pfull", "va", "wa", "rv"], "inst"
    vars = [get(simdir; short_name, reduction) for short_name in short_names]
    make_plots_generic(simulation_path, vars, z = 1500, time = LAST_SNAP)
end

function make_plots(
    ::Val{:sphere_baroclinic_wave_rhoe_topography_dcmip_rs},
    simulation_path,
)
    simdir = SimDir(simulation_path)
    short_names, reduction = ["pfull", "va", "wa", "rv"], "inst"
    vars = [get(simdir; short_name, reduction) for short_name in short_names]
    make_plots_generic(
        simulation_path,
        vars,
        z_reference = 1500,
        time = LAST_SNAP,
    )
end

function make_plots(::Val{:longrun_bw_rhoe_highres}, simulation_path)
    simdir = SimDir(simulation_path)
    short_names = ["pfull", "va", "wa", "rv"]
    vars = [get(simdir; short_name) for short_name in short_names]
    make_plots_generic(simulation_path, vars, z = 1500, time = 10days)
end

function make_plots(
    ::Val{:sphere_baroclinic_wave_rhoe_equilmoist},
    simulation_path,
)
    simdir = SimDir(simulation_path)
    short_names, reduction = ["pfull", "va", "wa", "rv", "hus"], "inst"
    vars = [get(simdir; short_name, reduction) for short_name in short_names]
    make_plots_generic(simulation_path, vars, z = 1500, time = LAST_SNAP)
end

function make_plots(
    ::Val{:sphere_baroclinic_wave_rhoe_equilmoist_expvdiff},
    simulation_path,
)
    simdir = SimDir(simulation_path)
    short_names, reduction = ["ta", "hus"], "average"
    vars = [
        get(simdir; short_name, reduction) |> ClimaAnalysis.average_lon for
        short_name in short_names
    ]
    make_plots_generic(
        simulation_path,
        vars,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
end

LongMoistBaroWavePlots = Union{
    Val{:longrun_bw_rhoe_equil_highres},
    Val{:longrun_zalesak_tracer_energy_bw_rhoe_equil_highres},
    Val{:longrun_ssp_bw_rhoe_equil_highres},
    Val{:longrun_bw_rhoe_equil_highres_topography_earth},
}

function make_plots(::LongMoistBaroWavePlots, simulation_path)
    simdir = SimDir(simulation_path)
    short_names = ["pfull", "va", "wa", "rv", "hus"]
    vars = [get(simdir; short_name) for short_name in short_names]
    make_plots_generic(simulation_path, vars, z = 1500, time = 10days)
end

DryHeldSuarezPlots = Union{
    Val{:sphere_held_suarez_rhoe_hightop},
    Val{:longrun_sphere_hydrostatic_balance_rhoe},
    Val{:longrun_hs_rhoe_dry_55km_nz63},
}

function make_plots(::DryHeldSuarezPlots, simulation_path)
    simdir = SimDir(simulation_path)

    short_names, reduction = ["ua", "ta"], "average"
    vars = [
        get(simdir; short_name, reduction) |> ClimaAnalysis.average_lon for
        short_name in short_names
    ]
    make_plots_generic(
        simulation_path,
        vars,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
end

MoistHeldSuarezPlots = Union{
    Val{:sphere_baroclinic_wave_rhoe_equilmoist_impvdiff},
    Val{:sphere_held_suarez_rhoe_equilmoist_hightop_sponge},
    Val{:longrun_hs_rhoe_equil_55km_nz63_0M},
}

function make_plots(::MoistHeldSuarezPlots, simulation_path)
    simdir = SimDir(simulation_path)

    short_names_3D, reduction = ["ua", "ta", "hus"], "average"
    short_names_sfc = ["hfes", "evspsbl"]
    vars_3D = [
        get(simdir; short_name, reduction) |> ClimaAnalysis.average_lon for
        short_name in short_names_3D
    ]
    vars_sfc =
        [get(simdir; short_name, reduction) for short_name in short_names_sfc]
    make_plots_generic(
        simulation_path,
        vars_3D,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
    make_plots_generic(
        simulation_path,
        vars_sfc,
        time = LAST_SNAP,
        output_name = "summary_sfc",
    )
end

function make_plots(
    ::Val{:sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric},
    simulation_path,
)
    simdir = SimDir(simulation_path)

    reduction = "average"
    short_names_3D = ["ua", "ta", "hus", "rsd", "rsu", "rld", "rlu"]
    short_names_sfc = ["hfes", "evspsbl"]
    available_periods = ClimaAnalysis.available_periods(
        simdir;
        short_name = short_names_3D[1],
        reduction,
    )
    if "10d" in available_periods
        period = "10d"
    elseif "1d" in available_periods
        period = "1d"
    elseif "12h" in available_periods
        period = "12h"
    end
    vars_3D = [
        get(simdir; short_name, reduction, period) |> ClimaAnalysis.average_lon for short_name in short_names_3D
    ]
    vars_sfc = [
        get(simdir; short_name, reduction, period) for
        short_name in short_names_sfc
    ]
    make_plots_generic(
        simulation_path,
        vars_3D,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
    make_plots_generic(
        simulation_path,
        vars_sfc,
        time = LAST_SNAP,
        output_name = "summary_sfc",
    )
end

function make_plots(
    ::Union{
        Val{:aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean},
        Val{:longrun_aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean},
        Val{
            :longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_tvinsol_0M_slabocean,
        },
    },
    simulation_path,
)
    simdir = SimDir(simulation_path)


    reduction = "average"
    period = "10d"
    short_names_3D = [
        "ta",
        "thetaa",
        "rhoa",
        "ua",
        "va",
        "wa",
        "hur",
        "hus",
        "clw",
        "cli",
        "rsd",
        "rsu",
        "rld",
        "rlu",
    ]
    short_names_sfc = ["hfes", "evspsbl", "ts"]
    vars_3D = [
        get(simdir; short_name, reduction, period) |> ClimaAnalysis.average_lon for short_name in short_names_3D
    ]
    vars_sfc = [
        get(simdir; short_name, reduction, period) for
        short_name in short_names_sfc
    ]
    make_plots_generic(
        simulation_path,
        vars_3D,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
    make_plots_generic(
        simulation_path,
        vars_sfc,
        time = LAST_SNAP,
        output_name = "summary_sfc",
    )
end

AquaplanetPlots = Union{
    Val{:sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res},
    Val{:mpi_sphere_aquaplanet_rhoe_equilmoist_clearsky},
    Val{:longrun_aquaplanet_rhoe_equil_55km_nz63_gray_0M},
    Val{:longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_0M},
    Val{:longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_diagedmf_diffonly_0M},
    Val{:longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_diagedmf_0M},
    Val{:longrun_aquaplanet_rhoe_equil_55km_nz63_allsky_diagedmf_0M},
    Val{:longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_tvinsol_0M_earth},
    Val{:longrun_aquaplanet_rhoe_equil_highres_allsky_ft32},
    Val{:longrun_aquaplanet_dyamond},
    Val{:longrun_aquaplanet_amip},
}

function make_plots(::AquaplanetPlots, simulation_path)
    simdir = SimDir(simulation_path)

    reduction = "average"
    short_names_3D = ["ua", "ta", "hus", "rsd", "rsu", "rld", "rlu"]
    short_names_sfc = ["hfes", "evspsbl"]
    available_periods = ClimaAnalysis.available_periods(
        simdir;
        short_name = short_names_3D[1],
        reduction,
    )
    if "10d" in available_periods
        period = "10d"
    elseif "1d" in available_periods
        period = "1d"
    elseif "12h" in available_periods
        period = "12h"
    end
    vars_3D = [
        get(simdir; short_name, reduction, period) |> ClimaAnalysis.average_lon for short_name in short_names_3D
    ]
    vars_sfc =
        [get(simdir; short_name, reduction) for short_name in short_names_sfc]
    make_plots_generic(
        simulation_path,
        vars_3D,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
    make_plots_generic(
        simulation_path,
        vars_sfc,
        time = LAST_SNAP,
        output_name = "summary_sfc",
    )
end


EDMFBoxPlots = Union{
    Val{:diagnostic_edmfx_gabls_box},
    Val{:diagnostic_edmfx_bomex_box},
    Val{:diagnostic_edmfx_bomex_stretched_box},
    Val{:diagnostic_edmfx_dycoms_rf01_box},
    Val{:diagnostic_edmfx_rico_box},
    Val{:diagnostic_edmfx_trmm_box},
    Val{:diagnostic_edmfx_trmm_stretched_box},
    Val{:diagnostic_edmfx_dycoms_rf01_explicit_box},
    Val{:prognostic_edmfx_adv_test_box},
    Val{:prognostic_edmfx_gabls_box},
    Val{:prognostic_edmfx_bomex_fixtke_box},
    Val{:prognostic_edmfx_bomex_box},
    Val{:prognostic_edmfx_bomex_stretched_box},
    Val{:prognostic_edmfx_dycoms_rf01_box},
    Val{:prognostic_edmfx_rico_column},
    Val{:prognostic_edmfx_trmm_column},
}

"""
    plot_edmf_vert_profile!(grid_loc, var_group)

Helper function for `make_plots_generic`. Takes a list of variables and plots
them on the same axis.
"""
function plot_edmf_vert_profile!(grid_loc, var_group)
    z = var_group[1].dims["z"]
    units = var_group[1].attributes["units"]
    ax = CairoMakie.Axis(
        grid_loc[1, 1],
        ylabel = "z [$(var_group[1].dim_attributes["z"]["units"])]",
        xlabel = "$(short_name(var_group[1])) [$units]",
        title = parse_var_attributes(var_group[1]),
    )
    for var in var_group
        CairoMakie.lines!(ax, var.data, z, label = short_name(var))
    end
    length(var_group) > 1 && Makie.axislegend(ax)
end


"""
    plot_parsed_attribute_title!(grid_loc, var)

Helper function for `make_plots_generic`. Plots an OutputVar `var`,
setting the axis title to `parse_var_attributes(var)`
"""
plot_parsed_attribute_title!(grid_loc, var) = viz.plot!(
    grid_loc,
    var;
    more_kwargs = Dict(:axis => ca_kwargs(title = parse_var_attributes(var))),
)

"""
    pair_edmf_names(vars)

Groups updraft and gridmean EDMF short names into tuples. 
Matches on the same variable short name with the suffix "up".
This assumes that the updraft variable name is the same as the corresponding
gridmean variable with the suffix "up".
"""
function pair_edmf_names(short_names)
    grouped_vars = Any[]
    short_names_to_be_processed = Set(short_names)

    for name in short_names
        # If we have already visited this name, go to the next one
        name in short_names_to_be_processed || continue

        # First, check if we have the pair of variables
        # We normalize the name to the gridmean version (base_name)
        # So, if we are visiting "va" or "vaup", we end up with 
        # base_name = "va" and up_name = "vaup"
        base_name = replace(name, "up" => "")
        up_name = base_name * "up"

        if base_name in short_names_to_be_processed &&
           up_name in short_names_to_be_processed
            # Gridmean and updraft are available
            tuple_to_be_added = (base_name, up_name)
        else
            # Only single var (updraft OR gridmean) is available
            tuple_to_be_added = (name,)
        end

        foreach(n -> delete!(short_names_to_be_processed, n), tuple_to_be_added)
        push!(grouped_vars, tuple_to_be_added)
    end
    return grouped_vars
end

function make_plots(::EDMFBoxPlots, simulation_path)
    simdir = SimDir(simulation_path)

    short_names = [
        "ua",
        "wa",
        "thetaa",
        "thetaaup",
        "ta",
        "taup",
        "ha",
        "haup",
        "waup",
        "tke",
        "arup",
        "hus",
        "husup",
        "hur",
        "hurup",
        "cl",
        "clw",
        "clwup",
        "cli",
        "cliup",
    ]
    reduction = "inst"
    period = "30m"

    short_name_tuples = pair_edmf_names(short_names)
    var_groups_zt = [
        (
            slice(get(simdir; short_name, reduction, period), x = 0.0, y = 0.0) for short_name in var_names
        ) for var_names in short_name_tuples
    ]
    var_groups_z = [
        ([slice(v, time = LAST_SNAP) for v in group]...,) for
        group in var_groups_zt
    ]

    tmp_file = make_plots_generic(
        simulation_path,
        output_name = "tmp",
        var_groups_z;
        plot_fn = plot_edmf_vert_profile!,
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )

    make_plots_generic(
        simulation_path,
        vcat((var_groups_zt...)...),
        plot_fn = plot_parsed_attribute_title!,
        summary_files = [tmp_file],
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )
end

EDMFSpherePlots =
    Union{Val{:diagnostic_edmfx_aquaplanet}, Val{:prognostic_edmfx_aquaplanet}}

function make_plots(::EDMFSpherePlots, simulation_path)
    simdir = SimDir(simulation_path)

    short_names =
        ["ua", "wa", "waup", "thetaa", "ta", "taup", "haup", "tke", "arup"]
    reduction = "average"
    period = "1h"
    latitudes = [0.0, 30.0, 60.0, 90.0]

    short_name_tuples = pair_edmf_names(short_names)
    var_groups_zt = [
        (
            slice(
                get(simdir; short_name, reduction, period),
                lon = 0.0,
                lat = lat,
            ) for short_name in var_names
        ) for lat in latitudes, var_names in short_name_tuples
    ]
    var_groups_z = [
        ([slice(v, time = LAST_SNAP) for v in group]...,) for
        group in var_groups_zt
    ]

    tmp_file = make_plots_generic(
        simulation_path,
        output_name = "tmp",
        var_groups_z;
        plot_fn = plot_edmf_vert_profile!,
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )
    make_plots_generic(
        simulation_path,
        vcat((var_groups_zt...)...),
        plot_fn = plot_parsed_attribute_title!,
        summary_files = [tmp_file],
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )
end
