import CairoMakie
import ClimaAnalysis
import ClimaAnalysis: Visualize as viz
import ClimaAnalysis: SimDir, slice_time, slice

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

# From https://github.com/scheidan/PDFmerger.jl/blob/main/src/PDFmerger.jl
# Licensed under MIT license
import Base.Filesystem
using Poppler_jll: pdfunite, pdfinfo, pdfseparate
function merge_pdfs(
    files::Vector{T},
    destination::AbstractString = joinpath(common_dirname(files), "merged.pdf");
    cleanup::Bool = false,
) where {T <: AbstractString}
    if destination ∈ files
        # rename existing file
        Filesystem.mv(destination, destination * "_x_")
        files[files .== destination] .= destination * "_x_"
    end

    # Merge large number of files iteratively, because there
    # is a (OS dependent) limit how many files 'pdfunit' can handle at once.
    # See: https://gitlab.freedesktop.org/poppler/poppler/-/issues/334
    filemax = 200

    k = 1
    for files_part in Base.Iterators.partition(files, filemax)
        if k == 1
            outfile_tmp2 = "_temp_destination_$k"

            pdfunite() do unite
                run(`$unite $files_part $outfile_tmp2`)
            end
        else
            outfile_tmp1 = "_temp_destination_$(k-1)"
            outfile_tmp2 = "_temp_destination_$k"

            pdfunite() do unite
                run(`$unite $outfile_tmp1 $files_part $outfile_tmp2`)
            end
        end
        k += 1
    end

    # rename last file
    Filesystem.mv("_temp_destination_$(k-1)", destination, force = true)

    # remove temp files
    Filesystem.rm(destination * "_x_", force = true)
    Filesystem.rm.("_temp_destination_$(i)" for i in 1:(k - 2); force = true)
    if cleanup
        Filesystem.rm.(files, force = true)
    end

    destination
end


function make_plots(sim, simulation_path)
    @warn "No plot found for $sim"
end

# The contour plot functions in ClimaAnalysis work by finding the nearest slice available.
# If we want the extremes, we can just ask for the slice closest to a very large number.
const LARGE_NUM = typemax(Int)
const LAST_SNAP = LARGE_NUM
const FIRST_SNAP = -LARGE_NUM
const BOTTOM_LVL = -LARGE_NUM
const TOP_LVL = LARGE_NUM
# Shorthand for logscale on y axis and to move the dimension to the y axis on line plots
# (because they are columns)
const YLOGSCALE =
    Dict(:axis => ClimaAnalysis.Utils.kwargs(yscale = log10, dim_on_y = true))

function make_plots_generic(
    output_path,
    vars,
    args...;
    output_name = "summary",
    kwargs...,
)
    MAX_PLOTS_PER_PAGE = 4
    vars_left_to_plot = length(vars)

    # Define fig, and p_loc, used below. (Needed for scope)
    fig = CairoMakie.Figure(
        resolution = (900, 300 * min(vars_left_to_plot, MAX_PLOTS_PER_PAGE)),
    )
    p_loc = [1, 1]
    page = 1
    summary_files = String[]

    for (var_num, var) in enumerate(vars)
        # Create a new page if this is the first plot
        if mod(var_num, MAX_PLOTS_PER_PAGE) == 1
            fig = CairoMakie.Figure(
                resolution = (
                    900,
                    300 * min(vars_left_to_plot, MAX_PLOTS_PER_PAGE),
                ),
            )
            p_loc = [1, 1]
        end
        viz.plot!(fig, var, args...; p_loc, kwargs...)

        p_loc[1] += 1

        # Flush current page
        if p_loc[1] > min(MAX_PLOTS_PER_PAGE, vars_left_to_plot)
            file_path = joinpath(output_path, "$(output_name)_$page.pdf")
            CairoMakie.save(file_path, fig)
            push!(summary_files, file_path)
            vars_left_to_plot -= MAX_PLOTS_PER_PAGE
            page += 1
        end
    end

    merge_pdfs(
        summary_files,
        joinpath(output_path, "$output_name.pdf"),
        cleanup = true,
    )

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
    short_names, reduction, period = ["ta", "wa"], "average", "1d"
    vars = [
        get(simdir; short_name, reduction, period) for short_name in short_names
    ]
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
    short_names = ["wa", "ua"]
    vars = [get(simdir; short_name) for short_name in short_names]
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
    hus, clw, cli, husra, hussn, ta = [
        slice(get(simdir; short_name), x = 0.0, y = 0.0) for
        short_name in short_names
    ]

    z_units = hus.dim_attributes["z"]["units"]
    z = hus.dims["z"]

    hus_units = hus.attributes["units"]
    clw_units = clw.attributes["units"]
    cli_units = cli.attributes["units"]
    husra_units = husra.attributes["units"]
    hussn_units = hussn.attributes["units"]
    ta_units = ta.attributes["units"]

    fig = CairoMakie.Figure(resolution = (1200, 600))
    ax1 = CairoMakie.Axis(
        fig[1, 1],
        ylabel = "z [$z_units]",
        xlabel = "hus [$hus_units]",
    )
    ax4 = CairoMakie.Axis(
        fig[2, 1],
        ylabel = "z [$z_units]",
        xlabel = "ta [$ta_units]",
    )
    ax2 = CairoMakie.Axis(fig[1, 2], xlabel = "q_liq [$clw_units]")
    ax3 = CairoMakie.Axis(fig[1, 3], xlabel = "q_ice [$cli_units]")
    ax5 = CairoMakie.Axis(fig[2, 2], xlabel = "q_rai [$husra_units]")
    ax6 = CairoMakie.Axis(fig[2, 3], xlabel = "q_sno [$hussn_units]")

    col = Dict(0 => :navy, 500 => :blue2, 1000 => :royalblue, 1500 => :skyblue1)

    for (time, color) in col
        CairoMakie.lines!(ax1, slice(hus; time).data, z, color = color)
        CairoMakie.lines!(ax2, slice(clw; time).data, z, color = color)
        CairoMakie.lines!(ax3, slice(cli; time).data, z, color = color)
        CairoMakie.lines!(ax4, slice(ta; time).data, z, color = color)
        CairoMakie.lines!(ax5, slice(husra; time).data, z, color = color)
        CairoMakie.lines!(ax6, slice(hussn; time).data, z, color = color)
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
    vars = [get(simdir, short_name = "wa")]
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
    short_names, reduction, period = ["ua", "wa"], "average", "1d"
    vars = [
        get(simdir; short_name, reduction, period) |> ClimaAnalysis.average_lon for short_name in short_names
    ]
    make_plots_generic(
        simulation_path,
        vars,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
end

DryBaroWavePlots = Union{
    Val{:sphere_baroclinic_wave_rhoe},
    Val{:sphere_baroclinic_wave_rhoe_topography_dcmip_rs},
    Val{:longrun_bw_rhoe_highres},
}

function make_plots(::DryBaroWavePlots, simulation_path)
    simdir = SimDir(simulation_path)
    short_names = ["pfull", "va", "wa", "rv"]
    vars = [get(simdir; short_name) for short_name in short_names]
    make_plots_generic(simulation_path, vars, z = 3000, time = LAST_SNAP)
end

function make_plots(
    ::Val{:sphere_baroclinic_wave_rhoe_equilmoist},
    simulation_path,
)
    simdir = SimDir(simulation_path)
    short_names = ["pfull", "va", "wa", "rv", "hus"]
    vars = [get(simdir; short_name) for short_name in short_names]
    make_plots_generic(simulation_path, vars, z = 3000, time = LAST_SNAP)
end

MoistBaroWavePlots = Union{
    Val{:sphere_baroclinic_wave_rhoe_equilmoist_expvdiff},
    Val{:sphere_baroclinic_wave_rhoe_equilmoist_impvdiff},
    Val{:longrun_zalesak_tracer_energy_bw_rhoe_equil_highres},
    Val{:longrun_ssp_bw_rhoe_equil_highres},
    Val{:longrun_bw_rhoe_equil_highres_topography_earth},
    Val{:longrun_bw_rhoe_equil_highres},
}

function make_plots(::MoistBaroWavePlots, simulation_path)
    simdir = SimDir(simulation_path)

    var1 = get(simdir; short_name = "ta")

    var1sliced = slice(var1, z = BOTTOM_LVL)
    var2 = get(simdir; short_name = "hus") |> ClimaAnalysis.average_lon

    make_plots_generic(simulation_path, [var1sliced, var2], time = LAST_SNAP)
end

DryHeldSuarezPlots = Union{
    Val{:sphere_held_suarez_rhoe_hightop},
    Val{:longrun_sphere_hydrostatic_balance_rhoe},
    Val{:longrun_hs_rhoe_dry_nz63_55km_rs35km},
}

function make_plots(::DryHeldSuarezPlots, simulation_path)
    simdir = SimDir(simulation_path)

    short_names, reduction, period = ["ua", "ta"], "average", "1d"
    vars = [
        get(simdir; short_name, reduction, period) |> ClimaAnalysis.average_lon for short_name in short_names
    ]
    make_plots_generic(
        simulation_path,
        vars,
        time = LAST_SNAP,
        more_kwargs = YLOGSCALE,
    )
end

MoistHeldSuarezPlots = Union{
    Val{:sphere_held_suarez_rhoe_equilmoist_hightop_sponge},
    Val{:sphere_held_suarez_rhoe_equilmoist_topography_dcmip},
    Val{:longrun_hs_rhoe_equil_highres_topography_earth},
}

function make_plots(::MoistHeldSuarezPlots, simulation_path)
    simdir = SimDir(simulation_path)

    short_names_3D, reduction, period = ["ua", "ta", "hus"], "average", "1d"
    short_names_sfc = ["hfes", "evspsbl"]
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
    ::Val{:sphere_held_suarez_rhoe_topography_dcmip},
    simulation_path,
)
    simdir = SimDir(simulation_path)

    short_names_3D, reduction, period = ["ua", "ta"], "average", "1d"
    short_names_sfc = ["hfes"]
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
    Val{:sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric},
    Val{:longrun_aquaplanet_rhoe_equil_gray_55km_nz63_0M},
    Val{
        :longrun_aquaplanet_rhoe_equilmoist_nz63_0M_55km_rs35km_clearsky_tvinsolation,
    },
    Val{
        :longrun_aquaplanet_rhoe_equilmoist_nz63_0M_55km_rs35km_clearsky_tvinsolation_earth,
    },
    Val{:longrun_aquaplanet_rhoe_equil_highres_clearsky_ft32_earth},
    Val{:longrun_aquaplanet_rhoe_equil_highres_allsky_ft32},
    Val{:longrun_aquaplanet_dyamond},
    Val{:longrun_aquaplanet_amip},
    Val{:longrun_hs_rhoe_equilmoist_nz63_0M_55km_rs35km},
}

function make_plots(::AquaplanetPlots, simulation_path)
    simdir = SimDir(simulation_path)

    reduction = "average"
    period = "12.0h"
    short_names_3D = ["ua", "ta", "hus", "rsd", "rsu", "rld", "rlu"]
    short_names_sfc = ["hfes", "evspsbl"]
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
    ::Val{:mpi_sphere_aquaplanet_rhoe_equilmoist_clearsky},
    simulation_path,
)
    simdir = SimDir(simulation_path)

    reduction = "average"
    period = "1d"
    short_names_3D = ["ua", "ta", "hus", "rsd", "rsu", "rld", "rlu"]
    short_names_sfc = ["hfes", "evspsbl"]
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

EDMFBoxPlots = Union{
    Val{:diagnostic_edmfx_gabls_box},
    Val{:diagnostic_edmfx_bomex_box},
    Val{:diagnostic_edmfx_bomex_stretched_box},
    Val{:diagnostic_edmfx_dycoms_rf01_box},
    Val{:diagnostic_edmfx_rico_box},
    Val{:diagnostic_edmfx_trmm_box},
    Val{:diagnostic_edmfx_trmm_stretched_box},
    Val{:diagnostic_edmfx_aquaplanet_tke},
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


function make_plots(::EDMFBoxPlots, simulation_path)
    simdir = SimDir(simulation_path)

    short_names = ["ua", "wa", "thetaa", "taup", "haup", "waup", "tke", "arup"]
    reduction = "average"
    period = "10m"
    vars = [
        get(simdir; short_name, reduction, period) for short_name in short_names
    ]
    vars_zt = [slice(var, x = 0.0, y = 0.0) for var in vars]
    vars_z = [slice(var, x = 0.0, y = 0.0, time = LAST_SNAP) for var in vars]
    make_plots_generic(
        simulation_path,
        [vars_zt..., vars_z...],
        more_kwargs = YLOGSCALE,
    )
end

EDMFSpherePlots =
    Union{Val{:diagnostic_edmfx_aquaplanet}, Val{:prognostic_edmfx_aquaplanet}}

function make_plots(::EDMFSpherePlots, simulation_path)
    simdir = SimDir(simulation_path)

    short_names = ["ua", "wa", "thetaa", "taup", "haup", "waup", "tke", "arup"]
    reduction = "average"
    period = "10m"
    vars = [
        get(simdir; short_name, reduction, period) for short_name in short_names
    ]
    vars_zt0_0 = [slice(var, lon = 0.0, lat = 0.0) for var in vars]
    vars_zt30_0 = [slice(var, lon = 0.0, lat = 30.0) for var in vars]
    vars_zt60_0 = [slice(var, lon = 0.0, lat = 60.0) for var in vars]
    vars_zt90_0 = [slice(var, lon = 0.0, lat = 90.0) for var in vars]
    vars_zt = [vars_zt0_0..., vars_zt30_0..., vars_zt60_0..., vars_zt90_0...]
    vars_z = [slice(var, time = LAST_SNAP) for var in vars_zt]

    make_plots_generic(
        simulation_path,
        [vars_zt..., vars_z...],
        more_kwargs = YLOGSCALE,
    )
end
