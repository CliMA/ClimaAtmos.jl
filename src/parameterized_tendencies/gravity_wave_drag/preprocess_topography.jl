import ClimaDiagnostics.Writers:
    HDF5Writer
import ClimaAtmos as CA
import ClimaCore: InputOutput, Spaces, Fields, to_cpu
import ClimaComms

# Save the computed drag data to a NetCDF file for diagnostics
using NCDatasets
using ClimaCoreTempestRemap # for apply_remap

include(
    joinpath(pkgdir(CA), "post_processing/remap", "remap_helpers.jl"),
)
include(
    joinpath(pkgdir(CA), "test/parameterized_tendencies/gravity_wave", "gw_plotutils.jl"))

function write_computed_drag!(computed_drag, parsed_args, config)
    (; output_filename, topography, topo_smoothing, topo_damping_factor, h_elem) =
        CA.gen_fn(parsed_args)
    # initialize HDF5 output
    hdfwriter = InputOutput.HDF5Writer("$(output_filename).hdf5", config.comms_ctx)

    # write attributes to the HDF5 file 
    InputOutput.HDF5.write_attribute(hdfwriter.file, "topography", topography)
    InputOutput.HDF5.write_attribute(hdfwriter.file, "topo_smoothing", topo_smoothing)
    InputOutput.HDF5.write_attribute(
        hdfwriter.file,
        "topo_damping_factor",
        topo_damping_factor,
    )
    InputOutput.HDF5.write_attribute(hdfwriter.file, "h_elem", h_elem)

    @info "Writing computed drag data to $(output_filename).hdf5"
    # write computed drag data to the HDF5 file
    InputOutput.write!(hdfwriter, computed_drag, "computed_drag")

    # close the HDF5 writer
    Base.close(hdfwriter)

    return output_filename
end


function save_nc_data(output_filename, topo_cg, spaces)
    topo_cg = to_cpu(topo_cg)
    FT = eltype(topo_cg.hmax)
    hspace = to_cpu(Spaces.horizontal_space(spaces.center_space))
    center_space = to_cpu(spaces.center_space)
    face_space = to_cpu(spaces.face_space)

    datafile_cg = joinpath(@__DIR__, "$(output_filename).nc")
    nc = NCDataset(datafile_cg, "c")
    def_space_coord(nc, center_space, type = "cgll")
    nc_time = def_time_coord(nc)
    nc_hmax = defVar(nc, "hmax", FT, hspace, ("time",))
    nc_hmin = defVar(nc, "hmin", FT, hspace, ("time",))
    nc_t11 = defVar(nc, "t11", FT, hspace, ("time",))
    nc_t12 = defVar(nc, "t12", FT, hspace, ("time",))
    nc_t21 = defVar(nc, "t21", FT, hspace, ("time",))
    nc_t22 = defVar(nc, "t22", FT, hspace, ("time",))

    nc_time[1] = 1
    nc_hmax[:, 1] = topo_cg.hmax
    nc_hmin[:, 1] = topo_cg.hmin
    nc_t11[:, 1] = topo_cg.t11
    nc_t12[:, 1] = topo_cg.t12
    nc_t21[:, 1] = topo_cg.t21
    nc_t22[:, 1] = topo_cg.t22
    close(nc)

    nlat = 90
    nlon = 180
    weightfile = joinpath(@__DIR__, "remap_weights.nc")
    create_weightfile(weightfile, center_space, face_space, nlat, nlon)

    return datafile_cg, weightfile
end

function remap_nc_data(output_filename)
    # remap from clima grid to lat/lon grid
    datafile_rll = joinpath(@__DIR__, "data_rll_$(output_filename).nc")
    apply_remap(
        datafile_rll,
        datafile_cg,
        weightfile,
        ["hmax", "hmin", "t11", "t12", "t21", "t22"],
    )

    return datafile_rll
end

function diagnostics(datafile_rll)
    nt = NCDataset(datafile_rll) do ds
        lon = Array(ds["lon"])
        lat = Array(ds["lat"])
        hmax = Array(ds["hmax"])
        hmin = Array(ds["hmin"])
        t11 = Array(ds["t11"])
        t12 = Array(ds["t12"])
        t21 = Array(ds["t21"])
        t22 = Array(ds["t22"])
        (; lon, lat, hmax, hmin, t11, t12, t21, t22)
    end

    return nt
end

function plot_diagnostics(lon, lat, hmax, hmin, t11, t12, t21, t22)
    @. hmax = max(0, hmax)
    @. hmin = max(0, hmin)

    ENV["GKSwstype"] = "nul"
    output_dir = joinpath(@__DIR__, "preprocess_topography")
    mkpath(output_dir)

    fig = generate_empty_figure()
    title = "hmax"
    create_plot!(
        fig;
        X = lon,
        Y = lat,
        Z = hmax[:, :, 1],
        levels = range(minimum(hmax), maximum(hmax); length = 10),
        title,
        p_loc = (1, 1),
        yreversed = false,
    )
    title = "t11"
    create_plot!(
        fig;
        X = lon,
        Y = lat,
        Z = t11[:, :, 1],
        levels = range(-50, 30; length = 40),
        title,
        p_loc = (2, 1),
        yreversed = false,
    )
    title = "t21"
    create_plot!(
        fig;
        X = lon,
        Y = lat,
        Z = t21[:, :, 1],
        levels = range(-50, 30; length = 40),
        title,
        p_loc = (3, 1),
        yreversed = false,
    )
    title = "hmin"
    create_plot!(
        fig;
        X = lon,
        Y = lat,
        Z = hmin[:, :, 1],
        levels = range(minimum(hmin), maximum(hmin); length = 10),
        title,
        p_loc = (1, 2),
        yreversed = false,
    )
    title = "t12"
    create_plot!(
        fig;
        X = lon,
        Y = lat,
        Z = t12[:, :, 1],
        levels = range(-50, 30; length = 40),
        title,
        p_loc = (2, 2),
        yreversed = false,
    )
    title = "t22"
    create_plot!(
        fig;
        X = lon,
        Y = lat,
        Z = t22[:, :, 1],
        levels = range(-50, 30; length = 40),
        title,
        p_loc = (3, 2),
        yreversed = false,
    )

    CairoMakie.save(joinpath(output_dir, "diagnostics.png"), fig)
end
