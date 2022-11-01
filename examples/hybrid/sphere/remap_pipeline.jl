#=
Sbatch script example:
# Serial remapping
```
#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=remap
#SBATCH --time=01:00:00
#SBATCH --output=remap.log
module purge
module load julia/1.8.2
export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export HDF5_DIR=/central/groups/esm/{username}/ClimaAtmos_remap/
CA_EXAMPLE=$HOME'/Documents/ClimaAtmos.jl/examples/'
DRIVER=$CA_EXAMPLE'hybrid/sphere/remap_pipeline.jl'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.instantiate()'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.build("HDF5")'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.API.precompile()'
julia --project=$CA_EXAMPLE $DRIVER
```
=#

#=
Here's a sketch of what this pipeline does:

Inputs:
 - data_dir/day0.0.1.hdf5
 - data_dir/day0.2.0.hdf5
 - data_dir/day0.3.0.hdf5
 - ...
 - tmpdir/weightsfile.nc
Apply remap pipeline:
 - tmpdir[/myid()]/test.nc -> remap -> out_dir/day0.0.1.nc, rm(tmpdir[/myid()]/test.nc)
 - tmpdir[/myid()]/test.nc -> remap -> out_dir/day0.2.0.nc, rm(tmpdir[/myid()]/test.nc)
 - tmpdir[/myid()]/test.nc -> remap -> out_dir/day0.3.0.nc, rm(tmpdir[/myid()]/test.nc)
rm(tmpdir/weightsfile.nc)
=#

import ClimaCore
import ClimaAtmos
using ClimaCore:
    Geometry, Meshes, Domains, Topologies, Spaces, Operators, InputOutput
using NCDatasets
using ClimaCoreTempestRemap

include(joinpath(pkgdir(ClimaAtmos), "examples", "hybrid", "remap_helpers.jl"))

function create_weightfile(filein, remap_tmpdir, nlat, nlon)
    @assert endswith(filein, "hdf5")
    reader = InputOutput.HDF5Reader(filein)
    Y = InputOutput.read_field(reader, "Y")
    weightfile = joinpath(remap_tmpdir, "remap_weights.nc")
    create_weightfile(weightfile, axes(Y.c), axes(Y.f), nlat, nlon)
    return weightfile
end

function remap2latlon(filein, data_dir, remap_tmpdir, weightfile, nlat, nlon)
    @assert endswith(filein, "hdf5")
    reader = InputOutput.HDF5Reader(filein)
    Y = InputOutput.read_field(reader, "Y")
    diag = InputOutput.read_field(reader, "diagnostics")
    t_now = InputOutput.HDF5.read_attribute(reader.file, "time")

    remap_tmpsubdir = if @isdefined pmap
        subdir = joinpath(remap_tmpdir, string(myid()))
        mkpath(subdir)
        subdir
    else
        remap_tmpdir
    end

    # float type
    FT = eltype(Y)
    ᶜinterp = Operators.InterpolateF2C()

    # reconstruct space
    cspace = axes(Y.c)
    fspace = axes(Y.f)
    hspace = cspace.horizontal_space

    ### create an nc file to store raw cg data
    # create data
    datafile_cc = joinpath(remap_tmpsubdir, "test.nc")
    nc = NCDataset(datafile_cc, "c")
    # defines the appropriate dimensions and variables for a space coordinate
    def_space_coord(nc, cspace, type = "cgll")
    def_space_coord(nc, fspace, type = "cgll")
    # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
    nc_time = def_time_coord(nc)
    # define variables for the prognostic states
    nc_rho = defVar(nc, "rho", FT, cspace, ("time",))
    thermo_var = if :ρe_tot in propertynames(Y.c)
        "e_tot"
    elseif :ρθ in propertynames(Y.c)
        "theta"
    else
        error("Unfound thermodynamic variable")
    end
    nc_thermo = defVar(nc, thermo_var, FT, cspace, ("time",))
    nc_u = defVar(nc, "u", FT, cspace, ("time",))
    nc_v = defVar(nc, "v", FT, cspace, ("time",))
    nc_w = defVar(nc, "w", FT, cspace, ("time",))
    # define variables for the diagnostic states
    nc_pres = defVar(nc, "pressure", FT, cspace, ("time",))
    nc_T = defVar(nc, "temperature", FT, cspace, ("time",))
    nc_θ = defVar(nc, "potential_temperature", FT, cspace, ("time",))
    nc_K = defVar(nc, "kinetic_energy", FT, cspace, ("time",))
    nc_vort = defVar(nc, "vorticity", FT, cspace, ("time",))
    nc_T_sfc = defVar(nc, "sfc_temperature", FT, hspace, ("time",))
    nc_qt_sfc = defVar(nc, "sfc_qt", FT, hspace, ("time",))
    # define moist variables
    if :ρq_tot in propertynames(Y.c)
        nc_qt = defVar(nc, "qt", FT, cspace, ("time",))
        nc_RH = defVar(nc, "RH", FT, cspace, ("time",))
        nc_cloudliq = defVar(nc, "cloud_liquid", FT, cspace, ("time",))
        nc_cloudice = defVar(nc, "cloud_ice", FT, cspace, ("time",))
        nc_watervapor = defVar(nc, "water_vapor", FT, cspace, ("time",))
        nc_precipitation_removal =
            defVar(nc, "precipitation_removal", FT, cspace, ("time",))
        nc_column_integrated_rain =
            defVar(nc, "column_integrated_rain", FT, hspace, ("time",))
        nc_column_integrated_snow =
            defVar(nc, "column_integrated_snow", FT, hspace, ("time",))
    end
    # define surface flux variables
    if :sfc_flux_energy in propertynames(diag)
        nc_sfc_flux_energy =
            defVar(nc, "sfc_flux_energy", FT, hspace, ("time",))
        nc_sfc_evaporation =
            defVar(nc, "sfc_evaporation", FT, hspace, ("time",))
        nc_sfc_flux_u = defVar(nc, "sfc_flux_u", FT, hspace, ("time",))
        nc_sfc_flux_v = defVar(nc, "sfc_flux_v", FT, hspace, ("time",))
    end
    # define radiative flux variables
    if :lw_flux_down in propertynames(diag)
        nc_lw_flux_down = defVar(nc, "lw_flux_down", FT, fspace, ("time",))
        nc_lw_flux_up = defVar(nc, "lw_flux_up", FT, fspace, ("time",))
        nc_sw_flux_down = defVar(nc, "sw_flux_down", FT, fspace, ("time",))
        nc_sw_flux_up = defVar(nc, "sw_flux_up", FT, fspace, ("time",))
    end
    if :clear_lw_flux_down in propertynames(diag)
        nc_clear_lw_flux_down =
            defVar(nc, "clear_lw_flux_down", FT, fspace, ("time",))
        nc_clear_lw_flux_up =
            defVar(nc, "clear_lw_flux_up", FT, fspace, ("time",))
        nc_clear_sw_flux_down =
            defVar(nc, "clear_sw_flux_down", FT, fspace, ("time",))
        nc_clear_sw_flux_up =
            defVar(nc, "clear_sw_flux_up", FT, fspace, ("time",))
    end

    # time
    nc_time[1] = t_now

    # reconstruct fields
    # density
    nc_rho[:, 1] = Y.c.ρ
    # thermodynamics
    if :ρe_tot in propertynames(Y.c)
        nc_thermo[:, 1] = Y.c.ρe_tot ./ Y.c.ρ
    elseif :ρθ in propertynames(Y.c)
        nc_thermo[:, 1] = Y.c.ρθ ./ Y.c.ρ
    end
    # physical horizontal velocity
    uh_phy = Geometry.transform.(Ref(Geometry.UVAxis()), Y.c.uₕ)
    nc_u[:, 1] = uh_phy.components.data.:1
    nc_v[:, 1] = uh_phy.components.data.:2
    # physical vertical velocity
    ᶠw_phy = Geometry.WVector.(Y.f.w)
    ᶜw_phy = ᶜinterp.(ᶠw_phy)
    nc_w[:, 1] = ᶜw_phy
    # diagnostic variables
    nc_pres[:, 1] = diag.pressure
    nc_T[:, 1] = diag.temperature
    nc_θ[:, 1] = diag.potential_temperature
    nc_K[:, 1] = diag.kinetic_energy
    nc_vort[:, 1] = diag.vorticity
    nc_T_sfc[:, 1] = diag.sfc_temperature
    nc_qt_sfc[:, 1] = diag.sfc_qt

    if :ρq_tot in propertynames(Y.c)
        nc_qt[:, 1] = Y.c.ρq_tot ./ Y.c.ρ
        nc_RH[:, 1] = diag.relative_humidity
        nc_cloudliq[:, 1] = diag.cloud_liquid
        nc_cloudice[:, 1] = diag.cloud_ice
        nc_watervapor[:, 1] = diag.water_vapor
        nc_precipitation_removal[:, 1] = diag.precipitation_removal
        nc_column_integrated_rain[:, 1] = diag.column_integrated_rain
        nc_column_integrated_snow[:, 1] = diag.column_integrated_snow
    end

    if :sfc_flux_energy in propertynames(diag)
        nc_sfc_flux_energy[:, 1] = diag.sfc_flux_energy.components.data.:1
        nc_sfc_evaporation[:, 1] = diag.sfc_evaporation.components.data.:1
        sfc_flux_momentum = diag.sfc_flux_momentum
        w_unit =
            Geometry.Covariant3Vector.(
                Geometry.WVector.(ones(axes(sfc_flux_momentum)))
            )
        sfc_flux_momentum_phy =
            Geometry.UVVector.(adjoint.(sfc_flux_momentum) .* w_unit)
        nc_sfc_flux_u[:, 1] = sfc_flux_momentum_phy.components.data.:1
        nc_sfc_flux_v[:, 1] = sfc_flux_momentum_phy.components.data.:2
    end

    if :lw_flux_down in propertynames(diag)
        nc_lw_flux_down[:, 1] = diag.lw_flux_down
        nc_lw_flux_up[:, 1] = diag.lw_flux_up
        nc_sw_flux_down[:, 1] = diag.sw_flux_down
        nc_sw_flux_up[:, 1] = diag.sw_flux_up
    end

    if :clear_lw_flux_down in propertynames(diag)
        nc_clear_lw_flux_down[:, 1] = diag.clear_lw_flux_down
        nc_clear_lw_flux_up[:, 1] = diag.clear_lw_flux_up
        nc_clear_sw_flux_down[:, 1] = diag.clear_sw_flux_down
        nc_clear_sw_flux_up[:, 1] = diag.clear_sw_flux_up
    end
    close(nc)


    datafile_latlon =
        joinpath(out_dir, first(splitext(basename(filein))) * ".nc")
    dry_variables = [
        "rho",
        thermo_var,
        "u",
        "v",
        "w",
        "pressure",
        "temperature",
        "potential_temperature",
        "kinetic_energy",
        "vorticity",
        "sfc_temperature",
        "sfc_qt",
    ]
    if :ρq_tot in propertynames(Y.c)
        moist_variables = [
            "qt",
            "RH",
            "cloud_ice",
            "cloud_liquid",
            "water_vapor",
            "precipitation_removal",
            "column_integrated_rain",
            "column_integrated_snow",
        ]
    else
        moist_variables = String[]
    end
    if :sfc_flux_energy in propertynames(diag)
        sfc_flux_variables =
            ["sfc_flux_energy", "sfc_evaporation", "sfc_flux_u", "sfc_flux_v"]
    else
        sfc_flux_variables = String[]
    end
    if :lw_flux_down in propertynames(diag)
        rad_flux_variables =
            ["lw_flux_down", "lw_flux_up", "sw_flux_down", "sw_flux_up"]
    else
        rad_flux_variables = String[]
    end
    if :clear_lw_flux_down in propertynames(diag)
        rad_flux_clear_variables = [
            "clear_lw_flux_down",
            "clear_lw_flux_up",
            "clear_sw_flux_down",
            "clear_sw_flux_up",
        ]
    else
        rad_flux_clear_variables = String[]
    end

    netcdf_variables = vcat(
        dry_variables,
        moist_variables,
        sfc_flux_variables,
        rad_flux_variables,
        rad_flux_clear_variables,
    )
    apply_remap(datafile_latlon, datafile_cc, weightfile, netcdf_variables)
    rm(datafile_cc)
end

import ArgParse
function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--data_dir"
        help = "Data directory"
        arg_type = String
        "--out_dir"
        help = "Output data directory"
        arg_type = String
        "--nlat"
        help = "Number of latitude points"
        arg_type = Int
        default = 90
        "--nlon"
        help = "Number of longitude points"
        arg_type = Int
        default = 180
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return (s, parsed_args)
end

function get_params()
    (s, parsed_args) = parse_commandline()
    data_dir = parsed_args["data_dir"]
    out_dir = parsed_args["out_dir"]
    nlat = parsed_args["nlat"]
    nlon = parsed_args["nlon"]
    if isnothing(out_dir)
        out_dir = joinpath(data_dir, "remap")
    end
    mkpath(out_dir)

    data_files =
        filter(x -> endswith(x, ".hdf5"), readdir(data_dir, join = true))

    remap_tmpdir = joinpath(out_dir, "remaptmp")
    mkpath(remap_tmpdir)
    return (; remap_tmpdir, data_files, out_dir, nlat, nlon)
end

# Only run this in serial mode:
parallel_mode = @isdefined pmap
if !parallel_mode
    (; remap_tmpdir, data_files, out_dir, nlat, nlon) = get_params()
    weightfile = create_weightfile(data_files[1], remap_tmpdir, nlat, nlon)
    map(data_files) do data_file
        remap2latlon(data_file, out_dir, remap_tmpdir, weightfile, nlat, nlon)
    end
    rm(remap_tmpdir; recursive = true)
end
