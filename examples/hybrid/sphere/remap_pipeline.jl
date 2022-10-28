import ClimaCore
using ClimaCore:
    Geometry, Meshes, Domains, Topologies, Spaces, Operators, InputOutput
using NCDatasets
using ClimaCoreTempestRemap

if haskey(ENV, "HDF5_DIR")
    data_dir = ENV["HDF5_DIR"]
else
    error("ENV[\"HDF5_DIR\"] require!")
end

if haskey(ENV, "THERMO_VAR")
    hs_thermo = ENV["THERMO_VAR"]
else
    error("ENV[\"THERMO_VAR\"] require (\"e_tot\" or \"theta\")")
end

if haskey(ENV, "NC_DIR")
    nc_dir = ENV["NC_DIR"]
else
    nc_dir = data_dir * "/nc/"
end
mkpath(nc_dir)

if haskey(ENV, "NLAT")
    nlat = NLAT
else
    nlat = 90
    println("NLAT is default to 90.")
end

if haskey(ENV, "NLON")
    nlon = NLON
else
    nlon = 180
    println("NLON is default to 180.")
end

const ᶜinterp = Operators.InterpolateF2C()

ext = ".hdf5"
data_files = filter(x -> endswith(x, ext), readdir(data_dir, join = true))

function create_weightfile(filein, nc_dir, nlat, nlon)
    if split(filein, ".")[end] == "hdf5"
        reader = InputOutput.HDF5Reader(filein)
        Y = InputOutput.read_field(reader, "Y")
    else
        error("Input data is not hdf5")
    end

    # reconstruct space, obtain Nq from space
    cspace = axes(Y.c)
    fspace = axes(Y.f)
    hspace = cspace.horizontal_space
    Nq = Spaces.Quadratures.degrees_of_freedom(
        cspace.horizontal_space.quadrature_style,
    )

    # create a temporary dir for intermediate data
    remap_tmpdir = nc_dir * "remaptmp/"
    mkpath(remap_tmpdir)

    # write out our cubed sphere mesh
    meshfile_cc = remap_tmpdir * "mesh_cubedsphere.g"
    write_exodus(meshfile_cc, hspace.topology)

    meshfile_rll = remap_tmpdir * "mesh_rll.g"
    rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

    meshfile_overlap = remap_tmpdir * "mesh_overlap.g"
    overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

    weightfile = remap_tmpdir * "remap_weights.nc"
    remap_weights(
        weightfile,
        meshfile_cc,
        meshfile_rll,
        meshfile_overlap;
        in_type = "cgll",
        in_np = Nq,
    )

    return weightfile
end

function remap2latlon(filein, nc_dir, weightfile, nlat, nlon)
    if split(filein, ".")[end] == "hdf5"
        reader = InputOutput.HDF5Reader(filein)
        Y = InputOutput.read_field(reader, "Y")
        diag = InputOutput.read_field(reader, "diagnostics")
        t_now = InputOutput.HDF5.read_attribute(reader.file, "time")
    else
        error("Input data is not hdf5")
    end

    # float type
    FT = eltype(Y)

    # reconstruct space
    cspace = axes(Y.c)
    fspace = axes(Y.f)
    hspace = cspace.horizontal_space

    ### create an nc file to store raw cg data 
    # create data
    remap_tmpdir = nc_dir * "remaptmp/"
    datafile_cc = remap_tmpdir * "test.nc"
    nc = NCDataset(datafile_cc, "c")
    # defines the appropriate dimensions and variables for a space coordinate
    def_space_coord(nc, cspace, type = "cgll")
    def_space_coord(nc, fspace, type = "cgll")
    # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
    nc_time = def_time_coord(nc)
    # define variables for the prognostic states 
    nc_rho = defVar(nc, "rho", FT, cspace, ("time",))
    nc_thermo = defVar(nc, ENV["THERMO_VAR"], FT, cspace, ("time",))
    nc_u = defVar(nc, "u", FT, cspace, ("time",))
    nc_v = defVar(nc, "v", FT, cspace, ("time",))
    nc_w = defVar(nc, "w", FT, cspace, ("time",))
    # define variables for the diagnostic states
    nc_pres = defVar(nc, "pressure", FT, cspace, ("time",))
    nc_T = defVar(nc, "temperature", FT, cspace, ("time",))
    nc_θ = defVar(nc, "potential_temperature", FT, cspace, ("time",))
    nc_K = defVar(nc, "kinetic_energy", FT, cspace, ("time",))
    nc_vort = defVar(nc, "vorticity", FT, cspace, ("time",))
    #nc_T_sfc = defVar(nc, "sfc_temperature", FT, hspace, ("time",))
    #nc_qt_sfc = defVar(nc, "sfc_qt", FT, hspace, ("time",))
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
    if ENV["THERMO_VAR"] == "e_tot"
        nc_thermo[:, 1] = Y.c.ρe_tot ./ Y.c.ρ
    elseif ENV["THERMO_VAR"] == "theta"
        nc_thermo[:, 1] = Y.c.ρθ ./ Y.c.ρ
    else
        error("Invalid ENV[[\"THERMO_VAR\"]!")
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
    #nc_T_sfc[:, 1] = diag.sfc_temperature
    #nc_qt_sfc[:, 1] = diag.sfc_qt

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



    datafile_latlon = nc_dir * split(split(filein, "/")[end], ".")[1] * ".nc"
    dry_variables = [
        "rho",
        ENV["THERMO_VAR"],
        "u",
        "v",
        "w",
        "pressure",
        "temperature",
        "potential_temperature",
        "kinetic_energy",
        "vorticity",
        #"sfc_temperature",
        #"sfc_qt",
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

function remove_tmpdir(nc_dir)
    remap_tmpdir = nc_dir * "remaptmp/"
    rm(remap_tmpdir, recursive = true)
end

weightfile = create_weightfile(data_files[1], nc_dir, nlat, nlon)
for i in 1:length(data_files)
    data_file = data_files[i]
    remap2latlon(data_file, nc_dir, weightfile, nlat, nlon)
end
remove_tmpdir(nc_dir)
