import ClimaCore
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Operators
using NCDatasets
using ClimaCoreTempestRemap

using JLD2

if haskey(ENV, "JLD2_DIR")
    jld2_dir = ENV["JLD2_DIR"]
else
    error("ENV[\"JLD2_DIR\"] require!")
end

if haskey(ENV, "THERMO_VAR")
    hs_thermo = ENV["THERMO_VAR"]
else
    error("ENV[\"THERMO_VAR\"] require (\"e_tot\" or \"theta\")")
end

if haskey(ENV, "NC_DIR")
    nc_dir = ENV["NC_DIR"]
else
    nc_dir = jld2_dir * "/nc/"
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

jld2_files = filter(x -> endswith(x, ".jld2"), readdir(jld2_dir, join = true))

function remap2latlon(filein, nc_dir, nlat, nlon)
    datain = jldopen(filein)

    # get time and states from jld2 data
    t_now = datain["t"]
    Y = datain["Y"]

    # float type
    FT = eltype(Y)

    # reconstruct space, obtain Nq from space
    cspace = axes(Y.c)
    hspace = cspace.horizontal_space
    Nq = Spaces.Quadratures.degrees_of_freedom(
        cspace.horizontal_space.quadrature_style,
    )

    # create a temporary dir for intermediate data
    remap_tmpdir = nc_dir * "remaptmp/"
    mkpath(remap_tmpdir)

    ### create an nc file to store raw cg data 
    # create data
    datafile_cc = remap_tmpdir * "test.nc"
    nc = NCDataset(datafile_cc, "c")
    # defines the appropriate dimensions and variables for a space coordinate
    def_space_coord(nc, cspace, type = "cgll")
    # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
    nc_time = def_time_coord(nc)
    # define variables for the prognostic states 
    nc_rho = defVar(nc, "rho", FT, cspace, ("time",))
    nc_thermo = defVar(nc, ENV["THERMO_VAR"], FT, cspace, ("time",))
    nc_u = defVar(nc, "u", FT, cspace, ("time",))
    nc_v = defVar(nc, "v", FT, cspace, ("time",))
    # TODO: interpolate w onto center space and save it the same way as the other vars

    # time
    nc_time[1] = t_now

    # reconstruct fields
    # density
    nc_rho[:, 1] = Y.c.ρ
    # thermodynamics
    if ENV["THERMO_VAR"] == "e_tot"
        nc_thermo[:, 1] = Y.c.ρe ./ Y.c.ρ
    elseif ENV["THERMO_VAR"] == "theta"
        nc_thermo[:, 1] = Y.c.ρθ ./ Y.c.ρ
    else
        error("Invalid ENV[[\"THERMO_VAR\"]!")
    end
    # physical horizontal velocity
    uh_phy = Geometry.transform.(Ref(Geometry.UVAxis()), Y.c.uₕ)
    nc_u[:, 1] = uh_phy.components.data.:1
    nc_v[:, 1] = uh_phy.components.data.:2

    close(nc)

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

    datafile_latlon = nc_dir * split(split(filein, "/")[end], ".")[1] * ".nc"
    apply_remap(
        datafile_latlon,
        datafile_cc,
        weightfile,
        ["rho", ENV["THERMO_VAR"], "u", "v"],
    )

    rm(remap_tmpdir, recursive = true)

end

for jld2_file in jld2_files
    remap2latlon(jld2_file, nc_dir, nlat, nlon)
end
