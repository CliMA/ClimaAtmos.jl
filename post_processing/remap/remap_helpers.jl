import ClimaCoreTempestRemap
import ClimaCore: Spaces, Fields, Quadratures
import ClimaComms
import ClimaAtmos: SurfaceConditions, CT3
import ClimaCore.Utilities: half
"""
    create_weightfile(
        weightfile::String,
        cspace::Spaces.AbstractSpace,
        fspace::Spaces.AbstractSpace,
        nlat::Int,
        nlon::Int
    )

A helper for creating weights for ClimaCoreTempestRemap. Example:

```julia
create_weightfile(weightfile, cspace, fspace, nlat, nlon)
```
TODO: should this live in ClimaCoreTempestRemap?
"""
function create_weightfile(
    weightfile::String,
    cspace::Spaces.AbstractSpace,
    fspace::Spaces.AbstractSpace,
    nlat::Int,
    nlon::Int;
    mono = false,
)
    # space info to generate nc raw data
    hspace = cspace.horizontal_space
    Nq =
        Quadratures.degrees_of_freedom(cspace.horizontal_space.quadrature_style)
    # create a temporary dir for intermediate data
    mktempdir() do tmp
        mkpath(tmp)
        # write out our cubed sphere mesh
        meshfile_cc = joinpath(tmp, "mesh_cubedsphere.g")
        ClimaCoreTempestRemap.write_exodus(meshfile_cc, hspace.topology)
        meshfile_rll = joinpath(tmp, "mesh_rll.g")
        ClimaCoreTempestRemap.rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)
        meshfile_overlap = joinpath(tmp, "mesh_overlap.g")
        ClimaCoreTempestRemap.overlap_mesh(
            meshfile_overlap,
            meshfile_cc,
            meshfile_rll,
        )
        tmp_weightfile = joinpath(tmp, "remap_weights.nc")
        kwargs = (; in_type = "cgll", in_np = Nq)
        kwargs = mono ? (; (kwargs)..., mono = mono) : kwargs
        ClimaCoreTempestRemap.remap_weights(
            tmp_weightfile,
            meshfile_cc,
            meshfile_rll,
            meshfile_overlap;
            kwargs...,
        )
        mv(tmp_weightfile, weightfile; force = true)
    end
end

function create_weightfile(filein, remap_tmpdir, nlat, nlon)
    @assert endswith(filein, "hdf5")
    reader = InputOutput.HDF5Reader(
        filein,
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
    )
    Y = InputOutput.read_field(reader, "Y")
    weightfile = joinpath(remap_tmpdir, "remap_weights.nc")
    create_weightfile(weightfile, axes(Y.c), axes(Y.f), nlat, nlon)
    return weightfile
end

function remap2latlon(filein, data_dir, remap_tmpdir, weightfile, nlat, nlon)
    @assert endswith(filein, "hdf5")
    reader = InputOutput.HDF5Reader(
        filein,
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
    )
    Y = InputOutput.read_field(reader, "Y")
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
    nc_thermo = defVar(nc, "e_tot", FT, cspace, ("time",))
    nc_u = defVar(nc, "u", FT, cspace, ("time",))
    nc_v = defVar(nc, "v", FT, cspace, ("time",))
    nc_w = defVar(nc, "w", FT, cspace, ("time",))

    # time
    nc_time[1] = t_now

    # reconstruct fields
    # density
    nc_rho[:, 1] = Y.c.ρ
    # thermodynamics
    nc_thermo[:, 1] = Y.c.ρe_tot ./ Y.c.ρ

    # physical horizontal velocity
    uh_phy = Geometry.transform.(tuple(Geometry.UVAxis()), Y.c.uₕ)
    nc_u[:, 1] = uh_phy.components.data.:1
    nc_v[:, 1] = uh_phy.components.data.:2
    # physical vertical velocity
    ᶠw = Geometry.WVector.(Y.f.u₃)
    ᶜw = ᶜinterp.(ᶠw)
    nc_w[:, 1] = ᶜw

    close(nc)

    datafile_latlon =
        joinpath(out_dir, first(splitext(basename(filein))) * ".nc")
    dry_variables = ["rho", "e_tot", "u", "v", "w"]

    if :ρq_tot in propertynames(Y.c)
        moist_variables = ["qt"]
    else
        moist_variables = String[]
    end

    netcdf_variables = vcat(dry_variables, moist_variables)
    apply_remap(datafile_latlon, datafile_cc, weightfile, netcdf_variables)
    rm(datafile_cc)
end
