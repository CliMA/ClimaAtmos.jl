### BoilerPlate Code
using IntervalSets

import ClimaCore:
    ClimaCore,
    Domains,
    Geometry,
    Grids,
    Fields,
    Operators,
    Meshes,
    Spaces,
    Quadratures,
    Topologies,
    Hypsography

### Unit Test Helpers
# If wrappers for general operations are required in unit tests
# (e.g. construct spaces, construct simulations from configs,
# specialised test functions with multiple uses, define them here.)

function generate_test_simulation(config)
    parsed_args = config.parsed_args
    simulation = CA.AtmosSimulation(config)
    (; integrator) = simulation
    Y = integrator.u
    p = integrator.p
    return (; Y = Y, p = p, params = p.params, simulation = simulation)
end

function get_spherical_spaces(; FT = Float32)
    context = ClimaComms.SingletonCommsContext()
    radius = FT(10π)
    ne = 4
    Nq = 4
    domain = Domains.SphereDomain(radius)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(topology, quad)
    enable_bubble = false
    no_bubble_space =
        Spaces.SpectralElementSpace2D(topology, quad; enable_bubble)

    # Now check constructor with bubble enabled
    enable_bubble = true
    bubble_space = Spaces.SpectralElementSpace2D(topology, quad; enable_bubble)

    lat = Fields.coordinate_field(bubble_space).lat
    long = Fields.coordinate_field(bubble_space).long
    coords = Fields.coordinate_field(bubble_space)
    return (;
        bubble_space = bubble_space,
        no_bubble_space = no_bubble_space,
        lat = lat,
        long = long,
        coords = coords,
        FT = FT,
    )
end

function get_cartesian_spaces(; FT = Float32)
    xlim = (FT(0), FT(π))
    zlim = (FT(0), FT(π))
    helem = 5
    velem = 10
    npoly = 5
    ndims = 3
    stretch = Meshes.Uniform()
    device = ClimaComms.CPUSingleThreaded()
    comms_context = ClimaComms.SingletonCommsContext(device)
    # Horizontal Grid Construction
    quad = Quadratures.GLL{npoly + 1}()
    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
        Geometry.YPoint{FT}(xlim[1]) .. Geometry.YPoint{FT}(xlim[2]),
        x1periodic = true,
        x2periodic = true,
    )
    # Assume same number of elems (helem) in (x,y) directions
    horzmesh = Meshes.RectilinearMesh(horzdomain, helem, helem)
    horz_topology = Topologies.Topology2D(
        comms_context,
        horzmesh,
        Topologies.spacefillingcurve(horzmesh),
    )
    h_space =
        Spaces.SpectralElementSpace2D(horz_topology, quad, enable_bubble = true)

    horz_grid = Spaces.grid(h_space)

    # Vertical Grid Construction
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = velem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)
    vert_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        vertmesh,
    )
    vert_grid = Grids.FiniteDifferenceGrid(vert_topology)
    ArrayType = ClimaComms.array_type(device)
    grid = Grids.ExtrudedFiniteDifferenceGrid(horz_grid, vert_grid)
    cent_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)
    return (
        helem = helem,
        cent_space = cent_space,
        face_space = face_space,
        xlim = xlim,
        zlim = zlim,
        velem = velem,
        npoly = npoly,
        quad = quad,
    )
end

function get_coords(cent_space, face_space)
    ccoords = Fields.coordinate_field(cent_space)
    fcoords = Fields.coordinate_field(face_space)
    return ccoords, fcoords
end

function taylor_green_ic(coords)
    u = @. sin(coords.x) * cos(coords.y) * cos(coords.z)
    v = @. -cos(coords.x) * sin(coords.y) * cos(coords.z)
    #TODO: If a w field is introduced include it here. 
    return u, v, u .* 0
end

"""
    get_test_functions(cent_space, face_space)
Given center and face space objects for the staggered grid
construction, generate velocity profiles defined by  simple
trigonometric functions (for checks against analytical solutions). 
This will be generalised to accept arbitrary input profiles 
(e.g. spherical harmonics) for testing purposes in the future. 
"""
function get_test_functions(cent_space, face_space)
    ccoords, fcoords = get_coords(cent_space, face_space)
    FT = eltype(ccoords)
    Q = zero.(ccoords.x)
    # Exact velocity profiles
    u, v, w = taylor_green_ic(ccoords)
    ᶠu, ᶠv, ᶠw = taylor_green_ic(fcoords)
    (; x, y, z) = ccoords
    UVW = Geometry.UVWVector
    # Assemble (Cartesian) velocity
    ᶜu = @. UVW(Geometry.UVector(u)) +
       UVW(Geometry.VVector(v)) +
       UVW(Geometry.WVector(w))
    ᶠu = @. UVW(Geometry.UVector(ᶠu)) +
       UVW(Geometry.VVector(ᶠv)) +
       UVW(Geometry.WVector(ᶠw))
    # Get covariant components
    uₕ = @. Geometry.Covariant12Vector(ᶜu)
    uᵥ = @. Geometry.Covariant3Vector(ᶠu)
    return uₕ, uᵥ
end


"""
    create_mock_era5_datasets(temporary_dir, start_date, FT; hours = 24, num_lat = 17, num_lon = 17, num_pressure = 37, base_date = "20000101")

Create mock 24 hour ERA5 datasets for testing forcing file generation. Returns paths to the created files. num points set for smoothing 
up to 4 points in each direction (e.g. smoothing over a 2°x2° box).

All datasets use the same time reference (base_date) to avoid concatenation issues.
"""
function create_mock_era5_datasets(
    temporary_dir,
    start_date,
    FT;
    hours = 24,
    num_lat = 17,
    num_lon = 17,
    num_pressure = 37,
    base_date = "20000101",
)
    column_data_path = joinpath(
        temporary_dir,
        "forcing_and_cloud_hourly_profiles_$(start_date).nc",
    )
    accum_data_path = joinpath(temporary_dir, "hourly_accum_$(start_date).nc")
    inst_data_path = joinpath(temporary_dir, "hourly_inst_$(start_date).nc")

    # Calculate hours since base_date for this start_date
    base_datetime = Dates.DateTime(base_date, "yyyymmdd")
    start_datetime = Dates.DateTime(start_date, "yyyymmdd")
    hours_offset = Dates.value(Dates.Hour(start_datetime - base_datetime))

    # Create column dataset (4D: lon, lat, pressure, time)
    tvforcing = NCDataset(column_data_path, "c")
    defDim(tvforcing, "valid_time", hours)
    defDim(tvforcing, "pressure_level", num_pressure)
    defDim(tvforcing, "latitude", num_lat)
    defDim(tvforcing, "longitude", num_lon)

    # define coordinate variables
    defVar(tvforcing, "latitude", FT, ("latitude",))
    defVar(tvforcing, "longitude", FT, ("longitude",))
    defVar(tvforcing, "pressure_level", FT, ("pressure_level",))
    defVar(tvforcing, "valid_time", FT, ("valid_time",))
    # Use consistent time reference for all datasets
    tvforcing["valid_time"].attrib["units"] = "hours since $(base_date[1:4])-$(base_date[5:6])-$(base_date[7:8]) 00:00:00"
    tvforcing["valid_time"].attrib["calendar"] = "standard"

    # fill coordinate variables
    lat_step = 4.0 / (num_lat - 1)
    lon_step = 4.0 / (num_lon - 1)
    tvforcing["latitude"][:] = collect(-2.0:lat_step:2.0)
    tvforcing["longitude"][:] = collect(-2.0:lon_step:2.0)
    tvforcing["pressure_level"][:] =
        10 .^ (range(1, stop = 4, length = num_pressure))
    # Time values are hours since base_date, not since this file's date
    tvforcing["valid_time"][:] =
        collect(hours_offset:(hours_offset + hours - 1))

    # define the forcing variables
    full_dims = ("longitude", "latitude", "pressure_level", "valid_time")
    defVar(tvforcing, "u", FT, full_dims)
    defVar(tvforcing, "v", FT, full_dims)
    defVar(tvforcing, "w", FT, full_dims)
    defVar(tvforcing, "t", FT, full_dims)
    defVar(tvforcing, "q", FT, full_dims)
    defVar(tvforcing, "z", FT, full_dims)
    defVar(tvforcing, "clwc", FT, full_dims)
    defVar(tvforcing, "ciwc", FT, full_dims)

    # fill the variables with uniform ones
    tvforcing["u"][:, :, :, :] .= ones(FT, size(tvforcing["u"]))
    tvforcing["v"][:, :, :, :] .= ones(FT, size(tvforcing["v"]))
    tvforcing["w"][:, :, :, :] .= ones(FT, size(tvforcing["w"]))
    tvforcing["t"][:, :, :, :] .= ones(FT, size(tvforcing["t"]))
    tvforcing["q"][:, :, :, :] .= ones(FT, size(tvforcing["q"]))
    tvforcing["z"][:, :, :, :] .= ones(FT, size(tvforcing["z"]))
    tvforcing["clwc"][:, :, :, :] .= ones(FT, size(tvforcing["clwc"]))
    tvforcing["ciwc"][:, :, :, :] .= ones(FT, size(tvforcing["ciwc"]))
    close(tvforcing)

    # Create accumulated dataset (3D: lon, lat, time)
    tv_accum = NCDataset(accum_data_path, "c")
    defDim(tv_accum, "valid_time", hours)
    defDim(tv_accum, "latitude", num_lat)
    defDim(tv_accum, "longitude", num_lon)
    defVar(tv_accum, "latitude", FT, ("latitude",))
    defVar(tv_accum, "longitude", FT, ("longitude",))
    defVar(tv_accum, "valid_time", FT, ("valid_time",))
    # Use consistent time reference for all datasets
    tv_accum["valid_time"].attrib["units"] = "hours since $(base_date[1:4])-$(base_date[5:6])-$(base_date[7:8]) 00:00:00"
    tv_accum["valid_time"].attrib["calendar"] = "standard"

    tv_accum["latitude"][:] = collect(-2.0:lat_step:2.0)
    tv_accum["longitude"][:] = collect(-2.0:lon_step:2.0)
    # Time values are hours since base_date, not since this file's date
    tv_accum["valid_time"][:] = collect(hours_offset:(hours_offset + hours - 1))

    defVar(tv_accum, "slhf", FT, ("longitude", "latitude", "valid_time"))
    defVar(tv_accum, "sshf", FT, ("longitude", "latitude", "valid_time"))
    tv_accum["slhf"][:, :, :] .= ones(FT, size(tv_accum["slhf"]))
    tv_accum["sshf"][:, :, :] .= ones(FT, size(tv_accum["sshf"]))
    close(tv_accum)

    # Create instantaneous dataset (3D: lon, lat, time)
    tv_inst = NCDataset(inst_data_path, "c")
    defDim(tv_inst, "valid_time", hours)
    defDim(tv_inst, "latitude", num_lat)
    defDim(tv_inst, "longitude", num_lon)
    defVar(tv_inst, "latitude", FT, ("latitude",))
    defVar(tv_inst, "longitude", FT, ("longitude",))
    defVar(tv_inst, "valid_time", FT, ("valid_time",))
    # Use consistent time reference for all datasets
    tv_inst["valid_time"].attrib["units"] = "hours since $(base_date[1:4])-$(base_date[5:6])-$(base_date[7:8]) 00:00:00"
    tv_inst["valid_time"].attrib["calendar"] = "standard"

    tv_inst["latitude"][:] = collect(-2.0:lat_step:2.0)
    tv_inst["longitude"][:] = collect(-2.0:lon_step:2.0)
    # Time values are hours since base_date, not since this file's date
    tv_inst["valid_time"][:] = collect(hours_offset:(hours_offset + hours - 1))

    defVar(tv_inst, "skt", FT, ("longitude", "latitude", "valid_time"))
    tv_inst["skt"][:, :, :] .= ones(FT, size(tv_inst["skt"]))
    close(tv_inst)

    return column_data_path, accum_data_path, inst_data_path
end
