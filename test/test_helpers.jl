### BoilerPlate Code
using ClimaComms
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
    simulation = CA.get_simulation(config)
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
