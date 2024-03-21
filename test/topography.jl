using Test
using Random
import ClimaAtmos as CA
Random.seed!(1234)

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

@testset "test topography functions" begin
    (; FT, coords) = get_spherical_spaces()
    @test extrema(CA.topography_dcmip200(coords)) == (FT(0), FT(2000))
    loc = findmax(parent(CA.topography_dcmip200(coords)))
    @test parent(coords.lat)[loc[2]] == FT(0)
    @test parent(coords.long)[loc[2]] == FT(-90)
end
