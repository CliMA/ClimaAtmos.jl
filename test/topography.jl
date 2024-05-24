using Test
using ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using Random
import ClimaAtmos as CA
Random.seed!(1234)

### BoilerPlate Code
using ClimaComms
using IntervalSets

### Unit Tests for topography
# Ensures that space construction hooks in ClimaAtmos
# result in correct warped spaces with surface elevation.

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

include("test_helpers.jl")

@testset "test topography functions" begin
    (; FT, coords) = get_spherical_spaces()
    @test extrema(CA.topography_dcmip200(coords)) == (FT(0), FT(2000))
    loc = findmax(parent(CA.topography_dcmip200(coords)))
    @test parent(coords.lat)[loc[2]] == FT(0)
    @test parent(coords.long)[loc[2]] == FT(-90)
end
