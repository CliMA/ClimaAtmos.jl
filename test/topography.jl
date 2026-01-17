using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore.Geometry as Geometry

include("test_helpers.jl")

@testset "Topography" begin

    @testset "DCMIP200 topography" begin
        (; FT, coords) = get_spherical_spaces()
        topo = CA.topography_dcmip200.(coords)

        # Maximum height is 2000 m, minimum is 0
        @test extrema(topo) == (FT(0), FT(2000))

        # Peak is at equator (lat=0), longitude=-90 (= 3/2 * 180 - 360)
        loc = findmax(parent(topo))
        @test parent(coords.lat)[loc[2]] == FT(0)
        @test parent(coords.long)[loc[2]] == FT(-90)
    end

    @testset "Hughes2023 topography" begin
        (; FT, coords) = get_spherical_spaces()
        topo = CA.topography_hughes2023.(coords)

        # Maximum height is close to 2000 m (h₀), minimum is near 0
        @test maximum(topo) <= FT(2000)
        @test minimum(topo) >= FT(0)

        # No NaNs
        @test !any(isnan, topo)
    end

    @testset "CosineTopography (3D)" begin
        FT = Float64
        topo = CA.CosineTopography{3, FT}(; h_max = 100.0, λ = 1000.0)

        # Test peak at origin
        coord = (; x = FT(0), y = FT(0))
        @test CA.topography_function(topo, coord) == FT(100)

        # Test trough at half wavelength
        coord_half = (; x = FT(500), y = FT(0))
        @test CA.topography_function(topo, coord_half) ≈ FT(-100) atol = eps(FT)
    end

    @testset "CosineTopography (2D)" begin
        FT = Float64
        topo = CA.CosineTopography{2, FT}(; h_max = 50.0, λ = 2000.0)

        # Test peak at origin
        coord = (; x = FT(0))
        @test CA.topography_function(topo, coord) == FT(50)

        # Test trough at half wavelength
        coord_half = (; x = FT(1000))
        @test CA.topography_function(topo, coord_half) ≈ FT(-50) atol = eps(FT)
    end

    @testset "AgnesiTopography" begin
        FT = Float64
        topo = CA.AgnesiTopography{FT}(; h_max = 1000.0, x_center = 500.0, a = 100.0)

        # Peak at x_center
        @test CA.topography_function(topo, (; x = FT(500))) == FT(1000)

        # Half-height at x = x_center ± a
        @test CA.topography_function(topo, (; x = FT(600))) == FT(500)

        # Decays smoothly away from center
        @test CA.topography_function(topo, (; x = FT(0))) < FT(50)
    end

    @testset "ScharTopography" begin
        FT = Float64
        topo =
            CA.ScharTopography{FT}(; h_max = 500.0, x_center = 1000.0, λ = 200.0, a = 100.0)

        # Peak at x_center
        @test CA.topography_function(topo, (; x = FT(1000))) == FT(500)

        # Zero at half-wavelength from center (due to cos² term)
        @test CA.topography_function(topo, (; x = FT(1100))) ≈ FT(0) atol = 1e-6

        # Non-negative everywhere
        for x in FT.(0:100:2000)
            @test CA.topography_function(topo, (; x)) >= FT(0)
        end
    end

    @testset "MeshWarpType constructors" begin
        # LinearWarp is a singleton
        @test CA.LinearWarp() isa CA.MeshWarpType

        # SLEVEWarp has parameters
        sleve = CA.SLEVEWarp{Float64}(; eta = 0.8, s = 12.0)
        @test sleve.eta == 0.8
        @test sleve.s == 12.0
    end
end
