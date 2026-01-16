using Test
import ClimaComms
ClimaComms.@import_required_backends
import Random
import ClimaAtmos as CA
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaParams as CP
import RRTMGP.Parameters.RRTMGPParameters

Random.seed!(1234)

include("test_helpers.jl")

# Setup RRTMGP parameters for extrapolation tests
const RRTMGP_PARAMS = RRTMGPParameters(CP.create_toml_dict(Float64))

@testset "Array <-> Field conversion" begin
    (; bubble_space) = get_spherical_spaces()
    test_field = ones(bubble_space)
    idx = size(parent(test_field))  # IJFH layout
    npts = idx[1] * idx[2] * idx[4]
    
    # Field -> Array
    test_array = Fields.field2array(test_field)
    @test test_array isa Array
    @test length(test_array) == npts
    
    # Array -> Field  
    new_field = Fields.array2field(test_array, bubble_space)
    @test new_field == test_field
    @test size(parent(new_field)) == idx
end

@testset "Interpolation methods" for FT in (Float32, Float64)
    # Test data
    T, Tꜛ, Tꜜ = FT(1), FT(1), FT(2)
    p, p₁, p₂ = FT(0.5), FT(2), FT(3)
    XT, XP = [FT(0)], [FT(0)]
    
    @testset "ArithmeticMean" begin
        RRTMGPI.interp!(RRTMGPI.ArithmeticMean(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ])
        @test XT[1] == (Tꜜ + Tꜛ) / 2
        @test XP[1] == (p₁ + p₂) / 2
    end

    @testset "GeometricMean" begin
        RRTMGPI.interp!(RRTMGPI.GeometricMean(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ])
        @test XT[1] == sqrt(Tꜜ * Tꜛ)
        @test XP[1] == sqrt(p₁ * p₂)
    end

    @testset "UniformZ" begin
        RRTMGPI.interp!(RRTMGPI.UniformZ(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ])
        @test XT[1] == (Tꜜ + Tꜛ) / 2
        @test XP[1] == RRTMGPI.uniform_z_p(XT[1], p₁, Tꜜ, p₂, Tꜛ)
    end

    @testset "UniformP" begin
        RRTMGPI.interp!(RRTMGPI.UniformP(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ])
        @test XP[1] == (p₁ + p₂) / 2
        # Same function form as uniform_z_p with swapped arguments
        @test XT[1] == RRTMGPI.uniform_z_p(XP[1], Tꜜ, p₁, Tꜛ, p₂)
    end
end

@testset "Extrapolation methods" for FT in (Float32, Float64)
    # Test data
    Tꜜ, Tꜛ = FT(2), FT(1)
    p₁, p₂ = FT(2), FT(3)
    Tₛ = FT(5)
    XT, XP = [FT(0)], [FT(0)]
    
    @testset "ArithmeticMean" begin
        RRTMGPI.extrap!(RRTMGPI.ArithmeticMean(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ], [Tₛ], RRTMGP_PARAMS)
        @test XP[1] == (3 * p₁ - p₂) / 2
        @test XT[1] == (3 * Tꜜ - Tꜛ) / 2
    end

    @testset "GeometricMean" begin
        RRTMGPI.extrap!(RRTMGPI.GeometricMean(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ], [Tₛ], RRTMGP_PARAMS)
        @test XP[1] == sqrt(p₁^3 / p₂)
        @test XT[1] == sqrt(Tꜜ^3 / Tꜛ)
    end

    @testset "UniformZ" begin
        RRTMGPI.extrap!(RRTMGPI.UniformZ(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ], [Tₛ], RRTMGP_PARAMS)
        # UniformZ: T is computed from uniform_z_p formula  
        @test XT[1] == RRTMGPI.uniform_z_p(XP[1], Tꜜ, p₁, Tꜛ, p₂)
    end

    @testset "UniformP" begin
        RRTMGPI.extrap!(RRTMGPI.UniformP(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ], [Tₛ], RRTMGP_PARAMS)
        # UniformP: P is computed from uniform_z_p formula (with swapped args)
        @test XP[1] == RRTMGPI.uniform_z_p(XT[1], p₁, Tꜜ, p₂, Tꜛ)
    end
end
