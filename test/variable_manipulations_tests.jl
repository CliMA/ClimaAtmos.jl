#=
Variable manipulations unit tests for ClimaAtmos.jl

Tests for functions in src/utils/variable_manipulations.jl that handle
density-weighted variables, SGS/EDMFX calculations, and tracer operations.
=#

using Test
import ClimaAtmos as CA

@testset "sgs_weight_function" begin
    # Test smooth weight function properties
    # w(0) should be 0, w(1) should be 1
    @test CA.sgs_weight_function(0.0, 0.5) ≈ 0.0 atol = 1e-10
    @test CA.sgs_weight_function(1.0, 0.5) ≈ 1.0 atol = 1e-10
    
    # w(a_half) should be 0.5
    @test CA.sgs_weight_function(0.5, 0.5) ≈ 0.5 atol = 1e-6
    @test CA.sgs_weight_function(0.3, 0.3) ≈ 0.5 atol = 1e-6
    
    # Should be monotonically increasing
    a_values = 0.0:0.1:1.0
    w_values = [CA.sgs_weight_function(a, 0.5) for a in a_values]
    @test all(diff(w_values) .>= 0)
    
    # Test with different a_half values
    @test 0 < CA.sgs_weight_function(0.2, 0.1) < 1
    @test 0 < CA.sgs_weight_function(0.8, 0.9) < 1
end

@testset "specific (regularized division)" begin
    # Basic case: simple division when ρ > 0
    @test CA.specific(10.0, 2.0) ≈ 5.0
    @test CA.specific(0.0, 1.0) ≈ 0.0
    
    # Edge case: very small density should still work
    @test isfinite(CA.specific(1e-20, 1e-20))
end
