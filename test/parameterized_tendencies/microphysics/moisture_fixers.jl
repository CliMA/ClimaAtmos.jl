#=
Unit tests for moisture_fixers.jl

Tests cover:
1. `clip()` - non-negative clipping
2. `tracer_nonnegativity_vapor_tendency()` - tendency computation
=#

using Test
using ClimaAtmos

# Import functions under test
import ClimaAtmos:
    clip, tracer_nonnegativity_vapor_tendency, limit, triangle_inequality_limiter

@testset "Moisture Fixers" begin

    @testset "clip()" begin
        @testset "basic functionality" begin
            # Positive values pass through
            @test clip(0.01) == 0.01
            @test clip(1.0) == 1.0

            # Zero stays zero
            @test clip(0.0) == 0.0

            # Negative values clipped to zero
            @test clip(-0.01) == 0.0
            @test clip(-1.0) == 0.0
        end

        @testset "type stability" begin
            @test eltype(clip(Float32(-0.1))) == Float32
            @test eltype(clip(Float64(-0.1))) == Float64
            @test clip(Float32(0.5)) == Float32(0.5)
        end

        @testset "edge cases" begin
            # Very small negative
            @test clip(-1e-15) == 0.0

            # Very small positive preserved
            @test clip(1e-15) ≈ 1e-15
        end
    end

    @testset "tracer_nonnegativity_vapor_tendency()" begin
        dt = 1.0
        q_vap = 0.01  # 10 g/kg vapor available

        @testset "no correction needed" begin
            # Positive tracer: no correction
            @test tracer_nonnegativity_vapor_tendency(0.001, q_vap, dt) ≈ 0.0 atol =
                eps(Float64)

            # Zero tracer: no correction
            @test tracer_nonnegativity_vapor_tendency(0.0, q_vap, dt) ≈ 0.0 atol =
                eps(Float64)
        end

        @testset "correction for negative tracer" begin
            # Negative tracer: positive tendency returned
            q_neg = -0.001
            S = tracer_nonnegativity_vapor_tendency(q_neg, q_vap, dt)

            @test S > 0.0  # Positive tendency

            # Tendency should try to restore toward zero
            @test S <= abs(q_neg) / dt + eps(Float64)
        end

        @testset "limited by available vapor" begin
            # Very negative tracer, limited vapor
            q_very_neg = -0.1
            q_vap_small = 0.001

            S = tracer_nonnegativity_vapor_tendency(q_very_neg, q_vap_small, dt)

            # Should be limited by vapor (using n=5 sharing)
            @test S > 0.0
            @test S <= q_vap_small / dt / 5 + eps(Float64)
        end

        @testset "no vapor available" begin
            # Negative tracer but no vapor to borrow from
            S = tracer_nonnegativity_vapor_tendency(-0.001, 0.0, dt)
            @test S ≈ 0.0 atol = eps(Float64)
        end

        @testset "type stability" begin
            @test eltype(
                tracer_nonnegativity_vapor_tendency(
                    Float32(-0.001), Float32(0.01), Float32(1.0),
                ),
            ) == Float32
        end

        @testset "physical consistency" begin
            dt = 1.0
            q_vap = 0.01

            # Correction is bounded: cannot add more mass than available
            for q in [-0.001, -0.01, -0.1]
                S = tracer_nonnegativity_vapor_tendency(q, q_vap, dt)
                # Tendency * dt * 5 (number of species) should not exceed vapor
                @test S * dt * 5 <= q_vap + eps(Float64)
            end
        end
    end
end
