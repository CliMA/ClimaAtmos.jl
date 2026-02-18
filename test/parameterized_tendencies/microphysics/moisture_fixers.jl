#=
Unit tests for moisture_fixers.jl

Tests cover:
1. `tracer_nonnegativity_vapor_tendency()` - tendency computation
=#

using Test
using ClimaAtmos

# Import functions under test
import ClimaAtmos:
    tracer_nonnegativity_vapor_tendency, limit

@testset "Moisture Fixers" begin


    @testset "tracer_nonnegativity_vapor_tendency()" begin
        dt = 1.0
        q_vap = 0.01  # 10 g/kg vapor available

        @testset "no correction needed" begin
            # Positive tracer: no correction (smooth approx gives ~ε residual)
            @test tracer_nonnegativity_vapor_tendency(0.001, q_vap, dt) ≈ 0.0 atol =
                1e-7

            # Zero tracer: no correction
            @test tracer_nonnegativity_vapor_tendency(0.0, q_vap, dt) ≈ 0.0 atol =
                1e-7
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
            @test S <= q_vap_small / dt / 5 + 1e-7
        end

        @testset "no vapor available" begin
            # Negative tracer but no vapor to borrow from
            S = tracer_nonnegativity_vapor_tendency(-0.001, 0.0, dt)
            @test S ≈ 0.0 atol = 1e-7
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
                # Tendency * dt * 5 (number of species sharing) should not exceed vapor
                @test S * dt * 5 <= q_vap + 1e-7
            end
        end
    end
end
