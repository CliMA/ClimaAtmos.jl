#=
Unit tests for tendency_limiters.jl

Tests cover:
1. `limit()` - basic rate limiting
2. `tendency_limiter()` - bidirectional tendency limiting
3. `coupled_sink_limit_factor()` - uniform scaling for coupled sinks
4. `limit_sink()` - sink-only tendency limiting
5. `explicit_1m_tendency_limit()` - end-to-end 1M explicit limiter
=#

using Test
using ClimaAtmos

# Import functions under test
import ClimaAtmos:
    limit,
    tendency_limiter,
    coupled_sink_limit_factor,
    limit_sink

@testset "Tendency Limiters" begin

    @testset "limit()" begin
        dt = 1.0
        q = 0.01  # 10 g/kg

        @testset "basic functionality" begin
            # Single sink: can deplete all of q in one timestep
            @test limit(q, dt, 1) ≈ q / dt

            # Multiple sinks: each gets fraction of available
            @test limit(q, dt, 2) ≈ q / (2 * dt)
            @test limit(q, dt, 3) ≈ q / (3 * dt)
        end

        @testset "edge cases" begin
            # Zero quantity
            @test limit(0.0, dt, 2) == 0.0

            # Negative quantity (clamped to zero)
            @test limit(-0.001, dt, 2) == 0.0
        end

        @testset "type stability" begin
            @test eltype(limit(Float32(q), Float32(dt), 2)) == Float32
            @test eltype(limit(Float64(q), Float64(dt), 2)) == Float64
        end
    end



    @testset "tendency_limiter()" begin
        @testset "positive tendency limited" begin
            # Source bound limits how much can grow
            L = tendency_limiter(0.1, 0.02, 1.0)
            @test L > 0.0
            @test L < 0.1  # Should be limited
        end

        @testset "positive tendency pass-through" begin
            # Large bound, small tendency → passes through
            L = tendency_limiter(0.001, 1.0, 1.0)
            @test L ≈ 0.001 atol = 1e-6
        end

        @testset "negative tendency limited" begin
            # Negative bound limits how much can be removed
            L = tendency_limiter(-0.1, 1.0, 0.01)
            @test L < 0.0
            @test abs(L) < 0.1  # Should be limited
        end

        @testset "negative tendency pass-through" begin
            # Large bound, small tendency → passes through
            L = tendency_limiter(-0.001, 1.0, 1.0)
            @test L ≈ -0.001 atol = 1e-6
        end

        @testset "zero tendency" begin
            @test tendency_limiter(0.0, 0.05, 0.05) ≈ 0.0 atol = 1e-9
        end

        @testset "zero positive bound prevents positive tendency" begin
            L = tendency_limiter(0.1, 0.0, 0.05)
            @test L ≈ 0.0 atol = 1e-7
        end

        @testset "zero negative bound prevents negative tendency" begin
            L = tendency_limiter(-0.1, 0.05, 0.0)
            @test L ≈ 0.0 atol = 1e-7
        end

        @testset "type stability" begin
            @test eltype(
                tendency_limiter(
                    Float32(0.5),
                    Float32(0.3),
                    Float32(0.3),
                ),
            ) == Float32
            @test eltype(
                tendency_limiter(
                    Float64(0.5),
                    Float64(0.3),
                    Float64(0.3),
                ),
            ) == Float64
        end
    end

    @testset "coupled_sink_limit_factor()" begin
        dt = 1.0

        @testset "both sources → no limiting" begin
            # Positive tendencies (sources) should return factor = 1
            f = coupled_sink_limit_factor(0.01, 1e7, 0.001, 1e8, dt)
            @test f == 1.0
        end

        @testset "uses more restrictive limit" begin
            # Mass limited more than number → mass factor wins
            q = 0.001
            n = 1e10  # Lots of number available
            Sq = -0.01  # Would deplete q quickly
            Sn = -1e7   # Would take long to deplete n

            f = coupled_sink_limit_factor(Sq, Sn, q, n, dt)
            # Mass bound = q/(dt*3) = 0.000333, |Sq| = 0.01
            # f_mass = 0.000333/0.01 ≈ 0.033
            # Number bound = n/(dt*3) ≈ 3.3e9, |Sn| = 1e7 → no limiting
            @test f < 0.1  # Should be limited by mass
        end

        @testset "preserves ratio when both limited" begin
            q = 0.001
            n = 1e6
            Sq = -0.01
            Sn = -1e8  # Both would deplete their source

            f = coupled_sink_limit_factor(Sq, Sn, q, n, dt)
            # Both should be limited, same factor applied
            @test f < 1.0
            @test f > 0.0

            # Verify the ratio is preserved
            Sq_limited = Sq * f
            Sn_limited = Sn * f
            @test Sq_limited / Sn_limited ≈ Sq / Sn
        end

        @testset "handles mixed signs" begin
            # One source, one sink
            Sq = 0.01   # Source
            Sn = -1e8   # Sink

            f = coupled_sink_limit_factor(Sq, Sn, 0.001, 1e6, dt)
            # Should still limit based on number sink
            @test f <= 1.0
        end

        @testset "custom n parameter" begin
            q = 0.001
            Sq = -0.01
            Sn = -1e5

            # More sinks → stricter bound
            f_3 = coupled_sink_limit_factor(Sq, Sn, q, 1e10, dt, 3)
            f_10 = coupled_sink_limit_factor(Sq, Sn, q, 1e10, dt, 10)
            @test f_10 < f_3  # More sinks = stricter = smaller factor
        end

        @testset "type stability" begin
            f = coupled_sink_limit_factor(
                Float32(-0.01), Float32(-1e8),
                Float32(0.001), Float32(1e8),
                Float32(1.0),
            )
            @test eltype(f) == Float32
        end
    end

    @testset "limit_sink()" begin
        dt = 1.0

        @testset "sources pass through unchanged" begin
            # Positive tendency (source) should not be limited
            @test limit_sink(0.1, 0.001, dt) == 0.1
            @test limit_sink(0.001, 0.0, dt) == 0.001  # Even with zero q
        end

        @testset "small sinks pass through" begin
            # Sink well within budget → passes through
            q = 0.01
            S = -0.0001  # |S| = 0.0001, budget = 0.01/(1*3) = 0.0033
            @test limit_sink(S, q, dt) ≈ S
        end

        @testset "large sinks are limited" begin
            q = 0.001
            S = -0.01  # |S| = 0.01, budget = 0.001/(1*3) ≈ 0.000333
            L = limit_sink(S, q, dt)
            @test L < 0.0              # Still a sink
            @test abs(L) < abs(S)      # Magnitude reduced
            @test abs(L) ≈ q / (dt * 3)  # Clamped to budget
        end

        @testset "zero quantity prevents depletion" begin
            L = limit_sink(-0.1, 0.0, dt)
            @test L ≈ 0.0 atol = 1e-15
        end

        @testset "custom n parameter" begin
            q = 0.001
            S = -0.01
            L3 = limit_sink(S, q, dt, 3)
            L5 = limit_sink(S, q, dt, 5)
            @test abs(L5) < abs(L3)  # More sinks = tighter limit
        end

        @testset "type stability" begin
            @test eltype(limit_sink(Float32(-0.01), Float32(0.001), Float32(1.0))) ==
                  Float32
            @test eltype(limit_sink(Float64(-0.01), Float64(0.001), Float64(1.0))) ==
                  Float64
        end
    end
end
