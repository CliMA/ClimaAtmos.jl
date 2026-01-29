#=
Unit tests for tendency_limiters.jl

Tests cover:
1. `limit()` - basic rate limiting
2. `triangle_inequality_limiter()` - Horn (2012) limiter properties
3. `smooth_tendency_limiter()` - bidirectional tendency limiting
=#

using Test
using ClimaAtmos

# Import functions under test
import ClimaAtmos:
    limit,
    triangle_inequality_limiter,
    smooth_min_limiter,
    smooth_tendency_limiter,
    coupled_sink_limit_factor

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

    @testset "triangle_inequality_limiter()" begin
        dt = 1.0

        @testset "bounded output" begin
            # L ≤ min(F, M) for F, M ≥ 0
            for F in [0.0, 0.001, 0.01, 0.1, 1.0]
                for M in [0.0, 0.001, 0.01, 0.1, 1.0]
                    L = triangle_inequality_limiter(F, M)
                    @test L >= 0.0
                    @test L <= min(F, M) + eps(Float64)
                end
            end
        end

        @testset "limiting behavior" begin
            # F >> M: output ≈ M (capped at available)
            @test triangle_inequality_limiter(1.0, 0.01) ≈ 0.01 atol = 0.001

            # F << M: output ≈ F (pass-through)
            @test triangle_inequality_limiter(0.01, 1.0) ≈ 0.01 atol = 0.001
        end

        @testset "symmetric case" begin
            # F = M: output = F + M - sqrt(2)*M ≈ 0.586*M
            F = M = 0.5
            L = triangle_inequality_limiter(F, M)
            expected = F + M - sqrt(2) * M
            @test L ≈ expected
        end

        @testset "zero inputs" begin
            @test triangle_inequality_limiter(0.0, 0.0) == 0.0
            @test triangle_inequality_limiter(0.0, 1.0) == 0.0
            @test triangle_inequality_limiter(1.0, 0.0) == 0.0
        end

        @testset "negative force" begin
            # Negative force should swap arguments and negate result
            L_pos = triangle_inequality_limiter(0.5, 0.3, 0.2)
            L_neg = triangle_inequality_limiter(-0.5, 0.2, 0.3)
            @test L_neg ≈ -L_pos
        end

        @testset "negative allowed_source_amount edge case" begin
            # When source goes negative, use limit_neg as fallback
            L = triangle_inequality_limiter(0.1, -0.01, 0.2)
            # Result is bounded by the reverse limiting
            @test isfinite(L)

            # Both negative: return zero
            L = triangle_inequality_limiter(0.1, -0.01, 0.0)
            @test L == 0.0
        end

        @testset "type stability" begin
            @test eltype(triangle_inequality_limiter(Float32(0.5), Float32(0.3))) == Float32
            @test eltype(triangle_inequality_limiter(Float64(0.5), Float64(0.3))) == Float64
        end
    end

    @testset "smooth_min_limiter()" begin
        @testset "bounded output" begin
            # L ≤ min(S, B) for S, B ≥ 0
            for S in [0.0, 0.001, 0.01, 0.1, 1.0]
                for B in [0.0, 0.001, 0.01, 0.1, 1.0]
                    L = smooth_min_limiter(S, B)
                    @test L >= 0.0 || isapprox(L, 0.0, atol = 1e-9)
                    @test L <= min(S, B) + 1e-9
                end
            end
        end

        @testset "limiting behavior" begin
            # S >> B: output ≈ B (capped at available)
            @test smooth_min_limiter(1.0, 0.01) ≈ 0.01 atol = 1e-8

            # S << B: output ≈ S (pass-through)
            @test smooth_min_limiter(0.01, 1.0) ≈ 0.01 atol = 1e-8
        end

        @testset "crossover case S = B" begin
            # S = B: output ≈ B 
            S = B = 0.5
            L = smooth_min_limiter(S, B)
            # Should be B - ε/2 ≈ B for small ε
            @test L ≈ B atol = 1e-9
        end

        @testset "symmetry" begin
            # smooth_min_limiter(S, B) = smooth_min_limiter(B, S)
            @test smooth_min_limiter(0.3, 0.7) ≈ smooth_min_limiter(0.7, 0.3)
            @test smooth_min_limiter(0.1, 0.9) ≈ smooth_min_limiter(0.9, 0.1)
        end

        @testset "zero inputs" begin
            @test smooth_min_limiter(0.0, 0.0) ≈ 0.0 atol = 1e-9
            @test smooth_min_limiter(0.0, 1.0) ≈ 0.0 atol = 1e-9
            @test smooth_min_limiter(1.0, 0.0) ≈ 0.0 atol = 1e-9
        end

        @testset "sharpness parameter" begin
            # Larger sharpness = smoother transition (more deviation from min)
            S = B = 0.5
            L_sharp = smooth_min_limiter(S, B, 1e-12)
            L_smooth = smooth_min_limiter(S, B, 0.1)
            # Sharp should be closer to B than smooth
            @test abs(L_sharp - B) < abs(L_smooth - B)
        end

        @testset "type stability" begin
            @test eltype(smooth_min_limiter(Float32(0.5), Float32(0.3))) == Float32
            @test eltype(smooth_min_limiter(Float64(0.5), Float64(0.3))) == Float64
        end
    end

    @testset "smooth_tendency_limiter()" begin
        dt = 1.0

        @testset "positive tendency limited by source" begin
            # Source (q_source) limits how much can grow
            # S = 0.1, q_source = 0.05 → limit = 0.05/dt/3 ≈ 0.0167
            L = smooth_tendency_limiter(0.1, 0.05, 1.0, dt)
            @test L > 0.0
            @test L < 0.1  # Should be limited
        end

        @testset "positive tendency pass-through" begin
            # Large source, small tendency → passes through
            L = smooth_tendency_limiter(0.001, 1.0, 1.0, dt)
            @test L ≈ 0.001 atol = 1e-6
        end

        @testset "negative tendency limited by sink" begin
            # Sink (q_sink) limits how much can be removed
            # S = -0.1, q_sink = 0.01 → limit = 0.01/dt/3 ≈ 0.0033
            L = smooth_tendency_limiter(-0.1, 1.0, 0.01, dt)
            @test L < 0.0
            @test abs(L) < 0.1  # Should be limited
        end

        @testset "negative tendency pass-through" begin
            # Large sink, small tendency → passes through
            L = smooth_tendency_limiter(-0.001, 1.0, 1.0, dt)
            @test L ≈ -0.001 atol = 1e-6
        end

        @testset "zero tendency" begin
            @test smooth_tendency_limiter(0.0, 0.05, 0.05, dt) ≈ 0.0 atol = 1e-9
        end

        @testset "zero source prevents positive tendency" begin
            L = smooth_tendency_limiter(0.1, 0.0, 0.05, dt)
            @test L == 0.0
        end

        @testset "zero sink prevents negative tendency" begin
            L = smooth_tendency_limiter(-0.1, 0.05, 0.0, dt)
            @test L == 0.0
        end

        @testset "type stability" begin
            @test eltype(
                smooth_tendency_limiter(
                    Float32(0.5),
                    Float32(0.3),
                    Float32(0.3),
                    Float32(1.0),
                ),
            ) == Float32
            @test eltype(
                smooth_tendency_limiter(
                    Float64(0.5),
                    Float64(0.3),
                    Float64(0.3),
                    Float64(1.0),
                ),
            ) == Float64
        end
    end

    @testset "smooth_tendency_limiter() condensation use case" begin
        dt = 1.0

        @testset "condensation (S > 0)" begin
            # Condensation limited by supersaturation
            S = 0.01  # condensation rate
            q_sat_excess = 0.005  # supersaturated by 5 g/kg
            q_cond = 0.001  # existing condensate

            S_lim = smooth_tendency_limiter(S, max(q_sat_excess, 0.0), q_cond, dt, 2)

            # Should be positive (condensation)
            @test S_lim > 0.0

            # Should be limited (can't condense more than supersaturation)
            @test S_lim <= S

            # Should respect supersaturation limit
            @test S_lim <= q_sat_excess / dt + eps(Float64)
        end

        @testset "evaporation (S < 0)" begin
            # Evaporation limited by available condensate and subsaturation
            S = -0.01  # evaporation rate
            q_sat_excess = -0.005  # subsaturated
            q_cond = 0.002  # available condensate

            S_lim = smooth_tendency_limiter(S, max(q_sat_excess, 0.0), q_cond, dt, 2)

            # Should be negative (evaporation)
            @test S_lim < 0.0

            # Should respect condensate limit
            @test abs(S_lim) <= q_cond / dt + eps(Float64)
        end

        @testset "sign consistency" begin
            dt = 1.0

            # Positive tendency → positive result
            S_lim = smooth_tendency_limiter(0.01, 0.01, 0.01, dt, 2)
            @test S_lim >= 0.0

            # Negative tendency → negative result
            S_lim = smooth_tendency_limiter(-0.01, max(-0.01, 0.0), 0.01, dt, 2)
            @test S_lim <= 0.0

            # Zero tendency → zero result
            S_lim = smooth_tendency_limiter(0.0, 0.01, 0.01, dt, 2)
            @test S_lim == 0.0
        end

        @testset "physical edge cases" begin
            dt = 1.0

            # No supersaturation → no condensation
            S_lim = smooth_tendency_limiter(0.01, 0.0, 0.01, dt, 2)
            @test S_lim == 0.0

            # No condensate → no evaporation
            S_lim = smooth_tendency_limiter(-0.01, max(-0.01, 0.0), 0.0, dt, 2)
            @test S_lim == 0.0
        end

        @testset "type stability" begin
            @test eltype(
                smooth_tendency_limiter(
                    Float32(0.01), Float32(0.01), Float32(0.01), Float32(1.0), 2,
                ),
            ) == Float32
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
            Sq = -0.01  # Would deplete q in 0.1s
            Sn = -1e7   # Would take 1000s to deplete n

            f = coupled_sink_limit_factor(Sq, Sn, q, n, dt)
            # Mass bound = q/(dt*3) = 0.00033, |Sq| = 0.01
            # f_mass = 0.00033/0.01 ≈ 0.033
            # Number bound = n/(dt*3) ≈ 3.3e9, |Sn| = 1e7 → no limiting (f_n = 1)
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

        @testset "type stability" begin
            f = coupled_sink_limit_factor(
                Float32(-0.01), Float32(-1e8),
                Float32(0.001), Float32(1e8),
                Float32(1.0),
            )
            @test eltype(f) == Float32
        end
    end

    @testset "zero allocations (scalar)" begin
        # These must be zero-allocation for GPU safety.
        # Any allocations indicate boxing of return values or type instability.

        @testset "coupled_sink_limit_factor" begin
            # Warmup
            coupled_sink_limit_factor(-0.01, -1e8, 0.001, 1e8, 1.0)
            allocs = @allocated coupled_sink_limit_factor(-0.01, -1e8, 0.001, 1e8, 1.0)
            @test allocs == 0
        end


    end
end
