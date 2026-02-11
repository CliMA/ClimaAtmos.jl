#=
Unit tests for tendency_limiters.jl

Tests cover:
1. `limit()` - basic rate limiting
2. `smooth_min_limiter()` - smooth minimum approximation
3. `smooth_tendency_limiter()` - bidirectional tendency limiting
4. `coupled_sink_limit_factor()` - uniform scaling for coupled sinks
5. `sink_scale_factor()` - scale factor for uniform sink limiting
6. `apply_1m_tendency_limits()` - end-to-end 1M limiter
=#

using Test
using ClimaAtmos

# Import functions under test
import ClimaAtmos:
    limit,
    smooth_min_limiter,
    smooth_tendency_limiter,
    coupled_sink_limit_factor,
    sink_scale_factor

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


    @testset "smooth_min_limiter()" begin
        @testset "bounded output" begin
            # L ≤ min(S, B) for S, B ≥ 0
            for S in [0.001, 0.01, 0.1, 1.0]
                for B in [0.001, 0.01, 0.1, 1.0]
                    L = smooth_min_limiter(S, B)
                    @test L >= 0.0 || isapprox(L, 0.0, atol = 1e-7)
                    @test L <= min(S, B) + 1e-7
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
            @test L ≈ B atol = 1e-7
        end

        @testset "symmetry" begin
            # smooth_min_limiter(S, B) = smooth_min_limiter(B, S)
            @test smooth_min_limiter(0.3, 0.7) ≈ smooth_min_limiter(0.7, 0.3)
            @test smooth_min_limiter(0.1, 0.9) ≈ smooth_min_limiter(0.9, 0.1)
        end

        @testset "zero inputs" begin
            # smooth_min_limiter returns -ε/2 ≈ 0 when both inputs are zero
            @test smooth_min_limiter(0.0, 0.0) ≈ 0.0 atol = 1e-7
            @test smooth_min_limiter(0.0, 1.0) ≈ 0.0 atol = 1e-7
            @test smooth_min_limiter(1.0, 0.0) ≈ 0.0 atol = 1e-7
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
        @testset "positive tendency limited" begin
            # Source bound limits how much can grow
            L = smooth_tendency_limiter(0.1, 0.02, 1.0)
            @test L > 0.0
            @test L < 0.1  # Should be limited
        end

        @testset "positive tendency pass-through" begin
            # Large bound, small tendency → passes through
            L = smooth_tendency_limiter(0.001, 1.0, 1.0)
            @test L ≈ 0.001 atol = 1e-6
        end

        @testset "negative tendency limited" begin
            # Negative bound limits how much can be removed
            L = smooth_tendency_limiter(-0.1, 1.0, 0.01)
            @test L < 0.0
            @test abs(L) < 0.1  # Should be limited
        end

        @testset "negative tendency pass-through" begin
            # Large bound, small tendency → passes through
            L = smooth_tendency_limiter(-0.001, 1.0, 1.0)
            @test L ≈ -0.001 atol = 1e-6
        end

        @testset "zero tendency" begin
            @test smooth_tendency_limiter(0.0, 0.05, 0.05) ≈ 0.0 atol = 1e-9
        end

        @testset "zero positive bound prevents positive tendency" begin
            L = smooth_tendency_limiter(0.1, 0.0, 0.05)
            @test L ≈ 0.0 atol = 1e-7
        end

        @testset "zero negative bound prevents negative tendency" begin
            L = smooth_tendency_limiter(-0.1, 0.05, 0.0)
            @test L ≈ 0.0 atol = 1e-7
        end

        @testset "type stability" begin
            @test eltype(
                smooth_tendency_limiter(
                    Float32(0.5),
                    Float32(0.3),
                    Float32(0.3),
                ),
            ) == Float32
            @test eltype(
                smooth_tendency_limiter(
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

    @testset "sink_scale_factor()" begin
        dt = 1.0

        @testset "sources → no limiting" begin
            @test sink_scale_factor(0.01, 0.001, dt, 3) == 1.0
            @test sink_scale_factor(0.0, 0.001, dt, 3) == 1.0
        end

        @testset "small sink → no limiting" begin
            # Sink is well within budget
            q = 0.01
            S = -0.0001  # Tiny sink
            @test sink_scale_factor(S, q, dt, 3) == 1.0
        end

        @testset "large sink → limiting" begin
            # Sink exceeds budget
            q = 0.001
            S = -0.01  # Would deplete q in 0.1s
            n = 3
            f = sink_scale_factor(S, q, dt, n)
            bound = q / dt / n  # 0.000333
            @test f ≈ bound / abs(S) rtol = 1e-10
            @test f < 1.0
            @test f > 0.0
        end

        @testset "negative q → zero bound → zero factor" begin
            # When q is negative, bound = 0, so any sink gets factor 0
            f = sink_scale_factor(-0.01, -0.001, dt, 3)
            @test f == 0.0
        end

        @testset "zero q → zero factor for sinks" begin
            f = sink_scale_factor(-0.01, 0.0, dt, 3)
            @test f == 0.0
        end

        @testset "type stability" begin
            @test eltype(
                sink_scale_factor(Float32(-0.01), Float32(0.001), Float32(1.0), 3),
            ) == Float32
            @test eltype(
                sink_scale_factor(Float64(-0.01), Float64(0.001), Float64(1.0), 3),
            ) == Float64
        end
    end

    @testset "apply_1m_tendency_limits() uniform scaling" begin
        import ClimaAtmos: apply_1m_tendency_limits
        import Thermodynamics as TD
        import ClimaParams as CP

        FT = Float64
        # Create minimal thermodynamics params
        toml_dict = CP.create_toml_dict(FT)
        thermo_params = TD.Parameters.ThermodynamicsParameters(toml_dict)

        dt = FT(600)  # 10 minute timestep

        @testset "no limiting when tendencies are small" begin
            # Tendencies must be small relative to sink budget: q/(dt*n_sink)
            # With n_sink=20, dt=600: budget for q_sno=0.01 is 0.01/(600*20)=8.3e-7
            mp_result = (
                dq_lcl_dt = FT(1e-8),
                dq_icl_dt = FT(-5e-9),
                dq_rai_dt = FT(-3e-9),
                dq_sno_dt = FT(-2e-9),
            )
            q_tot = FT(0.01)
            q_liq = FT(0.01)
            q_ice = FT(0.01)
            q_rai = FT(0.01)
            q_sno = FT(0.01)

            limited = apply_1m_tendency_limits(
                mp_result,
                thermo_params,
                q_tot,
                q_liq,
                q_ice,
                q_rai,
                q_sno,
                dt,
            )

            # Should pass through unchanged
            @test limited.dq_lcl_dt ≈ mp_result.dq_lcl_dt rtol = 1e-10
            @test limited.dq_icl_dt ≈ mp_result.dq_icl_dt rtol = 1e-10
            @test limited.dq_rai_dt ≈ mp_result.dq_rai_dt rtol = 1e-10
            @test limited.dq_sno_dt ≈ mp_result.dq_sno_dt rtol = 1e-10
        end

        @testset "uniform sink limiting prevents depletion" begin
            # One species would be depleted without limiting
            mp_result = (
                dq_lcl_dt = FT(0.001),
                dq_icl_dt = FT(-0.0005),
                dq_rai_dt = FT(-0.01),  # Would deplete rain in < dt
                dq_sno_dt = FT(-0.0002),
            )
            q_tot = FT(0.01)
            q_liq = FT(0.001)
            q_ice = FT(0.0005)
            q_rai = FT(0.0001)  # Small amount
            q_sno = FT(0.0002)

            limited = apply_1m_tendency_limits(
                mp_result,
                thermo_params,
                q_tot,
                q_liq,
                q_ice,
                q_rai,
                q_sno,
                dt,
            )

            # All tendencies should be scaled uniformly
            # Check that rain won't be depleted
            @test limited.dq_rai_dt * dt >= -q_rai
            @test abs(limited.dq_rai_dt) < abs(mp_result.dq_rai_dt)

            # Uniform scaling: all tendencies scaled by same factor
            expected_factor = limited.dq_rai_dt / mp_result.dq_rai_dt
            @test limited.dq_lcl_dt / mp_result.dq_lcl_dt ≈ expected_factor rtol = 1e-6
            @test limited.dq_icl_dt / mp_result.dq_icl_dt ≈ expected_factor rtol = 1e-6
            @test limited.dq_sno_dt / mp_result.dq_sno_dt ≈ expected_factor rtol = 1e-6
        end

        @testset "temperature-rate limiting" begin
            # Large phase change tendencies → temperature limit kicks in
            mp_result = (
                dq_lcl_dt = FT(0.01),   # Large condensation
                dq_icl_dt = FT(-0.001),
                dq_rai_dt = FT(-0.003),
                dq_sno_dt = FT(-0.001),
            )
            q_tot = FT(0.02)
            q_liq = FT(0.001)
            q_ice = FT(0.001)
            q_rai = FT(0.005)
            q_sno = FT(0.002)

            limited = apply_1m_tendency_limits(
                mp_result,
                thermo_params,
                q_tot,
                q_liq,
                q_ice,
                q_rai,
                q_sno,
                dt,
            )

            # Tendencies should be limited
            @test abs(limited.dq_lcl_dt) < abs(mp_result.dq_lcl_dt)
        end
    end
end
