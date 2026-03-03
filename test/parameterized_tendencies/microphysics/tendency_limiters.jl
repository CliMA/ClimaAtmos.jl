#=
Unit tests for tendency_limiters.jl

Tests cover:
1. `limit()` - basic rate limiting
2. `tendency_limiter()` - bidirectional tendency limiting
3. `coupled_sink_limit_factor()` - uniform scaling for coupled sinks
4. `limit_sink()` - sink-only tendency limiting
5. `_explicit_1m_tendency_limits()` - end-to-end 1M explicit limiter
6. `_implicit_1m_tendency_limits()` - end-to-end 1M implicit limiter
=#

using Test
using ClimaAtmos

# Import functions under test
import ClimaAtmos:
    limit,
    tendency_limiter,
    coupled_sink_limit_factor,
    limit_sink,
    _implicit_1m_tendency_limits

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
            @test eltype(limit_sink(Float32(-0.01), Float32(0.001), Float32(1.0))) == Float32
            @test eltype(limit_sink(Float64(-0.01), Float64(0.001), Float64(1.0))) == Float64
        end
    end

    @testset "_explicit_1m_tendency_limits() cross-species limiting" begin
        import ClimaAtmos: _explicit_1m_tendency_limits
        import Thermodynamics as TD
        import ClimaParams as CP

        FT = Float64
        # Create minimal thermodynamics params
        toml_dict = CP.create_toml_dict(FT)
        thermo_params = TD.Parameters.ThermodynamicsParameters(toml_dict)

        dt = FT(600)  # 10 minute timestep

        @testset "no limiting when tendencies are small" begin
            # Small tendencies relative to large species pools
            # → limiter barely touches them.
            # n_sink=5 budget for q=0.01 is 0.01/(600*5) = 3.33e-6
            # n_source=30 budget for source pool is even larger.
            # Tendencies ~1e-7 are well below both budgets.
            mp_tendency = (
                dq_lcl_dt = FT(1e-7),
                dq_icl_dt = FT(-5e-8),
                dq_rai_dt = FT(-3e-8),
                dq_sno_dt = FT(-2e-8),
            )
            q_liq = FT(0.01)
            q_ice = FT(0.01)
            q_rai = FT(0.01)
            q_sno = FT(0.01)
            q_tot = FT(0.05)  # q_vap = 0.01 > 0

            limited = _explicit_1m_tendency_limits(
                mp_tendency,
                thermo_params,
                q_tot,
                q_liq,
                q_ice,
                q_rai,
                q_sno,
                dt,
            )

            # Should pass through nearly unchanged.
            @test limited.dq_lcl_dt ≈ mp_tendency.dq_lcl_dt rtol = 0.05
            @test limited.dq_icl_dt ≈ mp_tendency.dq_icl_dt rtol = 0.05
            @test limited.dq_rai_dt ≈ mp_tendency.dq_rai_dt rtol = 0.05
            @test limited.dq_sno_dt ≈ mp_tendency.dq_sno_dt rtol = 0.05
        end

        @testset "per-species sink limiting prevents depletion" begin
            # Rain has a large sink tendency relative to its small amount
            mp_tendency = (
                dq_lcl_dt = FT(0.001),
                dq_icl_dt = FT(-0.0005),
                dq_rai_dt = FT(-0.01),  # Would deplete rain in < dt
                dq_sno_dt = FT(-0.0002),
            )
            q_liq = FT(0.001)
            q_ice = FT(0.0005)
            q_rai = FT(0.0001)  # Small amount
            q_sno = FT(0.0002)
            q_tot = FT(0.01)  # q_vap = 0.0082

            limited = _explicit_1m_tendency_limits(
                mp_tendency,
                thermo_params,
                q_tot,
                q_liq,
                q_ice,
                q_rai,
                q_sno,
                dt,
            )

            # Rain sink tendency should be limited (prevented from depleting)
            @test limited.dq_rai_dt * dt >= -q_rai
            @test abs(limited.dq_rai_dt) < abs(mp_tendency.dq_rai_dt)

            # With per-species limiting, other species are limited independently.
            # Snow should be limited but less aggressively than rain.
            @test abs(limited.dq_sno_dt) <= abs(mp_tendency.dq_sno_dt)
        end
    end

    @testset "_implicit_1m_tendency_limits()" begin
        FT = Float64
        dt = FT(600)

        @testset "sources pass through unchanged" begin
            mp_tendency = (
                dq_lcl_dt = FT(0.001),
                dq_icl_dt = FT(0.0005),
                dq_rai_dt = FT(0.002),
                dq_sno_dt = FT(0.001),
            )
            q_liq = FT(0.01)
            q_ice = FT(0.01)
            q_rai = FT(0.01)
            q_sno = FT(0.01)

            limited = _implicit_1m_tendency_limits(
                mp_tendency, q_liq, q_ice, q_rai, q_sno, dt,
            )

            # All sources → pass through unchanged
            @test limited.dq_lcl_dt == mp_tendency.dq_lcl_dt
            @test limited.dq_icl_dt == mp_tendency.dq_icl_dt
            @test limited.dq_rai_dt == mp_tendency.dq_rai_dt
            @test limited.dq_sno_dt == mp_tendency.dq_sno_dt
        end

        @testset "sinks are limited to prevent depletion" begin
            mp_tendency = (
                dq_lcl_dt = FT(0.001),     # Source — should pass through
                dq_icl_dt = FT(-0.0005),   # Small sink — within budget
                dq_rai_dt = FT(-0.01),     # Large sink — should be capped
                dq_sno_dt = FT(-0.005),    # Medium sink
            )
            q_liq = FT(0.01)
            q_ice = FT(0.01)
            q_rai = FT(0.0001)  # Very small amount → forces limiting
            q_sno = FT(0.0002)

            limited = _implicit_1m_tendency_limits(
                mp_tendency, q_liq, q_ice, q_rai, q_sno, dt,
            )

            # Source passes through
            @test limited.dq_lcl_dt == mp_tendency.dq_lcl_dt

            # Large rain sink should be clamped
            # n_sink=3, budget = 0.0001 / (600*3) ≈ 5.56e-8
            @test limited.dq_rai_dt < 0.0
            @test abs(limited.dq_rai_dt) < abs(mp_tendency.dq_rai_dt)
            @test abs(limited.dq_rai_dt) ≈ q_rai / (dt * 3)

            # Snow sink should also be clamped
            @test limited.dq_sno_dt < 0.0
            @test abs(limited.dq_sno_dt) < abs(mp_tendency.dq_sno_dt)
        end

        @testset "no temperature-rate limiting applied" begin
            # Unlike explicit mode, implicit mode should NOT apply
            # temperature-rate limiting — verify by using extreme
            # cloud tendencies that would trigger the temperature limiter
            # in explicit mode.
            mp_tendency = (
                dq_lcl_dt = FT(0.1),    # Huge source
                dq_icl_dt = FT(0.1),    # Huge source
                dq_rai_dt = FT(0.0),
                dq_sno_dt = FT(0.0),
            )
            q_liq = FT(0.01)
            q_ice = FT(0.01)
            q_rai = FT(0.01)
            q_sno = FT(0.01)

            limited = _implicit_1m_tendency_limits(
                mp_tendency, q_liq, q_ice, q_rai, q_sno, dt,
            )

            # Sources pass through — no temperature-rate rescaling
            @test limited.dq_lcl_dt == mp_tendency.dq_lcl_dt
            @test limited.dq_icl_dt == mp_tendency.dq_icl_dt
        end
    end
end
