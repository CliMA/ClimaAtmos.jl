#=
Unit tests for tendency_limiters.jl

Tests cover:
1. `limit()` - basic rate limiting
2. `tendency_limiter()` - bidirectional tendency limiting
3. `coupled_sink_limit_factor()` - uniform scaling for coupled sinks
4. `apply_1m_tendency_limits()` - end-to-end 1M limiter
=#

using Test
using ClimaAtmos

# Import functions under test
import ClimaAtmos:
    limit,
    tendency_limiter,
    coupled_sink_limit_factor,
    microphysics_jacobian_diagonal,
    ϵ_numerics

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

    @testset "apply_1m_tendency_limits() cross-species limiting" begin
        import ClimaAtmos: apply_1m_tendency_limits
        import Thermodynamics as TD
        import ClimaParams as CP

        FT = Float64
        # Create minimal thermodynamics params
        toml_dict = CP.create_toml_dict(FT)
        thermo_params = TD.Parameters.ThermodynamicsParameters(toml_dict)

        dt = FT(600)  # 10 minute timestep

        @testset "no limiting when tendencies are small" begin
            # Small tendencies relative to large species pools
            # → smooth limiter barely touches them.
            # n_sink=5 budget for q=0.01 is 0.01/(600*5) = 3.3e-6
            # Tendencies ~1e-7 are ~33× below budget.
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

            limited = apply_1m_tendency_limits(
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
            # The smooth limiter introduces O(ε²/S) corrections,
            # so we allow small relative deviation.
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

            limited = apply_1m_tendency_limits(
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

        @testset "temperature-rate limiting" begin
            # Large condensation → combined temperature limit kicks in.
            # Rain/snow tendencies are tiny relative to their n_sink=5 budgets
            # so mass limiting doesn't interfere with them.
            # Budget for q_rai=0.01: 0.01/(600*5) = 3.3e-6
            mp_tendency = (
                dq_lcl_dt = FT(0.01),   # Large condensation
                dq_icl_dt = FT(-0.001),
                dq_rai_dt = FT(-1e-8),  # Tiny, well within budget
                dq_sno_dt = FT(-1e-8),  # Tiny, well within budget
            )
            q_liq = FT(0.01)
            q_ice = FT(0.01)
            q_rai = FT(0.01)
            q_sno = FT(0.01)
            q_tot = FT(0.05)  # q_vap = 0.01

            limited = apply_1m_tendency_limits(
                mp_tendency,
                thermo_params,
                q_tot,
                q_liq,
                q_ice,
                q_rai,
                q_sno,
                dt,
            )

            # Combined temperature limiting: only condensate (lcl, icl) is scaled
            @test abs(limited.dq_lcl_dt) < abs(mp_tendency.dq_lcl_dt)

            # Rain and snow are NOT scaled by the temperature limiter
            @test limited.dq_rai_dt ≈ mp_tendency.dq_rai_dt rtol = 0.1
            @test limited.dq_sno_dt ≈ mp_tendency.dq_sno_dt rtol = 0.1
        end
    end
end

@testset "microphysics_jacobian_diagonal" begin

    @testset "sink returns Sq/q" begin
        Sq = -0.01
        q  = 0.001
        @test microphysics_jacobian_diagonal(Sq, q) ≈ Sq / q
    end

    @testset "source returns Sq/q" begin
        Sq = 0.005
        q  = 0.002
        @test microphysics_jacobian_diagonal(Sq, q) ≈ Sq / q
    end

    @testset "near-zero q uses ϵ_numerics guard" begin
        Sq = -1e-10
        q  = 0.0
        result = microphysics_jacobian_diagonal(Sq, q)
        @test isfinite(result)
        @test result ≈ Sq / ϵ_numerics(Float64)
    end

    @testset "zero tendency returns zero" begin
        @test microphysics_jacobian_diagonal(0.0, 0.01) == 0.0
        @test microphysics_jacobian_diagonal(0.0, 0.0) == 0.0
    end

    @testset "Float32 type stability" begin
        result = microphysics_jacobian_diagonal(Float32(-0.01), Float32(0.001))
        @test result isa Float32
        @test result ≈ Float32(-0.01) / Float32(0.001)
    end
end
