#=
Unit tests for microphysics_wrappers.jl
Tests wrapper functions for physical correctness, sign convention, and type stability.

Sign convention: all microphysics tendencies representing SINKS should be ≤ 0.
=#

using Test
using ClimaAtmos

import Thermodynamics as TD
import CloudMicrophysics as CM
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.Microphysics0M as CM0
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

# Import limiters
import ClimaAtmos: limit_sink

@testset "Microphysics Wrappers" begin

    @testset "BMT 0M sign convention" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics0MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                dt = FT(60.0)  # 1 minute timestep

                @testset "dq_tot_dt is always ≤ 0 (sink)" begin
                    # Condensate present → precipitation removes water (sink)
                    T = FT(280.0)
                    q_liq = FT(0.001)
                    q_ice = FT(0.0005)

                    result = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, T, q_liq, q_ice,
                    )
                    @test result.dq_tot_dt <= FT(0)
                    @test isfinite(result.dq_tot_dt)
                    @test isfinite(result.e_int_precip)
                end

                @testset "dq_tot_dt is zero when no condensate" begin
                    result = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, FT(280.0), FT(0), FT(0),
                    )
                    @test result.dq_tot_dt == FT(0)
                end

                @testset "limit_sink preserves sign from BMT" begin
                    T = FT(280.0)
                    q_tot = FT(0.015)
                    q_liq = FT(0.001)
                    q_ice = FT(0.0005)

                    bmt_result = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, T, q_liq, q_ice,
                    )
                    limited = limit_sink(bmt_result.dq_tot_dt, q_tot, dt, 1)

                    # limit_sink should keep the tendency negative (sink)
                    @test limited <= FT(0)

                    # Should not remove more water than available
                    @test limited * dt >= -q_tot

                    # Should be finite
                    @test isfinite(limited)
                end

                @testset "limit_sink with tiny q_tot" begin
                    # Edge case: very small q_tot should limit the magnitude
                    T = FT(280.0)
                    q_tot = FT(1e-6)   # Very small amount
                    q_liq = FT(0.01)   # Large condensate (edge case)
                    q_ice = FT(0)

                    bmt_result = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, T, q_liq, q_ice,
                    )
                    limited = limit_sink(bmt_result.dq_tot_dt, q_tot, dt, 1)

                    # Should still be a sink
                    @test limited <= FT(0)

                    # Should be limited to available water
                    @test limited * dt >= -q_tot
                end

                @testset "type stability" begin
                    result = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, FT(280.0), FT(0.001), FT(0.0005),
                    )
                    @test typeof(result.dq_tot_dt) == FT
                    @test typeof(result.e_int_precip) == FT

                    limited = limit_sink(result.dq_tot_dt, FT(0.01), FT(60.0), 1)
                    @test typeof(limited) == FT
                end
            end
        end
    end

    @testset "BMT 1M sign convention" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict; with_2M_autoconv = true)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                ρ = FT(1.0)
                T = FT(280.0)
                q_tot = FT(0.015)
                q_liq = FT(0.001)
                q_ice = FT(0.0005)
                q_rai = FT(0.0001)
                q_sno = FT(0.00005)
                dt = FT(60.0)

                result = BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics1Moment(),
                    mp, thp, ρ, T,
                    q_tot, q_liq, q_ice, q_rai, q_sno,
                )

                @testset "return type" begin
                    @test haskey(result, :dq_lcl_dt)
                    @test haskey(result, :dq_icl_dt)
                    @test haskey(result, :dq_rai_dt)
                    @test haskey(result, :dq_sno_dt)
                end

                @testset "finite values" begin
                    @test isfinite(result.dq_lcl_dt)
                    @test isfinite(result.dq_icl_dt)
                    @test isfinite(result.dq_rai_dt)
                    @test isfinite(result.dq_sno_dt)
                end

                @testset "limit_sink preserves sign" begin
                    # All tendency fields, when sinks, should remain ≤ 0 after limiting
                    for (field, q) in [
                        (:dq_lcl_dt, q_liq),
                        (:dq_icl_dt, q_ice),
                        (:dq_rai_dt, q_rai),
                        (:dq_sno_dt, q_sno),
                    ]
                        S = getfield(result, field)
                        limited = limit_sink(S, q, dt)
                        if S < FT(0)
                            @test limited <= FT(0)
                            @test limited * dt >= -q
                        else
                            @test limited == S  # sources pass through
                        end
                        @test isfinite(limited)
                    end
                end

                @testset "type stability" begin
                    @test typeof(result.dq_lcl_dt) == FT
                    @test typeof(result.dq_icl_dt) == FT
                    @test typeof(result.dq_rai_dt) == FT
                    @test typeof(result.dq_sno_dt) == FT
                end
            end
        end
    end

    @testset "e_tot_0M_precipitation_sources_helper" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                @testset "Warm conditions (all liquid)" begin
                    T = FT(290.0)
                    q_liq = FT(0.001)
                    q_ice = FT(0)
                    Φ = FT(1000.0)

                    energy = ClimaAtmos.e_tot_0M_precipitation_sources_helper(
                        thp, T, q_liq, q_ice, Φ,
                    )

                    @test isfinite(energy)
                    I_liq = TD.internal_energy_liquid(thp, T)
                    @test energy ≈ I_liq + Φ rtol = FT(1e-5)
                end

                @testset "Cold conditions (all ice)" begin
                    T = FT(240.0)
                    q_liq = FT(0)
                    q_ice = FT(0.001)
                    Φ = FT(5000.0)

                    energy = ClimaAtmos.e_tot_0M_precipitation_sources_helper(
                        thp, T, q_liq, q_ice, Φ,
                    )

                    @test isfinite(energy)
                    I_ice = TD.internal_energy_ice(thp, T)
                    @test energy ≈ I_ice + Φ rtol = FT(1e-5)
                end

                @testset "Type stability" begin
                    energy = ClimaAtmos.e_tot_0M_precipitation_sources_helper(
                        thp, FT(280.0), FT(0.001), FT(0.0005), FT(1000.0),
                    )
                    @test typeof(energy) == FT
                end
            end
        end
    end

    @testset "Allocation Test" begin
        FT = Float64
        toml_dict = CP.create_toml_dict(FT)
        mp = CMP.Microphysics0MParams(toml_dict)
        thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

        T = FT(280.0)
        q_liq = FT(0.001)
        q_ice = FT(0.0005)
        Φ = FT(1000.0)

        # Warm-up
        _ = BMT.bulk_microphysics_tendencies(
            BMT.Microphysics0Moment(), mp, thp, T, q_liq, q_ice,
        )
        _ = ClimaAtmos.e_tot_0M_precipitation_sources_helper(thp, T, q_liq, q_ice, Φ)
        _ = limit_sink(FT(-0.01), FT(0.01), FT(60.0), 1)

        # Test allocations
        allocs_bmt = @allocated BMT.bulk_microphysics_tendencies(
            BMT.Microphysics0Moment(), mp, thp, T, q_liq, q_ice,
        )
        allocs_energy = @allocated ClimaAtmos.e_tot_0M_precipitation_sources_helper(
            thp, T, q_liq, q_ice, Φ,
        )
        allocs_limit = @allocated limit_sink(FT(-0.01), FT(0.01), FT(60.0), 1)

        @test allocs_bmt == 0
        @test allocs_energy == 0
        @test allocs_limit == 0
    end

end
