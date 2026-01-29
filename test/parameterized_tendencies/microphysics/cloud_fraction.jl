#=
Unit tests for cloud_fraction.jl

Tests cover:
1. compute_cloud_fraction_sd - Sommeria-Deardorff moment-matching cloud fraction
=#

using Test
using ClimaAtmos
import ClimaAtmos as CA

import Thermodynamics as TD
import ClimaParams as CP

@testset "Cloud Fraction" begin

    @testset "compute_cloud_fraction_sd" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                T = FT(280.0)
                ρ = FT(1.0)

                @testset "No condensate → zero cloud fraction" begin
                    cf = CA.compute_cloud_fraction_sd(
                        thp, T, ρ,
                        FT(0), FT(0),         # q_liq, q_ice
                        FT(1.0), FT(1e-6), FT(0),  # T'T', q'q', T'q'
                    )
                    @test cf == FT(0)
                end

                @testset "Liquid condensate → positive cloud fraction" begin
                    cf = CA.compute_cloud_fraction_sd(
                        thp, T, ρ,
                        FT(1e-3), FT(0),       # q_liq present
                        FT(1.0), FT(1e-6), FT(0),
                    )
                    @test cf > FT(0)
                    @test cf <= FT(1)
                end

                @testset "Ice condensate → positive cloud fraction" begin
                    T_cold = FT(240.0)
                    cf = CA.compute_cloud_fraction_sd(
                        thp, T_cold, ρ,
                        FT(0), FT(1e-3),       # q_ice present
                        FT(1.0), FT(1e-6), FT(0),
                    )
                    @test cf > FT(0)
                    @test cf <= FT(1)
                end

                @testset "Large condensate, small variance → cf ≈ 1" begin
                    cf = CA.compute_cloud_fraction_sd(
                        thp, T, ρ,
                        FT(1e-2), FT(0),       # large q_liq
                        FT(0.01), FT(1e-10), FT(0),  # small variance
                    )
                    @test cf > FT(0.99)
                end

                @testset "Zero variance, nonzero condensate → cf = 1" begin
                    cf = CA.compute_cloud_fraction_sd(
                        thp, T, ρ,
                        FT(1e-3), FT(0),
                        FT(0), FT(0), FT(0),
                    )
                    @test cf > FT(0.9)
                end

                @testset "Both liquid and ice → max overlap" begin
                    cf_both = CA.compute_cloud_fraction_sd(
                        thp, FT(260.0), ρ,
                        FT(1e-3), FT(5e-4),
                        FT(1.0), FT(1e-6), FT(0),
                    )
                    cf_liq = CA.compute_cloud_fraction_sd(
                        thp, FT(260.0), ρ,
                        FT(1e-3), FT(0),
                        FT(1.0), FT(1e-6), FT(0),
                    )
                    # Max overlap: combined cf ≥ liquid-only cf
                    @test cf_both >= cf_liq - eps(FT)
                end

                @testset "Type stability" begin
                    cf = CA.compute_cloud_fraction_sd(
                        thp, T, ρ,
                        FT(1e-3), FT(0),
                        FT(1.0), FT(1e-6), FT(0),
                    )
                    @test cf isa FT
                end
            end
        end
    end

    @testset "Allocation Test" begin
        FT = Float64
        toml_dict = CP.create_toml_dict(FT)
        thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

        T = FT(280.0)
        ρ = FT(1.0)

        # Warm up
        _ = CA.compute_cloud_fraction_sd(
            thp, T, ρ,
            FT(1e-3), FT(0),
            FT(1.0), FT(1e-6), FT(0),
        )
        allocs = @allocated CA.compute_cloud_fraction_sd(
            thp, T, ρ,
            FT(1e-3), FT(0),
            FT(1.0), FT(1e-6), FT(0),
        )
        @test allocs == 0
    end

end
