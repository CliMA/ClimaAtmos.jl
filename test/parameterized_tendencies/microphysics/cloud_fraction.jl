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
                    @test cf > FT(0.99)
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
                    @test cf_both >= cf_liq
                end

                @testset "Type stability" begin
                    cf = CA.compute_cloud_fraction_sd(
                        thp, T, ρ,
                        FT(1e-3), FT(0),
                        FT(1.0), FT(1e-6), FT(0),
                    )
                    @test cf isa FT
                end

                @testset "Cloud fraction increases with T-q correlation" begin
                    # Physical reasoning: positive corr(T',q') means T and q
                    # perturbations partially cancel in the saturation deficit
                    # s = q − q_sat(T).  The PDF width σ_s² = σ_q² + b²σ_T²
                    # − 2b·corr·σ_T·σ_q shrinks when corr increases (b > 0),
                    # so Q̂ = q_cond / σ_s grows ⇒ cf = tanh(π/√6·Q̂) grows.
                    corr_vals = FT[-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]

                    # Liquid-only case
                    cf_liq = [
                        CA.compute_cloud_fraction_sd(
                            thp, T, ρ,
                            FT(1e-3), FT(0),
                            FT(1.0), FT(1e-6), c,
                        ) for c in corr_vals
                    ]
                    for i in 2:length(cf_liq)
                        @test cf_liq[i] > cf_liq[i - 1]
                    end

                    # Ice-only case (cold temperature)
                    T_cold = FT(240.0)
                    cf_ice = [
                        CA.compute_cloud_fraction_sd(
                            thp, T_cold, ρ,
                            FT(0), FT(1e-3),
                            FT(1.0), FT(1e-6), c,
                        ) for c in corr_vals
                    ]
                    for i in 2:length(cf_ice)
                        @test cf_ice[i] > cf_ice[i - 1]
                    end
                end
            end
        end
    end

end
